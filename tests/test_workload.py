from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.agentweave import ensure_mini_sglang_path
from benchmark.agentweave.workload import (
    ConfigError,
    build_workload,
    load_config,
    sample_distribution,
    write_workload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def _config(**overrides):
    config = {
        "model": "Qwen/Qwen3-0.6B",
        "workload": {
            "mode": "synthetic",
            "num_agents": 3,
            "seed": 5,
            "tools": [{"name": "lookup", "required_keys": ["result"]}],
            "duration": {"kind": "fixed", "value_s": 1.5},
            "cold_start": {"kind": "fixed", "value_s": 0.25},
        },
    }
    config["workload"].update(overrides)
    return config["workload"] | {"model": config["model"]}


def test_synthetic_workload_has_required_request_fields():
    requests = build_workload(_config())
    assert len(requests) == 3
    for index, request in enumerate(requests):
        assert request["model"] == "Qwen/Qwen3-0.6B"
        assert request["agent_id"] == f"agent-{index:04d}"
        assert request["messages"][0]["role"] == "system"
        assert request["messages"][1]["role"] == "user"
        assert request["max_steps"] == 4
        # 512, not 256: Qwen3 runs in thinking mode and burns the budget on a
        # <think> block before reaching the tool call.
        assert request["max_tokens"] == 512


def test_tools_use_simulated_backend_with_configured_delays():
    tool = build_workload(_config())[0]["tools"][0]
    assert tool["sandbox_backend"] == "simulated"
    assert float(tool["environment"]["AGENTWEAVE_SIMULATED_EXECUTION_DELAY_S"]) == 1.5
    assert float(tool["environment"]["AGENTWEAVE_SIMULATED_PREPARE_DELAY_S"]) == 0.25


def test_seed_makes_sampling_reproducible():
    spec = _config(duration={"kind": "lognormal", "median_s": 2.0, "sigma": 0.8})
    first = [request["_duration_s"] for request in build_workload(dict(spec), seed=99)]
    second = [request["_duration_s"] for request in build_workload(dict(spec), seed=99)]
    different = [request["_duration_s"] for request in build_workload(dict(spec), seed=100)]
    assert first == second
    assert first != different


def test_distributions_reject_bad_config():
    import random

    rng = random.Random(0)
    with pytest.raises(ConfigError):
        sample_distribution({"kind": "uniform", "low_s": 2.0, "high_s": 1.0}, rng)
    with pytest.raises(ConfigError):
        sample_distribution({"kind": "choice", "values_s": []}, rng)
    with pytest.raises(ConfigError):
        sample_distribution({"kind": "nope"}, rng)
    with pytest.raises(ConfigError):
        sample_distribution({"kind": "fixed"}, rng)


def test_choice_distribution_only_returns_configured_values():
    import random

    rng = random.Random(1)
    values = {
        sample_distribution({"kind": "choice", "values_s": [0.5, 3.0]}, rng) for _ in range(50)
    }
    assert values <= {0.5, 3.0}


def test_write_workload_strips_sampling_keys_into_sidecar(tmp_path):
    destination = write_workload(build_workload(_config()), tmp_path / "workload.jsonl")
    line = json.loads(destination.read_text(encoding="utf-8").splitlines()[0])
    assert "_delay_s" in line
    assert "_duration_s" not in line
    assert "_cold_start_s" not in line

    sidecar = tmp_path / "workload.jsonl.sampling.jsonl"
    sampling = json.loads(sidecar.read_text(encoding="utf-8").splitlines()[0])
    assert sampling["duration_s"] == 1.5
    assert sampling["cold_start_s"] == 0.25


def test_written_workload_is_accepted_by_the_client_loader(tmp_path):
    ensure_mini_sglang_path()
    from minisgl.benchmark.agent import load_workload

    destination = write_workload(build_workload(_config()), tmp_path / "workload.jsonl")
    items = load_workload(destination)
    assert len(items) == 3
    assert items[0].delay_s == 0.0
    assert items[0].request["tools"][0]["name"] == "lookup"


def test_poisson_arrival_staggers_submissions():
    requests = build_workload(
        _config(num_agents=20, arrival={"kind": "poisson", "rate_per_s": 4.0})
    )
    delays = [request["_delay_s"] for request in requests]
    assert any(delay > 0 for delay in delays)
    assert all(delay >= 0 for delay in delays)


def test_structured_mode_prompt_names_the_padded_argument():
    requests = build_workload(
        _config(
            mode="structured",
            argument_padding_tokens=64,
            padded_argument="payload",
            tools=[
                {"name": "search_corpus", "required_keys": ["query", "payload"]},
                {"name": "fetch_document", "required_keys": ["doc_id", "payload"]},
            ],
        )
    )
    prompt = requests[0]["messages"][0]["content"]
    assert "payload" in prompt
    assert "64 words" in prompt


def test_structured_mode_falls_back_to_a_generic_padding_instruction():
    requests = build_workload(
        _config(mode="structured", argument_padding_tokens=32, tools=[{"name": "lookup"}])
    )
    assert "every string argument" in requests[0]["messages"][0]["content"]


def test_synthetic_mode_adds_no_padding_instruction():
    requests = build_workload(_config(argument_padding_tokens=64, padded_argument="payload"))
    assert "words of plain text" not in requests[0]["messages"][0]["content"]


def test_padded_argument_must_be_a_string():
    with pytest.raises(ConfigError):
        build_workload(_config(padded_argument=7))


def test_prompt_padding_grows_the_user_message():
    short = build_workload(_config(prompt_padding_tokens=0))[0]["messages"][1]["content"]
    long = build_workload(_config(prompt_padding_tokens=200))[0]["messages"][1]["content"]
    assert len(long) > len(short)


def test_user_prompt_gives_the_tool_something_to_do():
    """Regression: a vague task let Qwen3-0.6B answer without calling the tool.

    The measured failure was `tool_calls: 0` with a <think> block explaining
    that the request named no data. The user turn must name the tool and the
    required keys so the call has a purpose.
    """

    user = build_workload(_config())[0]["messages"][1]["content"]
    assert "'lookup'" in user
    assert "result=result-0000" in user
    assert "report the returned value" in user


def test_user_prompt_matches_the_declared_required_keys():
    requests = build_workload(
        _config(tools=[{"name": "fetch", "required_keys": ["doc_id", "payload"]}])
    )
    user = requests[0]["messages"][1]["content"]
    for key in ("doc_id", "payload"):
        assert f"{key}={key}-0000" in user


def test_system_prompt_forbids_commentary():
    system = build_workload(_config())[0]["messages"][0]["content"]
    assert "emit exactly one JSON object and nothing else" in system


def test_call_plans_must_reference_declared_tools():
    with pytest.raises(ConfigError):
        build_workload(_config(call_plans=[["missing_tool"]]))


def test_empty_tools_and_model_are_rejected():
    with pytest.raises(ConfigError):
        build_workload({"model": "m", "tools": []})
    with pytest.raises(ConfigError):
        build_workload({"tools": [{"name": "lookup"}]})


@pytest.mark.parametrize("name", ["smoke.json", "kv_main.json", "prewarm.json"])
def test_shipped_configs_generate_loadable_workloads(name, tmp_path):
    ensure_mini_sglang_path()
    from minisgl.benchmark.agent import load_workload

    config = load_config(CONFIG_DIR / name)
    workload_config = {**config["workload"], "model": config["model"]}
    requests = build_workload(workload_config)
    assert requests
    # Both modes must survive the client's own request validation.
    items = load_workload(write_workload(requests, tmp_path / "workload.jsonl"))
    assert len(items) == len(requests)
