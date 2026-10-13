"""Regression tests for the server command line each cell launches.

The host-KV-tier rule is the important one: TokenCake's offload and prefetch
actions are recorded with ``implemented=False`` when no host tier is configured,
so a misconfigured cell looks like "the policy decided not to transfer" instead
of "the transfer could not run". These tests pin that behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from run_experiment import (
    ExperimentError,
    _extend_no_proxy,
    build_server_argv,
    effective_host_kv_pages,
)

CONFIG = {
    "name": "unit",
    "model": "test-model",
    "server": {
        "cache_type": "radix",
        "host_kv_pages": 2048,
        "prewarm_threshold": 0.5,
        "pool_size": 4,
    },
}


def _argv(policy: str, overrides=None, run_dir: Path | None = None):
    return build_server_argv(
        CONFIG,
        policy=policy,
        server_overrides=overrides,
        run_dir=run_dir or Path("/tmp/run"),
        python="python3",
    )


def test_tokencake_gets_the_configured_host_tier():
    argv = _argv("tokencake")
    assert argv[:3] == ["python3", "-m", "minisgl"]
    assert argv[argv.index("--agentweave-kv-policy") + 1] == "tokencake"
    assert argv[argv.index("--agentweave-host-kv-pages") + 1] == "2048"
    assert "--enable-agentweave-runtime" in argv


@pytest.mark.parametrize("policy", ["no-control", "continuum"])
def test_other_policies_are_pinned_to_zero_host_pages(policy):
    assert _argv(policy)[_argv(policy).index("--agentweave-host-kv-pages") + 1] == "0"


def test_tokencake_without_host_tier_is_rejected():
    config = {**CONFIG, "server": {**CONFIG["server"], "host_kv_pages": 0}}
    with pytest.raises(ExperimentError, match="host_kv_pages"):
        build_server_argv(
            config,
            policy="tokencake",
            server_overrides=None,
            run_dir=Path("/tmp/run"),
            python="python3",
        )


def test_variant_overrides_reach_the_command_line():
    argv = _argv("no-control", {"pool_size": 0, "prewarm_threshold": 1.0})
    assert argv[argv.index("--agentweave-pool-size") + 1] == "0"
    assert argv[argv.index("--agentweave-prewarm-threshold") + 1] == "1.0"


def test_event_log_is_an_absolute_path_with_placeholders(tmp_path):
    argv = _argv("no-control", run_dir=tmp_path)
    event_log = argv[argv.index("--agentweave-event-log") + 1]
    assert "{component}" in event_log and "{rank}" in event_log
    assert event_log.startswith("/")


def test_extra_args_are_appended_last():
    config = {
        **CONFIG,
        "server": {**CONFIG["server"], "extra_args": ["--num-page-override", "4096"]},
    }
    argv = build_server_argv(
        config,
        policy="no-control",
        server_overrides=None,
        run_dir=Path("/tmp/run"),
        python="python3",
    )
    assert argv[-2:] == ["--num-page-override", "4096"]


def test_model_is_required():
    with pytest.raises(ExperimentError, match="model"):
        build_server_argv(
            {"server": {}},
            policy="no-control",
            server_overrides=None,
            run_dir=Path("/tmp/run"),
            python="python3",
        )


def test_effective_host_pages_ignores_overrides_for_non_tokencake():
    server = {"host_kv_pages": 4096}
    assert effective_host_kv_pages(server, "tokencake") == 4096
    assert effective_host_kv_pages(server, "continuum") == 0


def test_no_proxy_always_covers_loopback():
    """Regression: a cluster-wide http_proxy must not intercept the server probe.

    Without this, `GET /v1/models` goes to the proxy and returns 403, which
    hides the real startup failure.
    """

    entries = _extend_no_proxy(None, "127.0.0.1", 1919).split(",")
    assert "localhost" in entries
    assert "127.0.0.1" in entries


def test_no_proxy_keeps_existing_entries():
    result = _extend_no_proxy("proxy.cse.cuhk.edu.hk,.internal", "127.0.0.1", 1919)
    assert result.startswith("proxy.cse.cuhk.edu.hk,.internal")
    assert "127.0.0.1" in result.split(",")


def test_no_proxy_is_idempotent():
    once = _extend_no_proxy(None, "127.0.0.1", 1919)
    assert _extend_no_proxy(once, "127.0.0.1", 1919) == once
