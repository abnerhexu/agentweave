"""Generate AgentWeave workloads (one ``AgentRunRequest`` JSON object per line).

The serving side accepts one agent run per HTTP request, so every line is a
self-contained agent: it carries its own tool sandbox delays, prompt length and
arrival stagger. Two modes are supported:

``synthetic``
    A single tool the agent is told to call once before answering. Use this for
    the KV experiments, where the controlled variables are tool duration,
    sandbox cold start, prompt/KV length and concurrency.

``structured``
    Several tools with different required-key sets and a long padded argument.
    Use this for the prewarm experiments: the confidence policy fires once the
    tool *name* completes, so the argument-generation time after that point is
    the window in which a cold start can be hidden.

Deterministic sandbox timing relies on the ``simulated`` backend reading
``AGENTWEAVE_SIMULATED_PREPARE_DELAY_S`` (cold start) and
``AGENTWEAVE_SIMULATED_EXECUTION_DELAY_S`` (tool execution) from the tool's
``environment`` map. Those values also participate in the warm-pool key, so
agents with different durations do not share pooled sandboxes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

# Approximate English filler. One word is roughly one tokenizer token, which is
# close enough to control KV footprint; exactness is not required because the
# harness compares policies at equal prompt length rather than at absolute
# token counts.
_FILLER_WORDS = (
    "the",
    "acme",
    "system",
    "reports",
    "latency",
    "measured",
    "between",
    "stages",
    "of",
    "each",
    "workflow",
    "and",
    "recorded",
    "in",
    "a",
    "shared",
    "event",
    "log",
    "for",
    "later",
    "analysis",
)

# Wording matters more than it looks. Qwen3 runs in thinking mode by default
# (the submodule's apply_chat_template does not pass enable_thinking=False), so a
# vague task can be answered purely with reasoning: measured on Qwen3-0.6B, the
# original "Run task N" prompt produced a <think> block and zero tool calls. The
# instruction therefore states the exact JSON shape and forbids commentary, and
# the user turn names a concrete record to look up so the tool has a purpose.
_DEFAULT_TASK_INSTRUCTION = (
    "You are a tool-using agent under an experiment harness. "
    "To use a tool, emit exactly one JSON object and nothing else, of the form "
    '{"name": "<tool_name>", "arguments": {"<required_key>": "<value>"}}, '
    "including every required key. Do not describe what you are about to do."
)


class ConfigError(ValueError):
    """Raised when a workload config is malformed."""


def sample_distribution(spec: Mapping[str, Any], rng: random.Random) -> float:
    """Draw one non-negative value (in seconds) from a config distribution."""

    kind = spec.get("kind", "fixed")
    if kind == "fixed":
        return _non_negative(spec, "value_s", kind)
    if kind == "uniform":
        low = _non_negative(spec, "low_s", kind)
        high = _non_negative(spec, "high_s", kind)
        if high < low:
            raise ConfigError("uniform distribution requires high_s >= low_s")
        return rng.uniform(low, high)
    if kind == "lognormal":
        median = _non_negative(spec, "median_s", kind)
        sigma = float(spec.get("sigma", 0.5))
        if sigma < 0:
            raise ConfigError("lognormal sigma must be non-negative")
        if median == 0:
            return 0.0
        import math

        return median * math.exp(sigma * rng.gauss(0.0, 1.0))
    if kind == "choice":
        values = spec.get("values_s")
        if not isinstance(values, list) or not values:
            raise ConfigError("choice distribution requires a non-empty values_s list")
        parsed = [float(value) for value in values]
        if any(value < 0 for value in parsed):
            raise ConfigError("choice values_s entries must be non-negative")
        return rng.choice(parsed)
    raise ConfigError(f"unknown distribution kind {kind!r}")


def build_workload(config: Mapping[str, Any], *, seed: int | None = None) -> list[dict[str, Any]]:
    """Return one request dictionary per agent, in submission order."""

    model = config.get("model")
    if not isinstance(model, str) or not model:
        raise ConfigError("config requires a non-empty 'model'")

    mode = config.get("mode", "synthetic")
    if mode not in ("synthetic", "structured"):
        raise ConfigError(f"unknown mode {mode!r}")

    tools = config.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ConfigError("config requires a non-empty 'tools' list")
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise ConfigError(f"tools[{index}] must be an object")
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            raise ConfigError(f"tools[{index}] requires a non-empty name")
        required_keys = tool.get("required_keys", [])
        if not isinstance(required_keys, list):
            raise ConfigError(f"tools[{index}].required_keys must be a list")

    num_agents = int(config.get("num_agents", 1))
    if num_agents <= 0:
        raise ConfigError("num_agents must be positive")

    effective_seed = seed if seed is not None else int(config.get("seed", 0))
    rng = random.Random(effective_seed)

    duration_spec = config.get("duration", {"kind": "fixed", "value_s": 1.0})
    cold_start_spec = config.get("cold_start", {"kind": "fixed", "value_s": 0.1})
    arrival_spec = config.get("arrival", {"kind": "burst"})
    padding = int(config.get("prompt_padding_tokens", 0))
    if padding < 0:
        raise ConfigError("prompt_padding_tokens must be non-negative")
    argument_padding = int(config.get("argument_padding_tokens", 0))
    if argument_padding < 0:
        raise ConfigError("argument_padding_tokens must be non-negative")
    padded_argument = config.get("padded_argument")
    if padded_argument is not None and not isinstance(padded_argument, str):
        raise ConfigError("padded_argument must be a string")

    tool_names = [str(tool["name"]) for tool in tools]
    call_plan = _resolve_call_plan(config, tool_names, mode)
    requests: list[dict[str, Any]] = []
    for index in range(num_agents):
        duration_s = sample_distribution(duration_spec, rng)
        cold_start_s = sample_distribution(cold_start_spec, rng)
        agent_tools = [_tool_payload(tool, duration_s, cold_start_s) for tool in tools]
        plan = call_plan[index % len(call_plan)]
        requests.append(
            {
                "model": model,
                "agent_id": f"agent-{index:04d}",
                "run_id": f"agent-{index:04d}",
                "messages": [
                    {
                        "role": "system",
                        "content": _system_prompt(
                            mode,
                            plan,
                            argument_padding,
                            config.get("task_instruction"),
                            padded_argument,
                        ),
                    },
                    {
                        "role": "user",
                        "content": _user_prompt(
                            index,
                            padding,
                            rng,
                            plan[0],
                            _required_keys(tools, plan[0]),
                        ),
                    },
                ],
                "tools": agent_tools,
                "max_steps": int(config.get("max_steps", 4)),
                "max_tokens": int(config.get("max_tokens", 512)),
                "_delay_s": _arrival_delay(arrival_spec, rng),
                "_duration_s": duration_s,
                "_cold_start_s": cold_start_s,
            }
        )
    return requests


def write_workload(requests: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the JSONL workload plus a sidecar of sampled parameters.

    Only the AgentWeave request schema and the client's ``_delay_s`` stagger are
    written to the workload. The sampled duration/cold-start values go to
    ``<output>.sampling.jsonl`` so the request body stays exactly what the
    serving API expects, while the report can still plot the generated
    distributions.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sampling_path = destination.with_suffix(destination.suffix + ".sampling.jsonl")
    with (
        destination.open("w", encoding="utf-8") as sink,
        sampling_path.open("w", encoding="utf-8") as sampling_sink,
    ):
        for request in requests:
            payload = {
                key: value
                for key, value in request.items()
                if not (key.startswith("_") and key != "_delay_s")
            }
            sink.write(json.dumps(payload, sort_keys=True) + "\n")
            sampling_sink.write(
                json.dumps(
                    {
                        "agent_id": request.get("agent_id"),
                        "duration_s": request.get("_duration_s"),
                        "cold_start_s": request.get("_cold_start_s"),
                        "arrival_delay_s": request.get("_delay_s"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return destination


def _resolve_call_plan(
    config: Mapping[str, Any],
    tool_names: Sequence[str],
    mode: str,
) -> list[list[str]]:
    """Per-agent tool-call sequence, used only to phrase the system prompt."""

    configured = config.get("call_plans")
    if configured is None:
        steps = int(config.get("tool_calls_per_agent", 1))
        if steps <= 0:
            raise ConfigError("tool_calls_per_agent must be positive")
        return [[tool_names[0]] * steps]
    if not isinstance(configured, list) or not configured:
        raise ConfigError("call_plans must be a non-empty list")
    plans: list[list[str]] = []
    for index, plan in enumerate(configured):
        if not isinstance(plan, list) or not plan:
            raise ConfigError(f"call_plans[{index}] must be a non-empty list of tool names")
        for name in plan:
            if name not in tool_names:
                raise ConfigError(f"call_plans[{index}] references unknown tool {name!r}")
        plans.append(list(plan))
    return plans


def _required_keys(tools: Sequence[Mapping[str, Any]], name: str) -> tuple[str, ...]:
    for tool in tools:
        if tool["name"] == name:
            return tuple(str(key) for key in tool.get("required_keys", []))
    return ()


def _system_prompt(
    mode: str,
    plan: Sequence[str],
    argument_padding: int,
    override: Any,
    padded_argument: str | None = None,
) -> str:
    if isinstance(override, str) and override:
        return override
    lines = [_DEFAULT_TASK_INSTRUCTION]
    if len(plan) == 1:
        lines.append(f"Call the {plan[0]!r} tool exactly once, then answer.")
    else:
        sequence = ", ".join(f"{step!r}" for step in plan)
        lines.append(f"Call the following tools in order, once each: {sequence}. Then answer.")
    if mode == "structured" and argument_padding > 0:
        # The window that can hide a cold start is the argument-generation time
        # after the tool name resolves. Naming a specific argument keeps that
        # window long and predictable instead of hoping the model volunteers it.
        target = (
            f"the {padded_argument!r} argument of every tool"
            if padded_argument
            else "every string argument"
        )
        lines.append(f"Set {target} to approximately {argument_padding} words of plain text.")
    return " ".join(lines)


def _user_prompt(
    index: int,
    padding: int,
    rng: random.Random,
    tool_name: str,
    required_keys: Sequence[str],
) -> str:
    """A concrete request that gives the tool something to do.

    The arguments named here become the keys the model is expected to emit, and
    the runtime rejects a call that omits one, so they mirror required_keys.
    """

    if required_keys:
        arguments = ", ".join(f"{key}={key}-{index:04d}" for key in required_keys)
    else:
        arguments = "<no arguments>"
    base = (
        f"Look up record {index:04d} with the {tool_name!r} tool "
        f"(arguments: {arguments}), then report the returned value."
    )
    if padding <= 0:
        return base
    filler = " ".join(rng.choice(_FILLER_WORDS) for _ in range(padding))
    return f"{base} Background context: {filler}"


def _tool_payload(
    tool: Mapping[str, Any], duration_s: float, cold_start_s: float
) -> dict[str, Any]:
    environment = dict(tool.get("environment", {}))
    environment["AGENTWEAVE_SIMULATED_PREPARE_DELAY_S"] = repr(cold_start_s)
    environment["AGENTWEAVE_SIMULATED_EXECUTION_DELAY_S"] = repr(duration_s)
    return {
        "name": tool["name"],
        "required_keys": list(tool.get("required_keys", [])),
        "sandbox_backend": tool.get("sandbox_backend", "simulated"),
        "runtime": tool.get("runtime", "default"),
        "dependencies": list(tool.get("dependencies", [])),
        "preparation_level": tool.get("preparation_level", "runtime"),
        "environment": environment,
    }


def _arrival_delay(spec: Mapping[str, Any], rng: random.Random) -> float:
    """Client-side stagger inserted before submitting the next agent."""

    kind = spec.get("kind", "burst")
    if kind == "burst":
        return 0.0
    if kind == "poisson":
        rate = float(spec.get("rate_per_s", 1.0))
        if rate <= 0:
            raise ConfigError("poisson arrival requires rate_per_s > 0")
        return rng.expovariate(rate)
    if kind == "fixed":
        return _non_negative(spec, "interval_s", kind)
    raise ConfigError(f"unknown arrival kind {kind!r}")


def _non_negative(spec: Mapping[str, Any], key: str, kind: str) -> float:
    if key not in spec:
        raise ConfigError(f"{kind} distribution requires {key!r}")
    value = float(spec[key])
    if value < 0:
        raise ConfigError(f"{kind} distribution {key!r} must be non-negative")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        config = json.load(source)
    if not isinstance(config, dict):
        raise ConfigError(f"{path}: config must be a JSON object")
    return config


def workload_config_from_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Accept either a bare workload config or a full experiment config.

    Experiment configs nest the generator settings under ``workload`` and keep
    ``model`` at the top level; a bare workload config has everything flat.
    Both are accepted so the generator CLI works on either file.
    """

    nested = config.get("workload")
    if isinstance(nested, Mapping):
        return {**nested, "model": config.get("model")}
    return dict(config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an AgentWeave JSONL workload")
    parser.add_argument("--config", required=True, help="Workload/experiment JSON config")
    parser.add_argument("--output", required=True, help="Destination JSONL path")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed")
    parser.add_argument("--num-agents", type=int, default=None, help="Override num_agents")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Re-read the output through the AgentWeave workload client loader.",
    )
    parser.add_argument(
        "--mini-sglang-dir",
        default=None,
        help="Path to the mini-sglang submodule (default: ./mini-sglang).",
    )
    args = parser.parse_args(argv)

    config = workload_config_from_experiment(load_config(args.config))
    if args.num_agents is not None:
        config = {**config, "num_agents": args.num_agents}
    requests = build_workload(config, seed=args.seed)
    destination = write_workload(requests, args.output)

    durations = sorted(request["_duration_s"] for request in requests)
    summary = {
        "output": str(destination),
        "agents": len(requests),
        "duration_s_min": durations[0],
        "duration_s_max": durations[-1],
    }

    if args.validate:
        from . import ensure_mini_sglang_path

        ensure_mini_sglang_path(args.mini_sglang_dir)
        from minisgl.benchmark.agent import load_workload

        summary["validated_items"] = len(load_workload(destination))

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
