#!/usr/bin/env python3
"""Run one AgentWeave experiment cell: launch server, replay workload, collect logs.

A "cell" is one combination of (KV policy, server variant, concurrency). The
server is restarted for every cell because the warm pool and the KV/duration
history are process-local state; reusing a server would let one policy's history
leak into the next (see docs/testing.md section 5).

The workload is NOT generated here by default. All cells of an experiment must
replay the identical workload file, request order and concurrency, so the
generator runs once per experiment in ``run_matrix`` and the path is passed in.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.agentweave.workload import (
    build_workload,
    load_config,
    workload_config_from_experiment,
    write_workload,
)

# Server flags that map 1:1 onto config["server"] keys.
_SERVER_FLAG_MAP = {
    "prewarm_threshold": "--agentweave-prewarm-threshold",
    "pool_size": "--agentweave-pool-size",
    "pool_per_key": "--agentweave-pool-per-key",
    "pool_ttl_s": "--agentweave-pool-ttl-s",
    "continuation_restore_timeout_s": "--agentweave-continuation-restore-timeout-s",
    "d2h_bandwidth_gbps": "--agentweave-d2h-bandwidth-gbps",
    "h2d_bandwidth_gbps": "--agentweave-h2d-bandwidth-gbps",
    "d2h_base_ms": "--agentweave-d2h-base-ms",
    "h2d_base_ms": "--agentweave-h2d-base-ms",
    "transfer_safety_margin_ms": "--agentweave-transfer-safety-margin-ms",
    "transfer_calibration_alpha": "--agentweave-transfer-calibration-alpha",
}

_TRANSFER_CAPABLE_POLICIES = ("tokencake",)


class ExperimentError(RuntimeError):
    """Raised when a cell cannot be set up or the server fails to come up."""


def effective_host_kv_pages(server_config: Mapping[str, Any], policy: str) -> int:
    """Host KV pages actually passed to the server for this policy.

    TokenCake can only move KV between GPU and host when a host tier exists.
    Without ``--agentweave-host-kv-pages`` its offload/prefetch actions are
    recorded with ``implemented=False``, which would silently produce meaningless
    numbers. The other policies must stay at zero so they cannot borrow the host
    tier and be compared unfairly.
    """

    configured = int(server_config.get("host_kv_pages", 0))
    if policy in _TRANSFER_CAPABLE_POLICIES:
        return configured
    return 0


def build_server_argv(
    config: Mapping[str, Any],
    *,
    policy: str,
    server_overrides: Mapping[str, Any] | None,
    run_dir: Path,
    python: str,
) -> list[str]:
    """Assemble the ``python -m minisgl ...`` command line for one cell."""

    server = dict(config.get("server", {}))
    server.update(server_overrides or {})
    model = config.get("model")
    if not model:
        raise ExperimentError("config requires a top-level 'model'")

    host_pages = effective_host_kv_pages(server, policy)
    if policy in _TRANSFER_CAPABLE_POLICIES and host_pages <= 0:
        raise ExperimentError(
            f"policy {policy!r} needs server.host_kv_pages > 0; "
            "without a host tier its transfers are not executed."
        )

    argv = [
        python,
        "-m",
        "minisgl",
        "--model",
        str(model),
        "--cache-type",
        str(server.get("cache_type", "radix")),
        "--host",
        str(server.get("host", "127.0.0.1")),
        "--port",
        str(server.get("port", 1919)),
        "--enable-agentweave-runtime",
        "--agentweave-kv-policy",
        policy,
        "--agentweave-host-kv-pages",
        str(host_pages),
        "--agentweave-event-log",
        str((run_dir / "events-{component}-{rank}.jsonl").resolve()),
    ]
    for key, flag in _SERVER_FLAG_MAP.items():
        if key in server:
            argv.extend([flag, str(server[key])])
    argv.extend(str(item) for item in server.get("extra_args", []))
    return argv


def free_port(preferred: int, host: str = "127.0.0.1") -> int:
    """Return ``preferred`` when idle, otherwise an OS-assigned free port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _extend_no_proxy(existing: str | None, host: str, port: int) -> str:
    """Add loopback entries to a ``no_proxy`` value without dropping existing ones."""

    entries = [entry.strip() for entry in (existing or "").split(",") if entry.strip()]
    for entry in ("localhost", "127.0.0.1", host, f"{host}:{port}"):
        if entry and entry not in entries:
            entries.append(entry)
    return ",".join(entries)


def wait_for_server(
    base_url: str,
    process: subprocess.Popen[bytes],
    *,
    timeout_s: float,
    poll_s: float = 1.0,
) -> None:
    """Poll ``GET /v1/models`` until the server answers or dies.

    Proxies are bypassed explicitly. A cluster that exports ``http_proxy`` would
    otherwise send this loopback request to the proxy, which answers 403 and
    hides the real reason the server never came up.
    """

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_s
    url = f"{base_url}/v1/models"
    last_error = "no attempt made"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ExperimentError(
                f"server exited early with code {process.returncode}; see server.log"
            )
        try:
            with opener.open(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except urllib.error.URLError as exc:
            last_error = str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(poll_s)
    raise ExperimentError(f"server not ready within {timeout_s}s ({last_error})")


def _git_sha(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _gpu_summary() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def run_cell(
    *,
    config: Mapping[str, Any],
    policy: str,
    server_overrides: Mapping[str, Any] | None,
    variant_name: str | None,
    concurrency: int,
    repeats: int,
    workload_path: Path,
    run_dir: Path,
    python: str,
    mini_sglang_dir: Path,
    startup_timeout_s: float,
    client_timeout_s: float,
) -> dict[str, Any]:
    """Execute one cell and return its manifest."""

    run_dir.mkdir(parents=True, exist_ok=True)
    port = free_port(int(config.get("server", {}).get("port", 1919)))
    host = str(config.get("server", {}).get("host", "127.0.0.1"))
    base_url = f"http://{host}:{port}"

    argv = build_server_argv(
        config,
        policy=policy,
        server_overrides=server_overrides,
        run_dir=run_dir,
        python=python,
    )
    # The port resolved above may differ from the configured one.
    port_index = argv.index("--port") + 1
    argv[port_index] = str(port)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str((mini_sglang_dir / "python").resolve()), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # The benchmark client posts to loopback. On a cluster that exports
    # http_proxy, urllib would route that through the proxy and get a 403
    # instead of reaching the server, so loopback is excluded from proxying.
    no_proxy = _extend_no_proxy(env.get("no_proxy") or env.get("NO_PROXY"), host, port)
    env["no_proxy"] = no_proxy
    env["NO_PROXY"] = no_proxy

    manifest: dict[str, Any] = {
        "cell": {
            "policy": policy,
            "variant": variant_name,
            "concurrency": concurrency,
            "repeats": repeats,
        },
        "config_name": config.get("name"),
        "model": config.get("model"),
        "workload": str(workload_path.resolve()),
        "server_argv": argv,
        "host_kv_pages": effective_host_kv_pages(
            {**dict(config.get("server", {})), **(server_overrides or {})}, policy
        ),
        "port": port,
        "started_at_ns": time.time_ns(),
        "repo_sha": _git_sha(_REPO_ROOT),
        "submodule_sha": _git_sha(mini_sglang_dir),
        "gpu": _gpu_summary(),
    }

    with (run_dir / "server.log").open("wb") as log:
        process = subprocess.Popen(
            argv,
            cwd=str(mini_sglang_dir),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            # Own process group so the whole server tree can be torn down at the
            # end of the cell. mini-sglang forks tokenizer/scheduler children;
            # signalling only the parent leaves them alive, and on a Slurm
            # allocation the orphaned children keep the job from finishing.
            start_new_session=True,
        )
    try:
        wait_for_server(base_url, process, timeout_s=startup_timeout_s)
        manifest["ready_at_ns"] = time.time_ns()
        client_argv = [
            python,
            "-m",
            "minisgl.benchmark.agent",
            str(workload_path.resolve()),
            "--endpoint",
            f"{base_url}/v1/agent/runs",
            "--concurrency",
            str(concurrency),
            "--repeats",
            str(repeats),
            "--timeout-s",
            str(client_timeout_s),
            "--output",
            str((run_dir / "client.jsonl").resolve()),
        ]
        manifest["client_argv"] = client_argv
        completed = subprocess.run(
            client_argv,
            cwd=str(mini_sglang_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        manifest["client_returncode"] = completed.returncode
        manifest["client_stdout"] = completed.stdout.strip()[-2000:]
        manifest["client_stderr"] = completed.stderr.strip()[-2000:]
    finally:
        manifest["finished_at_ns"] = time.time_ns()
        _terminate(process)
        manifest["server_returncode"] = process.returncode

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not manifest.get("client_argv"):
        raise ExperimentError(f"cell failed before the client ran; see {run_dir}/server.log")
    return manifest


def _terminate(process: subprocess.Popen[bytes], *, grace_s: float = 30.0) -> None:
    """Stop the server and every process it forked.

    The server is started in its own session, so signalling the process group
    reaches the tokenizer and scheduler children too. Killing only the parent
    would leave them running, which on a Slurm allocation keeps the job alive
    until the scheduler time limit expires.
    """

    if process.poll() is not None:
        return
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:  # pragma: no cover - exited between check and call
        return
    _signal_group(group, signal.SIGTERM)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        _signal_group(group, signal.SIGKILL)
        process.wait(timeout=grace_s)


def _signal_group(group: int, sig: signal.Signals) -> None:
    """Signal a process group, tolerating one that has already exited."""

    if process_group_exists(group):
        os.killpg(group, sig)


def process_group_exists(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - not our group
        return True
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one AgentWeave experiment cell")
    parser.add_argument("--config", required=True, help="Experiment config JSON")
    parser.add_argument("--run-dir", required=True, help="Destination directory for this cell")
    parser.add_argument("--policy", default="no-control")
    parser.add_argument("--variant", default=None, help="matrix.server_variants name")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workload", default=None, help="Pre-generated workload JSONL")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mini-sglang-dir", default=None)
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--client-timeout-s", type=float, default=600.0)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    mini_sglang_dir = (
        Path(args.mini_sglang_dir) if args.mini_sglang_dir else _REPO_ROOT / "mini-sglang"
    )

    overrides: dict[str, Any] = {}
    variant_name = None
    matrix = config.get("matrix", {})
    if args.variant:
        variants = {variant.get("name"): variant for variant in matrix.get("server_variants", [])}
        if args.variant not in variants:
            parser.error(f"unknown variant {args.variant!r}; have {sorted(variants)}")
        overrides = dict(variants[args.variant].get("overrides", {}))
        variant_name = args.variant

    run_dir = Path(args.run_dir)
    if args.workload:
        workload_path = Path(args.workload)
    else:
        workload_path = write_workload(
            build_workload(workload_config_from_experiment(config)),
            run_dir / "workload.jsonl",
        )

    manifest = run_cell(
        config=config,
        policy=args.policy,
        server_overrides=overrides,
        variant_name=variant_name,
        concurrency=args.concurrency,
        repeats=args.repeats,
        workload_path=workload_path,
        run_dir=run_dir,
        python=args.python,
        mini_sglang_dir=mini_sglang_dir,
        startup_timeout_s=args.startup_timeout_s,
        client_timeout_s=args.client_timeout_s,
    )
    print(json.dumps({"run_dir": str(run_dir), "client": manifest.get("client_stdout")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
