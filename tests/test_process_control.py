"""Tests for tearing down the server process tree.

mini-sglang forks tokenizer and scheduler children. On a Slurm allocation,
orphaned children keep the job alive until its time limit, so `_terminate` must
reach the whole process group rather than just the process it started.
"""

from __future__ import annotations

import os
import subprocess
import time

from run_experiment import _terminate, process_group_exists


def _spawn_group() -> subprocess.Popen[bytes]:
    """A parent that forks a long-lived child, mimicking the server."""

    return subprocess.Popen(
        ["bash", "-c", "sleep 300 & sleep 300"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_for_group_exit(group: int, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not process_group_exists(group):
            return True
        time.sleep(0.1)
    return False


def test_terminate_stops_the_whole_group_including_children():
    process = _spawn_group()
    group = os.getpgid(process.pid)
    try:
        assert process_group_exists(group)
        _terminate(process, grace_s=5.0)
        assert _wait_for_group_exit(group), "child processes survived termination"
    finally:
        if process_group_exists(group):
            os.killpg(group, 9)


def test_terminate_is_a_noop_for_an_already_exited_process():
    process = subprocess.Popen(["true"], start_new_session=True)
    process.wait(timeout=10)
    _terminate(process, grace_s=5.0)  # must not raise


def test_process_group_exists_reports_a_stopped_group():
    process = _spawn_group()
    group = os.getpgid(process.pid)
    try:
        assert process_group_exists(group)
    finally:
        os.killpg(group, 9)
        # Reap the parent: an unreaped child stays in the group as a zombie, so
        # killpg would keep succeeding and the group would look alive.
        process.wait(timeout=10)
        assert _wait_for_group_exit(group)


def test_process_group_exists_rejects_an_unused_group_id():
    # A group id that no process can be using.
    assert not process_group_exists(2_000_000_000)
