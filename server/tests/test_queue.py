"""The waiting line: one test at a time, the next visitors wait in order, a full line is refused."""

import sys
import time

import pytest

from verifier_server.jobs import JobBusy, JobManager

NAP = [sys.executable, "-c", "import time; time.sleep(1.2)"]
QUICK = [sys.executable, "-c", "print('ok')"]


def wait(manager, job, state, secs=20):
    end = time.time() + secs
    while time.time() < end:
        if manager.get(job.id).state == state:
            return
        time.sleep(0.05)
    raise AssertionError(f"{job.id} is {manager.get(job.id).state}, not {state}")


def test_visitors_wait_in_order_and_a_full_line_is_refused(tmp_path):
    m = JobManager(tmp_path, max_queue=2)
    a = m.start("quick", NAP, 30)
    b = m.start("quick", QUICK, 30, tokens=900)
    c = m.start("quick", QUICK, 30, tokens=900)
    assert (a.state, b.state, c.state) == ("running", "queued", "queued")
    assert (b.view()["position"], c.view()["position"]) == (1, 2)
    assert m.queued_tokens() == 1800
    with pytest.raises(JobBusy):
        m.start("quick", QUICK, 30)  # a third waiting test does not fit
    wait(m, a, "succeeded")
    wait(m, b, "succeeded")
    wait(m, c, "succeeded")
    assert m.queued_tokens() == 0
    assert a.finished <= b.started <= b.finished <= c.started  # strictly one after another


def test_without_a_line_a_busy_demo_refuses_like_before(tmp_path):
    m = JobManager(tmp_path, max_queue=0)
    a = m.start("quick", NAP, 30)
    with pytest.raises(JobBusy):
        m.start("quick", QUICK, 30)
    wait(m, a, "succeeded")


def test_a_failed_test_does_not_block_the_line(tmp_path):
    m = JobManager(tmp_path, max_queue=1)
    a = m.start(
        "quick", [sys.executable, "-c", "import sys, time; time.sleep(0.5); sys.exit(3)"], 30
    )
    b = m.start("quick", QUICK, 30)
    wait(m, a, "failed")
    wait(m, b, "succeeded")
