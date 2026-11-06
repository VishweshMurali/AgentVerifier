import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, InternalServerError, NotFoundError

from agent_committee import llm, throughput
from agent_committee.schemas import Case
from agent_committee.throughput import (
    BenchmarkFailed,
    OllamaUnreachable,
    calls_per_hour,
    run_benchmark,
)

CASES = [Case(id=str(i), text=f"text {i}", label="card_arrival") for i in range(3)]
_REQ = httpx.Request("POST", "http://x")


def test_calls_per_hour_arithmetic():
    assert calls_per_hour(30, 60) == pytest.approx(1800)
    assert calls_per_hour(0, 10) == 0


@pytest.mark.parametrize("n,t", [(-1, 1), (5, 0), (5, -2)])
def test_calls_per_hour_rejects_invalid_input(n, t):
    with pytest.raises(ValueError):
        calls_per_hour(n, t)


def _usage(pt=10, ct=2):
    return SimpleNamespace(prompt_tokens=pt, completion_tokens=ct)


class FakeClient:
    """Stands in for the OpenAI client: `outcomes` are returned or raised in order, then repeat."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        out = self.outcomes[min(len(self.requests), len(self.outcomes)) - 1]
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(usage=out)


def test_run_benchmark_counts_calls_and_tokens_excluding_warmup():
    fake = FakeClient(_usage(100, 50), _usage(10, 2))
    result = run_benchmark(CASES, client=fake)
    assert len(fake.requests) == 1 + len(CASES)  # warm-up + one per case
    assert result.n_calls == 3 and result.concurrency == 1
    assert (result.prompt_tokens, result.completion_tokens) == (30, 6)  # warm-up not counted
    assert result.calls_per_hour > 0
    assert {r["model"] for r in fake.requests} == {llm.MODEL}


def test_endpoint_and_model_come_from_the_llm_wrapper():
    assert throughput.BASE_URL is llm.BASE_URL and throughput.MODEL is llm.MODEL


@pytest.mark.parametrize("cases", [[], ()])
def test_empty_input_is_a_clear_error_before_any_call(cases):
    fake = FakeClient(_usage())
    with pytest.raises(ValueError, match="no cases"):
        run_benchmark(list(cases), client=fake)
    assert fake.requests == []


def test_concurrency_below_one_rejected():
    with pytest.raises(ValueError, match="concurrency"):
        run_benchmark(CASES, concurrency=0, client=FakeClient(_usage()))


@pytest.mark.parametrize(
    "usage",
    [
        None,
        SimpleNamespace(prompt_tokens=None, completion_tokens=2),
        SimpleNamespace(prompt_tokens=10, completion_tokens=-1),
        SimpleNamespace(prompt_tokens=float("nan"), completion_tokens=2),
        SimpleNamespace(prompt_tokens=True, completion_tokens=2),
    ],
)
def test_missing_or_invalid_usage_is_an_error_not_zero(usage):
    with pytest.raises(BenchmarkFailed, match="usage"):
        run_benchmark(CASES, client=FakeClient(usage))


def test_missing_usage_mid_run_is_an_error():
    fake = FakeClient(_usage(), _usage(), None)
    with pytest.raises(BenchmarkFailed, match="usage"):
        run_benchmark(CASES, client=fake)


@pytest.mark.parametrize(
    "exc,match",
    [
        (APIConnectionError(request=_REQ), "cannot reach"),
        (APITimeoutError(request=_REQ), "cannot reach"),
        (
            NotFoundError("no model", response=httpx.Response(404, request=_REQ), body=None),
            "not found",
        ),
        (
            InternalServerError("boom", response=httpx.Response(500, request=_REQ), body=None),
            "HTTP 500",
        ),
    ],
)
def test_transport_errors_become_ollama_unreachable(exc, match):
    with pytest.raises(OllamaUnreachable, match=match):
        run_benchmark(CASES, client=FakeClient(exc))


def test_bad_request_maps_to_ollama_unreachable():
    from openai import BadRequestError

    exc = BadRequestError("bad", response=httpx.Response(400, request=_REQ), body=None)
    with pytest.raises(OllamaUnreachable, match="HTTP 400"):
        run_benchmark(CASES, client=FakeClient(_usage(), exc))


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_throughput.py"
    spec = importlib.util.spec_from_file_location("benchmark_throughput_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "error", [OllamaUnreachable("down"), BenchmarkFailed("no usage"), ValueError("no cases")]
)
def test_cli_exits_2_with_a_message_on_failure(monkeypatch, capsys, error):
    cli = _load_cli()

    def fail(cases, *, concurrency):
        raise error

    monkeypatch.setattr(cli, "load_dev", lambda: CASES)
    monkeypatch.setattr(cli, "run_benchmark", fail)
    monkeypatch.setattr("sys.argv", ["benchmark_throughput.py"])
    assert cli.main() == 2
    assert f"ERROR: {error}" in capsys.readouterr().err


def test_cli_output_says_the_figure_is_not_a_full_arm(monkeypatch, capsys):
    cli = _load_cli()
    monkeypatch.setattr(cli, "load_dev", lambda: CASES)
    monkeypatch.setattr(
        cli,
        "run_benchmark",
        lambda cases, *, concurrency: run_benchmark(cases, client=FakeClient(_usage())),
    )
    monkeypatch.setattr("sys.argv", ["benchmark_throughput.py"])
    assert cli.main() == 0
    assert "not a full arm" in capsys.readouterr().out
