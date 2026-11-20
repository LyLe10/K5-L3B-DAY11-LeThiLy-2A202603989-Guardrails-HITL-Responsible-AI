"""Offline checks of Blue's actual HTTP transport and SDK integration."""
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import openai_runtime as runtime


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def test_budget_counts_http_attempts_and_blocks_19th():
    sent = []
    clock = FakeClock()
    budget = runtime._BlueBudget(clock=clock.monotonic, sleep=clock.sleep)
    inner = httpx.MockTransport(lambda request: (sent.append(request) or httpx.Response(200)))
    transport = runtime._BlueTransport(budget, inner)
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions",
                            json={"model": runtime.BLUE_MODEL})
    for _ in range(18):
        assert transport.handle_request(request).status_code == 200
    with pytest.raises(runtime.BlueRequestBudgetExceeded):
        transport.handle_request(request)
    assert len(sent) == budget.count == 18
    assert clock.waits == [5.0] * 17


def test_pacing_respects_elapsed_time_and_configuration(monkeypatch):
    clock = FakeClock()
    monkeypatch.setenv("BLUE_MIN_INTERVAL_SECONDS", "3")
    budget = runtime._BlueBudget(min_interval=runtime._blue_pacing_seconds(),
                                 clock=clock.monotonic, sleep=clock.sleep)
    budget.acquire()
    clock.now += 2
    budget.acquire()
    clock.now += 4
    budget.acquire()
    assert clock.waits == [1.0]


def test_pacing_is_shared_between_independent_budgets():
    clock = FakeClock()
    pacing = runtime._BluePacing()
    first = runtime._BlueBudget(clock=clock.monotonic, sleep=clock.sleep, pacing=pacing)
    second = runtime._BlueBudget(clock=clock.monotonic, sleep=clock.sleep, pacing=pacing)
    first.acquire()
    second.acquire()
    assert clock.waits == [5.0]
    assert first.count == second.count == 1


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "invalid"])
def test_invalid_pacing_fails_closed(monkeypatch, value):
    monkeypatch.setenv("BLUE_MIN_INTERVAL_SECONDS", value)
    with pytest.raises(ValueError):
        runtime._blue_pacing_seconds()


def test_transport_rejects_other_model_and_destination():
    budget = runtime._BlueBudget()
    transport = runtime._BlueTransport(budget, httpx.MockTransport(lambda _: httpx.Response(200)))
    for url, body in [
        ("https://other.example/api/v1/chat/completions", {"model": runtime.BLUE_MODEL}),
        ("https://openrouter.ai/api/v1/chat/completions", {"model": "paid-model"}),
        ("https://openrouter.ai/api/v1/chat/completions", {"model": runtime.BLUE_MODEL, "models": ["paid-model"]}),
    ]:
        with pytest.raises(RuntimeError):
            transport.handle_request(httpx.Request("POST", url, json=body))
    assert budget.count == 0


def test_sdk_failure_does_not_retry_or_expose_response(monkeypatch):
    calls = []
    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(500, json={"error": {"message": "SENSITIVE_TEST_MARKER"}})

    monkeypatch.setattr(runtime.httpx, "HTTPTransport", lambda **kwargs: httpx.MockTransport(respond))
    agent, runner = runtime.create_blue_pair(name="blue", instruction="test", app_name="test")
    runner._blue_budget = runtime._BlueBudget(min_interval=0)
    runner.client_kwargs["api_key"] = "test-key"
    with pytest.raises(RuntimeError) as error:
        asyncio.run(runner.chat(agent, "account balance"))
    assert "SENSITIVE_TEST_MARKER" not in str(error.value)
    assert len(calls) == runner._blue_budget.count == 1
    assert calls[0]["model"] == runtime.BLUE_MODEL


@pytest.mark.parametrize(("header", "expected"), [
    ("42", "42"),
    ("Wed, 21 Oct 2015 07:28:00 GMT", "Wed, 21 Oct 2015 07:28:00 GMT"),
    (None, "unavailable"),
    ("sensitive invalid header", "unavailable"),
])
def test_429_reports_only_safe_retry_after_without_retry(monkeypatch, header, expected):
    calls = []
    def respond(request):
        calls.append(request)
        headers = {"retry-after": header} if header is not None else {}
        headers["x-secret"] = "SECRET_HEADER_MARKER"
        return httpx.Response(429, headers=headers,
                              json={"error": {"message": "SECRET_BODY_MARKER"}})

    monkeypatch.setattr(runtime.httpx, "HTTPTransport", lambda **kwargs: httpx.MockTransport(respond))
    agent, runner = runtime.create_blue_pair(name="blue", instruction="test", app_name="test")
    runner._blue_budget = runtime._BlueBudget(min_interval=0)
    runner.client_kwargs["api_key"] = "SECRET_KEY_MARKER"
    with pytest.raises(RuntimeError) as error:
        asyncio.run(runner.chat(agent, "account balance"))
    assert str(error.value) == ("Blue request failed: RateLimitError; HTTP status: 429; "
                                f"Retry-After: {expected}; HTTP attempts: 1")
    assert len(calls) == runner._blue_budget.count == 1
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize(("status", "header", "expected"), [
    (500, "20", "20"),
    (502, None, "unavailable"),
    (503, "SECRET_INVALID_HEADER", "unavailable"),
])
def test_5xx_safe_diagnostics_without_retry(monkeypatch, status, header, expected):
    calls = []
    def respond(request):
        calls.append(request)
        headers = {"retry-after": header} if header is not None else {}
        headers["x-secret"] = "SECRET_HEADER_MARKER"
        return httpx.Response(status, headers=headers,
                              json={"error": {"message": "SECRET_BODY_MARKER"}})

    monkeypatch.setattr(runtime.httpx, "HTTPTransport", lambda **kwargs: httpx.MockTransport(respond))
    agent, runner = runtime.create_blue_pair(name="blue", instruction="test", app_name="test")
    runner._blue_budget = runtime._BlueBudget(min_interval=0)
    runner.client_kwargs["api_key"] = "SECRET_KEY_MARKER"
    with pytest.raises(RuntimeError) as error:
        asyncio.run(runner.chat(agent, "account balance"))
    assert str(error.value) == (f"Blue request failed: InternalServerError; HTTP status: {status}; "
                                f"Retry-After: {expected}; HTTP attempts: 1")
    assert len(calls) == runner._blue_budget.count == 1
    assert "SECRET" not in str(error.value)


def test_connection_error_reports_unavailable_status(monkeypatch):
    calls = []
    def respond(request):
        calls.append(request)
        raise httpx.ConnectError("SECRET_CONNECTION_DETAIL", request=request)

    monkeypatch.setattr(runtime.httpx, "HTTPTransport", lambda **kwargs: httpx.MockTransport(respond))
    agent, runner = runtime.create_blue_pair(name="blue", instruction="test", app_name="test")
    runner._blue_budget = runtime._BlueBudget(min_interval=0)
    runner.client_kwargs["api_key"] = "SECRET_KEY_MARKER"
    with pytest.raises(RuntimeError) as error:
        asyncio.run(runner.chat(agent, "account balance"))
    assert str(error.value) == ("Blue request failed: APIConnectionError; HTTP status: unavailable; "
                                "Retry-After: unavailable; HTTP attempts: 1")
    assert len(calls) == runner._blue_budget.count == 1


def test_independent_runners_have_bounded_independent_budgets(monkeypatch):
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
    monkeypatch.setattr(runtime.httpx, "HTTPTransport", lambda **kwargs: httpx.MockTransport(respond))
    monkeypatch.setenv("BLUE_MIN_INTERVAL_SECONDS", "0")
    for _ in range(2):
        agent, runner = runtime.create_blue_pair(name="blue", instruction="test", app_name="test")
        runner.client_kwargs["api_key"] = "fake-test-key"
        for _ in range(15):
            assert asyncio.run(runner.chat(agent, "balance")) == "OK"
        assert runner._blue_budget.count == 15
        for _ in range(3):
            asyncio.run(runner.chat(agent, "balance"))
        with pytest.raises(RuntimeError, match="HTTP attempts: 18"):
            asyncio.run(runner.chat(agent, "balance"))
        assert runner._blue_budget.count == 18
    assert len(calls) == 36
