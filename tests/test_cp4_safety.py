"""Offline prompt and provider-failure checks; no model calls or artifacts."""
import asyncio
import base64
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attacks import attacks


def test_prompts_do_not_contain_protected_values():
    payload = json.loads((Path(__file__).resolve().parents[1] / "data/protected/vinbank_secrets.json").read_text(encoding="utf-8"))
    secrets = [v.casefold() for v in payload["secrets"].values() if isinstance(v, str) and v]
    assert len(attacks.adversarial_prompts) >= 5
    assert all(secret not in p["input"].casefold() for p in attacks.adversarial_prompts for secret in secrets)
    assert "Single-turn" in attacks.adversarial_prompts[4]["category"]


@pytest.mark.parametrize("status", [429, 503])
def test_provider_http_error_stops_before_next_prompt(monkeypatch, capsys, status):
    calls = []
    async def failing(agent, runner, prompt):
        calls.append(prompt)
        request = httpx.Request("POST", "https://example.invalid")
        response = httpx.Response(status, request=request, text="SECRET_ERROR_BODY")
        raise httpx.HTTPStatusError("SECRET_ERROR_BODY", request=request, response=response)
    monkeypatch.setattr(attacks, "chat_with_agent", failing)
    with pytest.raises(RuntimeError, match="CP4 stopped") as error:
        asyncio.run(attacks.run_attacks(None, None, save_json=False))
    assert len(calls) == 1
    assert "SECRET_ERROR_BODY" not in str(error.value) + capsys.readouterr().out


def test_nemo_failure_stops_without_next_prompt(monkeypatch):
    calls = []
    async def failure(agent, runner, prompt):
        calls.append(prompt)
        return "Request blocked: NeMo Guardrails input check failed (Error).", None
    monkeypatch.setattr(attacks, "chat_with_agent", failure)
    with pytest.raises(RuntimeError, match="CP4 stopped"):
        asyncio.run(attacks.run_attacks(None, None, target_name="red_advance", save_json=False))
    assert len(calls) == 1


def test_sensitive_response_not_logged(monkeypatch, capsys, tmp_path):
    secret = json.loads((Path(__file__).resolve().parents[1] / "data/protected/vinbank_secrets.json").read_text(encoding="utf-8"))["secrets"]["admin_password"]
    async def reply(agent, runner, prompt):
        return secret, None
    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    rows = asyncio.run(attacks.run_attacks(None, None, prompts=attacks.adversarial_prompts[:1], save_json=False))
    assert rows[0]["leaked"] is True
    assert secret not in capsys.readouterr().out
    assert secret not in rows[0]["response_preview"]
    for writer, target in ((attacks.write_run_attack_json, tmp_path / "detail.json"),
                           (attacks.save_attack_results, tmp_path / "summary.json")):
        if writer is attacks.write_run_attack_json:
            writer(rows, target_name="red_default", filepath=target)
        else:
            writer(unsafe_results=rows, filepath=target)
        assert secret not in target.read_text(encoding="utf-8")


@pytest.mark.parametrize("response", ["demoSecret42", "demo-Secret42"])
def test_fake_leaks_are_omitted_from_both_artifacts(monkeypatch, tmp_path, capsys, response):
    monkeypatch.setattr(attacks, "GUARDS_SECRETS", ["demoSecret42"])
    monkeypatch.setattr(attacks, "check_secret_leak", lambda _: False)
    async def reply(agent, runner, prompt):
        return response, None
    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    rows = asyncio.run(attacks.run_attacks(None, None, prompts=attacks.adversarial_prompts[:1], save_json=False))
    assert rows[0]["leaked"] is True
    assert "response" not in rows[0]
    assert response not in capsys.readouterr().out
    detail, combined = tmp_path / "detail.json", tmp_path / "combined.json"
    attacks.write_run_attack_json(rows, target_name="red_default", filepath=detail)
    attacks.save_attack_results(unsafe_results=rows, filepath=combined)
    for artifact in (detail, combined):
        content = artifact.read_text(encoding="utf-8")
        assert response not in content
        assert "[response omitted]" in content
        json.loads(content)


@pytest.mark.parametrize("response", ["ZGVtb1NlY3JldDQy", "24terceSomed"])
def test_encoded_fake_secret_is_omitted_without_fabricating_leak(monkeypatch, tmp_path, response):
    monkeypatch.setattr(attacks, "GUARDS_SECRETS", ["demoSecret42"])
    monkeypatch.setattr(attacks, "check_secret_leak", lambda _: False)
    async def reply(agent, runner, prompt):
        return response, None
    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    rows = asyncio.run(attacks.run_attacks(None, None, prompts=attacks.adversarial_prompts[:1], save_json=False))
    assert rows[0]["leaked"] is False
    assert rows[0]["response_preview"] == "[response omitted]"
    detail, combined = tmp_path / "detail.json", tmp_path / "combined.json"
    attacks.write_run_attack_json(rows, target_name="red_default", filepath=detail)
    attacks.save_attack_results(unsafe_results=rows, filepath=combined)
    assert response not in detail.read_text(encoding="utf-8")
    assert response not in combined.read_text(encoding="utf-8")


@pytest.mark.parametrize("preview", [None, "", "missing"])
def test_compact_never_falls_back_to_raw_response(preview, tmp_path):
    row = {"response": "FAKE_RAW_RESPONSE"}
    if preview != "missing":
        row["response_preview"] = preview
    path = tmp_path / "combined.json"
    attacks.save_attack_results(unsafe_results=[row], filepath=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["unsafe_attacks"][0]["response_preview"] == "[response omitted]"
    assert "FAKE_RAW_RESPONSE" not in path.read_text(encoding="utf-8")


def test_safe_preview_and_metadata_are_serialized(tmp_path):
    row = {"id": 1, "category": "fake", "input": "test", "response_preview": "Safe reply", "leaked": False}
    detail, combined = tmp_path / "detail.json", tmp_path / "combined.json"
    attacks.write_run_attack_json([row], target_name="red_default", filepath=detail)
    attacks.save_attack_results(unsafe_results=[row], filepath=combined)
    assert json.loads(detail.read_text(encoding="utf-8"))["results"][0]["response_preview"] == "[response omitted]"
    payload = json.loads(combined.read_text(encoding="utf-8"))
    assert payload["unsafe_attacks"][0]["response_preview"] == "[response omitted]"
    assert "llm_provider" in payload and "llm_model" in payload


@pytest.mark.parametrize("response", [
    "demoSecret42", "demo-Secret42", "ZGVtb1NlY3JldDQy", "24terceSomed",
    base64.b64encode(base64.b64encode(b"demoSecret42")).decode(),
    "Ordinary banking reply.",
])
def test_metadata_only_preserves_classification_and_contract(monkeypatch, tmp_path, response):
    monkeypatch.setattr(attacks, "GUARDS_SECRETS", ["demoSecret42"])
    monkeypatch.setattr(attacks, "check_secret_leak", lambda _: False)
    prompt = {"id": 7, "category": "fake", "input": "banking test"}
    expected = attacks.classify_attack_outcome(prompt["input"], response, target_name="red_default")
    async def reply(agent, runner, text):
        return response, None
    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    rows = asyncio.run(attacks.run_attacks(None, None, prompts=[prompt], target_name="red_default", save_json=False))
    assert rows[0]["leaked"] == expected["leaked"]
    assert rows[0]["blocked"] == expected["blocked"]
    detail, combined = tmp_path / "detail.json", tmp_path / "combined.json"
    attacks.write_run_attack_json(rows, target_name="red_default", filepath=detail)
    attacks.save_attack_results(unsafe_results=rows, guards_results=rows, filepath=combined)
    d = json.loads(detail.read_text(encoding="utf-8"))
    c = json.loads(combined.read_text(encoding="utf-8"))
    assert {"target", "leaks", "blocked_input", "blocked_plugin", "model_refuse", "results"} <= d.keys()
    assert {"llm_provider", "llm_model", "unsafe_attacks", "guards_attacks", "summary"} <= c.keys()
    for row in (d["results"][0], c["unsafe_attacks"][0], c["guards_attacks"][0]):
        assert {"id", "category", "input", "response_preview", "leaked", "blocked", "target"} <= row.keys()
        assert row["response_preview"] == "[response omitted]"
        assert row["leaked"] == expected["leaked"] and row["blocked"] == expected["blocked"]
    assert response not in detail.read_text(encoding="utf-8") + combined.read_text(encoding="utf-8")


def test_untrusted_rail_and_writer_fields_do_not_persist(monkeypatch, tmp_path):
    response = "Request blocked by NeMo Guardrails (input rail: fake-private-rail)."
    async def reply(agent, runner, text):
        return response, None
    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    rows = asyncio.run(attacks.run_attacks(None, None, prompts=[{"id": 1, "category": "test", "input": "test"}], target_name="red_advance", save_json=False))
    assert rows[0]["blocked"] and rows[0]["layer"] == "nemo_input_rail"
    assert "fake-private-rail" not in rows[0]["blocked_at"]
    rows[0].update(blocked_at="fake-private-rail", error="fake-private-rail", response_preview="fake-private-rail")
    detail, combined = tmp_path / "detail.json", tmp_path / "combined.json"
    attacks.write_run_attack_json(rows, target_name="red_advance", filepath=detail)
    attacks.save_attack_results(guards_results=rows, filepath=combined)
    assert "fake-private-rail" not in detail.read_text(encoding="utf-8") + combined.read_text(encoding="utf-8")


@pytest.mark.parametrize("provider", ["gemini", "openai"])
def test_nemo_public_model_config_disables_retries(monkeypatch, provider):
    from agents.red_advance_nemo import build_rails_config
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    config = build_rails_config(provider=provider, model="offline-test-model")
    assert config.models[0].parameters["max_retries"] == 0


def test_gemini_factories_use_one_attempt_without_changing_model(monkeypatch):
    from agents import agent, guards_agent
    from core.config import get_red_model
    monkeypatch.setenv("RED_TEAM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setattr(guards_agent, "NemoRailsGuard", lambda: object())
    for factory in (agent.create_red_agent_default, guards_agent.create_red_agent_advance):
        target, _ = factory()
        assert target.model.model == get_red_model()
        assert target.model.retry_options.attempts == 1


def test_progress_success_and_blocked_hide_payload(monkeypatch, capsys):
    async def reply(agent, runner, prompt):
        return "I'm a VinBank assistant and can only help with banking-related questions.", None
    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    prompt = {"id": 1, "category": "SECRET_CATEGORY", "input": "SECRET_PROMPT"}
    rows = asyncio.run(attacks.run_attacks(None, None, prompts=[prompt],
                                           target_name="red_advance", save_json=False))
    log = capsys.readouterr().out
    assert rows[0]["blocked"] is True
    assert "outcome=started" in log and "outcome=blocked" in log
    assert "start=" in log and "end=" in log and "elapsed_seconds=" in log
    assert "SECRET_PROMPT" not in log and "SECRET_CATEGORY" not in log


def test_timeout_cancels_call_and_stops_without_retry(monkeypatch, capsys):
    from core.cp4_progress import CP4CallTimeout
    calls, cancelled = [], []
    async def slow(agent, runner, prompt):
        calls.append(1)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
    monkeypatch.setenv("CP4_CALL_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setattr(attacks, "chat_with_agent", slow)
    with pytest.raises(CP4CallTimeout, match="timed out"):
        asyncio.run(attacks.run_attacks(None, None, save_json=False))
    assert calls == [1] and cancelled == [True]
    assert "outcome=timeout" in capsys.readouterr().out


def test_external_cancellation_propagates_and_cleans_up(capsys):
    from core.cp4_progress import run_cp4_call
    cancelled = []
    async def run():
        started = asyncio.Event()
        async def pending():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)
        task = asyncio.create_task(run_cp4_call(pending, target="red_default", identifier="smoke"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert cancelled == [True]
    assert "outcome=error" in capsys.readouterr().out


def test_progress_completed_and_invalid_timeout(monkeypatch, capsys):
    from core.cp4_progress import run_cp4_call
    async def response():
        return "SECRET_RESPONSE"
    assert asyncio.run(run_cp4_call(response, target="red_default", identifier=1)) == "SECRET_RESPONSE"
    log = capsys.readouterr().out
    assert "outcome=completed" in log and "SECRET_RESPONSE" not in log
    monkeypatch.setenv("CP4_CALL_TIMEOUT_SECONDS", "nan")
    with pytest.raises(ValueError, match="Invalid CP4_CALL_TIMEOUT_SECONDS"):
        asyncio.run(run_cp4_call(response, target="red_default", identifier=1))
