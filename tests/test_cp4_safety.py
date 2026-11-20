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


def _checkpoint(tmp_path):
    return attacks.CP4Checkpoint("gemini", "offline-model", directory=tmp_path)


def test_checkpoint_preserves_first_real_classification_on_timeout(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    monkeypatch.setattr(attacks, "GUARDS_SECRETS", ["fakeSensitive42"])
    monkeypatch.setattr(attacks, "check_secret_leak", lambda _: False)
    monkeypatch.setenv("CP4_CALL_TIMEOUT_SECONDS", "0.01")
    calls = []
    async def reply(agent, runner, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "fakeSensitive42", None
        await asyncio.Event().wait()
    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    with pytest.raises(attacks.CP4CallTimeout):
        asyncio.run(attacks.run_attacks(None, None, prompts=attacks.adversarial_prompts[:2],
                                         target_name="red_default", save_json=False,
                                         checkpoint=checkpoint))
    content = checkpoint.path.read_text(encoding="utf-8")
    data = json.loads(content)
    assert data["status"] == "incomplete" and data["error_category"] == "timeout"
    assert len(data["completed_prompts"]) == 1 and data["completed_prompts"][0]["leaked"] is True
    assert data["completed_prompts"][0]["response_preview"] == "[response omitted]"
    assert "fakeSensitive42" not in content and len(calls) == 2
    assert not (tmp_path / "attack_results.json").exists()


@pytest.mark.parametrize("failure", ["provider", "nemo", "cancelled"])
def test_checkpoint_failure_stops_without_fabricated_rows(monkeypatch, tmp_path, failure):
    checkpoint = _checkpoint(tmp_path)
    calls = []
    async def reply(agent, runner, prompt):
        calls.append(prompt)
        if failure == "provider":
            raise RuntimeError("fake secret provider detail")
        if failure == "cancelled":
            raise asyncio.CancelledError()
        return "Request blocked: NeMo Guardrails input check failed (Error).", None
    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    expected = asyncio.CancelledError if failure == "cancelled" else RuntimeError
    with pytest.raises(expected):
        asyncio.run(attacks.run_attacks(None, None, target_name="red_advance", save_json=False,
                                         checkpoint=checkpoint))
    data = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    assert data["status"] == "incomplete" and not data["completed_prompts"]
    assert data["error_category"] == {"provider": "provider_error", "nemo": "nemo_error", "cancelled": "cancelled"}[failure]
    assert len(calls) == 1 and "fake secret provider detail" not in checkpoint.path.read_text(encoding="utf-8")


def test_atomic_failure_preserves_existing_checkpoint_and_final(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    before = checkpoint.path.read_text(encoding="utf-8")
    final = tmp_path / "detail.json"
    final.write_text('{"existing": true}', encoding="utf-8")
    def fail_replace(source, destination):
        raise OSError("fake replace failure")
    monkeypatch.setattr(attacks.os, "replace", fail_replace)
    with pytest.raises(OSError):
        checkpoint.set_target("red_advance")
    with pytest.raises(OSError):
        attacks.write_run_attack_json([], target_name="red_default", filepath=final)
    assert checkpoint.path.read_text(encoding="utf-8") == before
    assert json.loads(final.read_text(encoding="utf-8")) == {"existing": True}
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*.tmp"))


def test_checkpoints_are_distinct_and_never_merge(tmp_path):
    first, second = _checkpoint(tmp_path), _checkpoint(tmp_path)
    first.record({"id": 1, "leaked": True, "blocked_at": "untrusted"})
    a = json.loads(first.path.read_text(encoding="utf-8"))
    b = json.loads(second.path.read_text(encoding="utf-8"))
    assert first.path != second.path and a["run_id"] != b["run_id"]
    assert b["stale_checkpoint_policy"] == "different_run_id_is_stale"
    assert len(a["completed_prompts"]) == 1 and b["completed_prompts"] == []
    assert a["completed_prompts"][0]["blocked_at"] == "[classification detail omitted]"


def test_mocked_full_cp4_writes_compatible_artifacts(monkeypatch, tmp_path):
    import main
    from agents import agent, guards_agent
    from core import config, utils
    monkeypatch.setattr(attacks, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(config, "get_llm_provider", lambda: "gemini")
    monkeypatch.setattr(config, "get_model_name", lambda: "offline-model")
    monkeypatch.setattr(agent, "create_red_agent_default", lambda: (object(), object()))
    monkeypatch.setattr(guards_agent, "create_red_agent_advance", lambda: (object(), object()))
    monkeypatch.setattr(attacks, "GUARDS_SECRETS", ["fakeSensitive42"])
    monkeypatch.setattr(attacks, "check_secret_leak", lambda _: False)
    async def reply(agent, runner, prompt):
        return "fakeSensitive42", None
    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    monkeypatch.setattr(utils, "chat_with_agent", reply)
    asyncio.run(main.part4_attacks())
    out = tmp_path / "outputs"
    detail_red = json.loads((out / "unsafe_attack_result.json").read_text(encoding="utf-8"))
    detail_guard = json.loads((out / "guards_attack_result.json").read_text(encoding="utf-8"))
    combined = json.loads((out / "attack_results.json").read_text(encoding="utf-8"))
    checkpoint_files = list((out / "cp4_progress").glob("*.json"))
    assert len(checkpoint_files) == 1
    progress = json.loads(checkpoint_files[0].read_text(encoding="utf-8"))
    assert progress["status"] == "completed" and len(progress["completed_prompts"]) == 10
    assert progress["provider"] == combined["llm_provider"] == "gemini"
    assert progress["model"] == combined["llm_model"] == "offline-model"
    assert len(detail_red["results"]) == len(detail_guard["results"]) == 5
    assert len(combined["unsafe_attacks"]) == len(combined["guards_attacks"]) == 5
    assert all(row["leaked"] for row in combined["unsafe_attacks"] + combined["guards_attacks"])
    for p in [checkpoint_files[0], out / "unsafe_attack_result.json", out / "guards_attack_result.json", out / "attack_results.json"]:
        assert "fakeSensitive42" not in p.read_text(encoding="utf-8")


@pytest.mark.parametrize("filename", ["attack_results.json", "unsafe_attack_result.json", "guards_attack_result.json"])
def test_existing_final_artifact_stops_before_checkpoint_and_provider(monkeypatch, tmp_path, filename):
    import main
    from agents import agent
    monkeypatch.setattr(attacks, "_repo_root", lambda: tmp_path)
    out = tmp_path / "outputs"
    out.mkdir()
    artifact = out / filename
    artifact.write_text('{"prior": true}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(agent, "create_red_agent_default", lambda: calls.append(1))
    with pytest.raises(RuntimeError, match=filename):
        asyncio.run(main.part4_attacks())
    assert calls == []
    assert artifact.read_text(encoding="utf-8") == '{"prior": true}'
    assert not (out / "cp4_progress").exists()


def test_red_completed_advance_failed_blocks_new_run(monkeypatch, tmp_path):
    import main
    from agents import agent, guards_agent
    from core import utils
    monkeypatch.setattr(attacks, "_repo_root", lambda: tmp_path)
    red_runner, guards_runner = object(), object()
    calls = []
    monkeypatch.setattr(agent, "create_red_agent_default", lambda: (object(), red_runner))
    monkeypatch.setattr(guards_agent, "create_red_agent_advance", lambda: (object(), guards_runner))
    async def reply(agent, runner, prompt):
        calls.append(runner)
        if runner is guards_runner:
            raise RuntimeError("fake provider secret")
        return "Safe banking reply", None
    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    monkeypatch.setattr(utils, "chat_with_agent", reply)
    with pytest.raises(RuntimeError, match="CP4 stopped"):
        asyncio.run(main.part4_attacks())
    out = tmp_path / "outputs"
    progress = json.loads(next((out / "cp4_progress").glob("*.json")).read_text(encoding="utf-8"))
    assert progress["status"] == "incomplete" and progress["target"] == "red_advance"
    assert len(progress["completed_prompts"]) == 5
    assert all(row["target"] == "red_default" for row in progress["completed_prompts"])
    assert not (out / "unsafe_attack_result.json").exists()
    assert not (out / "guards_attack_result.json").exists()
    assert not (out / "attack_results.json").exists()
    before = len(calls)
    with pytest.raises(RuntimeError, match="prior run checkpoint"):
        asyncio.run(main.part4_attacks())
    assert len(calls) == before


@pytest.mark.parametrize("failure", ["timeout", "cancelled", "provider", "nemo"])
def test_original_failure_survives_checkpoint_failure(monkeypatch, tmp_path, capsys, failure):
    checkpoint = _checkpoint(tmp_path)
    calls, snapshots = [], []
    provider_error = RuntimeError("fake secret provider detail")
    def failed_save(category):
        raise OSError("fake secret storage detail")
    monkeypatch.setattr(checkpoint, "fail", failed_save)
    async def reply(agent, runner, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "Safe banking reply", None
        snapshots.append(checkpoint.path.read_text(encoding="utf-8"))
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "cancelled":
            raise asyncio.CancelledError()
        if failure == "provider":
            raise provider_error
        return "Request blocked: NeMo Guardrails input check failed (Error).", None
    monkeypatch.setenv("CP4_CALL_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    expected = {"timeout": attacks.CP4CallTimeout, "cancelled": asyncio.CancelledError,
                "provider": RuntimeError, "nemo": attacks.CP4NemoError}[failure]
    with pytest.raises(expected) as observed:
        asyncio.run(attacks.run_attacks(None, None, target_name="red_advance", save_json=False,
                                         checkpoint=checkpoint))
    captured = capsys.readouterr()
    assert "CP4 checkpoint update failed" in captured.err
    assert "fake secret" not in captured.err + captured.out
    assert checkpoint.path.read_text(encoding="utf-8") == snapshots[0]
    assert len(json.loads(snapshots[0])["completed_prompts"]) == 1
    assert len(calls) == 2
    if failure == "provider":
        assert observed.value.__context__ is provider_error and observed.value.__suppress_context__
        assert "fake secret" not in str(observed.value)


def _prepare_mocked_publication(monkeypatch, tmp_path):
    import main
    from agents import agent, guards_agent
    from core import config, utils
    monkeypatch.setattr(attacks, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(config, "get_llm_provider", lambda: "gemini")
    monkeypatch.setattr(config, "get_model_name", lambda: "offline-model")
    monkeypatch.setattr(agent, "create_red_agent_default", lambda: (object(), object()))
    monkeypatch.setattr(guards_agent, "create_red_agent_advance", lambda: (object(), object()))
    async def reply(agent, runner, prompt):
        return "safe fake response", None
    monkeypatch.setattr(attacks, "chat_with_agent", reply)
    monkeypatch.setattr(utils, "chat_with_agent", reply)
    return main, tmp_path / "outputs"


def test_checkpoint_complete_failure_does_not_publish(monkeypatch, tmp_path, capsys):
    main, out = _prepare_mocked_publication(monkeypatch, tmp_path)
    def fail_complete(self):
        raise OSError("fake completion write failure")
    monkeypatch.setattr(attacks.CP4Checkpoint, "complete", fail_complete)
    with pytest.raises(OSError):
        asyncio.run(main.part4_attacks())
    assert not any((out / name).exists() for name in (
        "attack_results.json", "unsafe_attack_result.json", "guards_attack_result.json"))
    checkpoint = json.loads(next((out / "cp4_progress").glob("*.json")).read_text(encoding="utf-8"))
    assert checkpoint["status"] == "incomplete"
    assert len(checkpoint["completed_prompts"]) == 10
    assert "Saved attack evidence" not in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="prior run checkpoint"):
        asyncio.run(main.part4_attacks())


def test_detail_preparation_failure_does_not_publish_aggregate(monkeypatch, tmp_path):
    main, out = _prepare_mocked_publication(monkeypatch, tmp_path)
    original_dump = attacks.json.dump
    def fail_detail(payload, handle, **kwargs):
        if ".unsafe_attack_result.json." in str(handle.name):
            raise ValueError("fake serialization failure")
        return original_dump(payload, handle, **kwargs)
    monkeypatch.setattr(attacks.json, "dump", fail_detail)
    with pytest.raises(ValueError, match="fake serialization failure"):
        asyncio.run(main.part4_attacks())
    assert not (out / "attack_results.json").exists()
    assert not (out / "unsafe_attack_result.json").exists()
    assert not list(out.rglob(".*.tmp"))


@pytest.mark.parametrize("failed_name", ["unsafe_attack_result.json", "guards_attack_result.json", "attack_results.json"])
def test_publication_failure_has_no_aggregate_or_truncation(monkeypatch, tmp_path, capsys, failed_name):
    main, out = _prepare_mocked_publication(monkeypatch, tmp_path)
    original_link = attacks.os.link
    def fail_selected(source, destination):
        if Path(destination).name == failed_name:
            raise OSError("fake publication failure")
        return original_link(source, destination)
    monkeypatch.setattr(attacks.os, "link", fail_selected)
    with pytest.raises(OSError):
        asyncio.run(main.part4_attacks())
    assert "Lab 11 complete!" not in capsys.readouterr().out
    assert not (out / "attack_results.json").exists()
    assert not list(out.rglob("*.tmp")) and not list(out.rglob(".*.tmp"))
    progress = json.loads(next((out / "cp4_progress").glob("*.json")).read_text(encoding="utf-8"))
    assert progress["status"] == "incomplete"
    existing = {p.name: p.read_bytes() for p in out.glob("*.json")}
    with pytest.raises(RuntimeError, match="existing final artifacts|prior run checkpoint"):
        asyncio.run(main.part4_attacks())
    assert {p.name: p.read_bytes() for p in out.glob("*.json")} == existing


def test_publication_order_checkpoint_and_contract(monkeypatch, tmp_path):
    main, out = _prepare_mocked_publication(monkeypatch, tmp_path)
    original_link = attacks.os.link
    order = []
    def check_order(source, destination):
        progress = json.loads(next((out / "cp4_progress").glob("*.json")).read_text(encoding="utf-8"))
        assert progress["status"] == "completed"
        order.append(Path(destination).name)
        return original_link(source, destination)
    monkeypatch.setattr(attacks.os, "link", check_order)
    asyncio.run(main.part4_attacks())
    assert order == ["unsafe_attack_result.json", "guards_attack_result.json", "attack_results.json"]
    combined = json.loads((out / "attack_results.json").read_text(encoding="utf-8"))
    assert {"llm_provider", "llm_model", "unsafe_attacks", "guards_attacks", "summary"} <= combined.keys()
    assert len(combined["unsafe_attacks"]) == len(combined["guards_attacks"]) == 5
    assert all(row["response_preview"] == "[response omitted]" for row in combined["unsafe_attacks"] + combined["guards_attacks"])


def test_publication_never_overwrites_existing_file(tmp_path):
    target = tmp_path / "detail.json"
    target.write_text('{"prior": true}', encoding="utf-8")
    with pytest.raises(FileExistsError):
        attacks.write_run_attack_json([], target_name="red_default", filepath=target)
    assert target.read_text(encoding="utf-8") == '{"prior": true}'
