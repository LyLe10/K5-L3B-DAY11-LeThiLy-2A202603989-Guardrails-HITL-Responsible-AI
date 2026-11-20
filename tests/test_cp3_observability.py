"""Offline CP3 suite and boundary checks; never call a model or write artifacts."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import validate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assignment import audit_log, monitoring, pipeline
from core import utils
from core.openai_runtime import OpenAIRunner
import agents.agent as agent_module


@pytest.mark.parametrize("payload", [
    "password: fake-password", "api_key: fake-key", "db_host=postgres.example.com",
    "database_host=postgres.example.com", "database host: postgres.example.com",
    "call 0901234567", "email person@example.com",
])
def test_egress_blocks_sensitive_payloads(payload):
    assert pipeline.is_egress_allowed("https://api.vinbank.example/v1/transfers", payload) is False


@pytest.mark.parametrize("destination", [
    "https://evil.example/collect", "https://api.vinbank.example.evil.com/v1",
    "http://api.vinbank.example/v1", "https://user@api.vinbank.example/v1",
    "https://api.vinbank.example:444/v1",
])
def test_egress_rejects_unapproved_destinations(destination):
    assert pipeline.is_egress_allowed(destination, "approved transfer amount 500000") is False


@pytest.mark.parametrize("payload", ["approved transfer amount 500000", "What is my account balance?"])
def test_egress_allows_benign_banking_payloads(payload):
    assert pipeline.is_egress_allowed("https://api.vinbank.example/v1/transfers", payload) is True


def _prepare(monkeypatch, tmp_path):
    audit, monitor = pipeline.build_observability()
    plugins = pipeline.build_production_plugins()
    runner = OpenAIRunner(app_name="offline", model="offline", plugins=plugins)
    monkeypatch.setattr(agent_module, "create_blue_agent", lambda plugins: (object(), runner))
    monkeypatch.setattr(audit_log, "default_audit_log_path", lambda: str(tmp_path / "audit_log.json"))
    monkeypatch.setattr(monitoring, "default_metrics_path", lambda: str(tmp_path / "metrics.json"))
    original = Path.write_text

    def safe_write(path, data, *args, **kwargs):
        if path.name == "results.json" and path.parent.name == "outputs":
            path = tmp_path / path.name
        assert path.parent == tmp_path
        return original(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", safe_write)
    return {"plugins": plugins, "audit": audit, "monitor": monitor}, runner


def test_suite_audits_real_interactions_and_separates_probes(monkeypatch, tmp_path):
    suite, runner = _prepare(monkeypatch, tmp_path)
    async def mock_chat(agent, ignored_runner, text):
        block = await runner._run_input_plugins(text)
        if block:
            return block, None
        response = ("Contact 0901234567 or person@example.com; key sk-test-secret123"
                    if text == "How do I transfer money?" else "Safe banking response.")
        return await runner._run_output_plugins(response), None
    monkeypatch.setattr(utils, "chat_with_agent", mock_chat)
    result = asyncio.run(pipeline.run_assignment_suite(suite))
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/results.schema.json").read_text(encoding="utf-8"))
    validate(result, schema)
    audit = json.loads((tmp_path / "audit_log.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert len(audit) == len(result["safe_queries"]) + len(result["attack_queries"]) + len(result["edge_cases"])
    assert metrics["total_requests"] == len(audit) + result["rate_limit"]["sent"]
    assert audit[0]["output"] == "Safe banking response."
    assert "[REDACTED]" in audit[1]["output"]
    assert all(secret not in json.dumps(audit) for secret in ("0901234567", "person@example.com", "sk-test-secret123"))
    blocked = [row for row in audit if row["blocked"]]
    assert blocked and all(row["output"] == "Request blocked by input guardrail." for row in blocked)
    assert result["rate_limit"]["blocked"] >= 1
    assert sum(row["blocked"] for row in result["safe_queries"]) == 0
    assert sum(row["blocked"] for row in result["attack_queries"]) >= 5


def test_failed_interaction_is_audited_without_payload(monkeypatch, tmp_path):
    suite, _ = _prepare(monkeypatch, tmp_path)
    async def fail(agent, runner, text):
        raise RuntimeError("sensitive provider error payload")
    monkeypatch.setattr(utils, "chat_with_agent", fail)
    with pytest.raises(RuntimeError):
        asyncio.run(pipeline.run_assignment_suite(suite))
    audit = json.loads((tmp_path / "audit_log.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert len(audit) == 1 and audit[0]["layer"] == "runtime_error"
    assert audit[0]["input"] == "What is my account balance?"
    assert audit[0]["output"] == "[omitted]"
    assert "sensitive provider error payload" not in json.dumps(audit)
    assert metrics["total_requests"] == 1
    assert not (tmp_path / "results.json").exists()


def test_unverified_response_is_omitted_and_input_is_redacted(monkeypatch, tmp_path):
    suite, _ = _prepare(monkeypatch, tmp_path)
    async def unverified(agent, runner, text):
        return "Unverified model output", None
    monkeypatch.setattr(utils, "chat_with_agent", unverified)
    asyncio.run(pipeline.run_assignment_suite(suite))
    audit = json.loads((tmp_path / "audit_log.json").read_text(encoding="utf-8"))
    assert all(row["output"] == "[omitted]" for row in audit)
    logger = audit_log.AuditLogPlugin()
    logger.record_input(user_id="test", text="Call 0901234567 or person@example.com")
    logger.record_output(user_id="test", text="not verified")
    assert "0901234567" not in json.dumps(logger.logs)
    assert "person@example.com" not in json.dumps(logger.logs)
    assert logger.logs[0]["output"] == "[omitted]"


def test_blue_chat_declares_no_action_tools():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Banking reply"))])
    runner = OpenAIRunner(app_name="offline", model="offline")
    runner._client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = asyncio.run(runner.chat(SimpleNamespace(instruction="test"), "account balance"))
    assert result == "Banking reply"
    assert "tools" not in calls[0] and "functions" not in calls[0]
