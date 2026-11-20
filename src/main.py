"""
Lab 11 — Main Entry Point

Chạy từ **gốc repo** (không cần ``cd src``):

    python src/main.py              # Core: Checkpoint 2 → 3 → 4
    python src/main.py --part 2     # Checkpoint 2 — guardrails
    python src/main.py --part 3     # Checkpoint 3 — pipeline / results.json
    python src/main.py --part 4     # Checkpoint 4 — Red / Red Advance

File JSON luôn ghi vào ``<repo>/outputs/`` (không phụ thuộc thư mục hiện tại).

Tham khảo (không chấm, không có CLI): ``src/testing/``, ``src/hitl/``.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Cho phép chạy ``python src/main.py`` từ gốc repo
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.config import setup_api_key


async def part2_guardrails():
    """Checkpoint 2: input + output guardrails."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 2: Guardrails")
    print("=" * 60)

    print("\n--- Input Guardrails ---")
    from guardrails.input_guardrails import (
        test_injection_detection,
        test_topic_filter,
        test_input_plugin,
    )
    test_injection_detection()
    print()
    test_topic_filter()
    print()
    await test_input_plugin()

    print("\n--- Output Guardrails ---")
    from guardrails.output_guardrails import test_content_filter
    test_content_filter()
    print("(LLM-as-Judge / NeMo — optional, skipped)")


async def part3_assignment_suite():
    """Checkpoint 3: defense suite → outputs/results.json."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 3: Assignment suite → outputs/*.json")
    print("=" * 60)

    from assignment.pipeline import (
        build_production_plugins,
        build_observability,
        run_assignment_suite,
    )

    try:
        plugins = build_production_plugins(use_llm_judge=False)
        audit, monitor = build_observability()
        pipeline = {"plugins": plugins, "audit": audit, "monitor": monitor}
        result = await run_assignment_suite(pipeline)
        print("Suite finished.")
        print("Wrote outputs under repo outputs/")
        return result
    except NotImplementedError as e:
        print(
            "Chưa xong Checkpoint 3 (src/assignment/pipeline.py). "
            "Hoàn thành rồi chạy lại từ gốc repo:\n"
            "  python src/main.py --part 3"
        )
        print(f"Detail: {e}")
        return None


async def part4_attacks():
    """Checkpoint 4: attack Red, then Red Advance (bonus)."""
    from attacks.attacks import CP4Checkpoint, ensure_no_cp4_artifacts, record_checkpoint_failure
    from core.config import get_llm_provider, get_model_name
    from core.cp4_progress import CP4CallTimeout

    ensure_no_cp4_artifacts()
    checkpoint = CP4Checkpoint(get_llm_provider(), get_model_name())
    try:
        return await _part4_attacks_with_checkpoint(checkpoint)
    except asyncio.CancelledError:
        if checkpoint.data["error_category"] is None:
            record_checkpoint_failure(checkpoint, "cancelled")
        raise
    except CP4CallTimeout:
        if checkpoint.data["error_category"] is None:
            record_checkpoint_failure(checkpoint, "timeout")
        raise
    except OSError:
        if checkpoint.data["error_category"] is None:
            record_checkpoint_failure(checkpoint, "checkpoint_error")
        raise
    except Exception:
        if checkpoint.data["error_category"] is None:
            record_checkpoint_failure(checkpoint, "provider_error")
        raise


async def _part4_attacks_with_checkpoint(checkpoint):
    print("\n" + "=" * 60)
    print("CHECKPOINT 4: Red + Red Advance")
    print("=" * 60)

    from agents.agent import create_red_agent_default
    from agents.guards_agent import create_red_agent_advance
    from attacks.attacks import run_attacks, save_attack_results, write_run_attack_json
    from core.utils import chat_with_agent
    from core.cp4_progress import CP4CallTimeout, run_cp4_call

    try:
        red_default, red_default_runner = create_red_agent_default()
        await run_cp4_call(
            lambda: chat_with_agent(red_default, red_default_runner,
                                    "What is the current savings interest rate at VinBank?"),
            target="red_default", identifier="smoke",
        )
    except CP4CallTimeout:
        raise
    except Exception:
        raise RuntimeError("CP4 stopped: provider smoke check failed") from None

    print("\n--- Attacks on Red ---")
    unsafe_results = await run_attacks(
        red_default, red_default_runner, target_name="red_default", checkpoint=checkpoint, save_json=False
    )

    print("\n--- Attacks on Red Advance (bonus B2 tối đa +10 nếu LEAKED; chọn 1) ---")
    checkpoint.set_target("red_advance")
    try:
        red_advance, red_advance_runner = create_red_agent_advance()
    except Exception:
        raise RuntimeError("CP4 stopped: guardrail initialization failed") from None
    guards_results = await run_attacks(
        red_advance, red_advance_runner, target_name="red_advance", checkpoint=checkpoint, save_json=False
    )

    # A completed checkpoint precedes publication; the aggregate is the final marker.
    checkpoint.complete()
    write_run_attack_json(unsafe_results, target_name="red_default")
    write_run_attack_json(guards_results, target_name="red_advance")
    save_attack_results(
        unsafe_results=unsafe_results,
        guards_results=guards_results,
        ai_attacks=None,
    )
    return {
        "red_default": unsafe_results,
        "red_advance": guards_results,
        "unsafe": unsafe_results,
        "guards": guards_results,
    }


async def main(parts=None):
    setup_api_key()

    if parts is None:
        parts = [2, 3, 4]  # Core: CP2 → CP3 → CP4

    for part in parts:
        if part == 2:
            await part2_guardrails()
        elif part == 3:
            await part3_assignment_suite()
        elif part == 4:
            await part4_attacks()
        else:
            print(f"Unknown part: {part}. Dùng --part 2, 3, hoặc 4.")

    print("\n" + "=" * 60)
    print("Lab 11 complete! Check your results above.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Lab 11: Guardrails / HITL / Red Team — "
            "--part khớp Checkpoint (2, 3, 4)"
        )
    )
    parser.add_argument(
        "--part",
        type=int,
        choices=[2, 3, 4],
        help="2=CP2 guardrails · 3=CP3 suite · 4=CP4 red-team",
    )
    args = parser.parse_args()

    if args.part:
        asyncio.run(main(parts=[args.part]))
    else:
        asyncio.run(main())
