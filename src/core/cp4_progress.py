"""Bound and report CP4 calls without logging request or response content."""
import asyncio
import math
import os
import time
from datetime import datetime, timezone


class CP4CallTimeout(TimeoutError):
    """A CP4 call exceeded its application deadline."""


def _deadline_seconds():
    try:
        value = float(os.environ.get("CP4_CALL_TIMEOUT_SECONDS", "180"))
    except ValueError:
        raise ValueError("Invalid CP4_CALL_TIMEOUT_SECONDS") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid CP4_CALL_TIMEOUT_SECONDS")
    return value


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


async def run_cp4_call(call, *, target, identifier, is_blocked=None):
    """Wait for one call and propagate cancellation; never retry it."""
    target = target if target in {"red_default", "red_advance"} else "other"
    identifier = identifier if identifier == "smoke" or (type(identifier) is int and identifier > 0) else "unknown"
    started = _timestamp()
    start_clock = time.monotonic()
    print(f"CP4 target={target} call={identifier} outcome=started start={started}", flush=True)
    outcome = "error"
    try:
        deadline = _deadline_seconds()
        result = await asyncio.wait_for(call(), timeout=deadline)
        outcome = "blocked" if is_blocked and is_blocked(result) else "completed"
        return result
    except asyncio.TimeoutError:
        outcome = "timeout"
        raise CP4CallTimeout("CP4 stopped: agent call timed out") from None
    finally:
        elapsed = time.monotonic() - start_clock
        print(f"CP4 target={target} call={identifier} outcome={outcome} start={started} "
              f"end={_timestamp()} elapsed_seconds={elapsed:.3f}", flush=True)
