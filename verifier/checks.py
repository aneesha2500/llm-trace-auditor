"""Checks shared by the tool verifiers. Pure functions of their arguments."""

import re

TRACE_ID = re.compile(r"^trace-\d{2}-[a-z0-9-]+$")
COST_TOLERANCE = 0.0000005  # cost is reported to 6 decimals


def result(valid, message, *, score=None, **details):
    return {
        "valid": valid,
        "score": float(valid) if score is None else score,
        "message": message,
        "details": details,
    }


def trace_id_problem(trace_id):
    return None if TRACE_ID.fullmatch(trace_id) else f"{trace_id!r} is not a trace id like trace-02-retry"


def fields(audit):
    """A contract AuditOutput's audited fields as a plain mapping."""
    return {
        "root_run_id": audit.root_run_id,
        "run_count": audit.run_count,
        "total_latency_ms": audit.total_latency_ms,
        "llm_calls": audit.llm_calls,
        "failed_run_ids": list(audit.failed_run_ids),
        "prompt_tokens": audit.prompt_tokens,
        "completion_tokens": audit.completion_tokens,
        "cost_usd": audit.cost_usd,
        "slowest_llm_run_id": audit.slowest_llm_run_id,
        "critical_path": list(audit.critical_path),
        "max_depth": audit.max_depth,
    }
