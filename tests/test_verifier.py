from __future__ import annotations

from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
from contract import AuditOutput, ListTracesInput, TraceInput, TraceOutput
from plugin import analyze_trace, get_trace, list_traces

TRACE = "trace-02-retry"


def reference(trace_id: str = TRACE) -> AuditOutput:
    return build_output(plain_output(analyze_trace(TraceInput(trace_id=trace_id), Path("."))), AuditOutput)


def verify(changes: dict | None = None, trace_id: str = TRACE) -> dict:
    output = plain_output(reference(trace_id)) | (changes or {})
    return verifier.verify_task(
        "audit-trace", build_output({"trace_id": trace_id}, TraceInput), build_output(output, AuditOutput),
        reference(trace_id), ".",
    )


@pytest.mark.parametrize("trace_id", [f"trace-0{i}" for i in range(1, 9)])
def test_approves_the_exact_audit_of_every_trace(trace_id: str) -> None:
    [full] = [t.trace_id for t in list_traces(ListTracesInput(contains=trace_id), Path(".")).traces]
    verdict = verify(trace_id=full)
    assert verdict["valid"] is True and verdict["score"] == 1.0


def test_the_retry_trace_by_hand() -> None:
    # run-04 is the root; run-01 is the failed first call; run-06 is the successful retry.
    audit = reference()
    assert audit.root_run_id == "run-04" and audit.failed_run_ids == ["run-01"]
    assert audit.llm_calls == 2 and audit.slowest_llm_run_id == "run-06"


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"llm_calls": 1}, "llm_calls"),  # forgot the failed call
        ({"failed_run_ids": []}, "failed_run_ids"),
        ({"prompt_tokens": 492}, "prompt_tokens"),  # only the successful call
        ({"cost_usd": 0.0012}, "cost_usd"),
        ({"critical_path": ["run-04"]}, "critical_path"),
        ({"total_latency_ms": 5347}, "total_latency_ms"),
        ({"max_depth": 2}, "max_depth"),
        ({"root_run_id": "run-01"}, "root_run_id"),
    ],
)
def test_refuses_each_wrong_field_with_partial_score(changes: dict, field: str) -> None:
    verdict = verify(changes)
    assert verdict["valid"] is False and list(verdict["details"]["wrong"]) == [field]
    assert verdict["score"] == pytest.approx(10 / 11)


def test_cost_tolerance_is_half_a_unit_in_the_sixth_decimal() -> None:
    cost = reference().cost_usd
    assert verify({"cost_usd": cost + 0.0000004})["valid"] is True
    assert verify({"cost_usd": cost + 0.000001})["valid"] is False
    assert verify({"cost_usd": float("inf")})["valid"] is False


def test_tie_rules_on_the_tie_trace() -> None:
    audit = reference("trace-08-tie")
    # Both branches end at the same ms, and both llm calls last as long: the smaller id wins.
    assert audit.critical_path == ["run-03", "run-01", "run-02"]
    assert audit.slowest_llm_run_id == "run-02"
    verdict = verify({"critical_path": ["run-03", "run-05", "run-04"]}, trace_id="trace-08-tie")
    assert verdict["valid"] is False and "critical_path" in verdict["details"]["wrong"]


def test_refuses_an_audit_of_another_trace() -> None:
    verdict = verify({"trace_id": "trace-01-simple"})
    assert verdict["valid"] is False and verdict["score"] == 0.0


def test_checks_the_trace_id_and_tree() -> None:
    assert verifier.verify_input("get_trace", build_output({"trace_id": TRACE}, TraceInput))["valid"] is True
    assert verifier.verify_input("get_trace", build_output({"trace_id": "retry"}, TraceInput))["valid"] is False
    given = TraceInput(trace_id=TRACE)
    trace = build_output(plain_output(get_trace(given, Path("."))), TraceOutput)
    assert verifier.verify_output("get_trace", given, trace, ".")["valid"] is True
    trace.runs[0].end_ms = trace.runs[0].start_ms - 1
    assert verifier.verify_output("get_trace", given, trace, ".")["valid"] is False


def test_checks_the_reference_audit_is_well_formed() -> None:
    given = TraceInput(trace_id=TRACE)
    assert verifier.verify_output("analyze_trace", given, reference(), ".")["valid"] is True
    bad = build_output(plain_output(reference()) | {"critical_path": ["run-03"]}, AuditOutput)
    assert verifier.verify_output("analyze_trace", given, bad, ".")["valid"] is False
