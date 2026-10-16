"""The bundled traces (data/traces.json, made by tracegen.py) and the audit the task asks for."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from contract import (
    AuditOutput,
    ListTracesInput,
    ListTracesOutput,
    Price,
    TraceInput,
    TraceOutput,
    TraceRun,
    TraceSummary,
)

DATA = Path(__file__).with_name("data") / "traces.json"


@cache
def traces() -> dict[str, dict]:
    return {trace["trace_id"]: trace for trace in json.loads(DATA.read_text(encoding="utf-8"))["traces"]}


def trace(trace_id: str) -> dict:
    if trace_id not in traces():
        raise ValueError(f"unknown trace_id {trace_id!r}; list_traces gives the trace ids")
    return traces()[trace_id]


def list_traces(given: ListTracesInput, workspace: Path) -> ListTracesOutput:
    return ListTracesOutput(
        traces=[
            TraceSummary(trace_id=t["trace_id"], description=t["description"], runs=len(t["runs"]))
            for t in sorted(traces().values(), key=lambda t: t["trace_id"])
            if given.contains in t["trace_id"] or given.contains in t["description"]
        ]
    )


def get_trace(given: TraceInput, workspace: Path) -> TraceOutput:
    t = trace(given.trace_id)
    return TraceOutput(
        trace_id=t["trace_id"],
        description=t["description"],
        prices=[Price(**p) for p in t["prices"]],
        runs=[TraceRun(**r) for r in t["runs"]],
    )


def audit(t: dict) -> AuditOutput:
    runs = {r["id"]: r for r in t["runs"]}
    [root] = [r for r in runs.values() if r["parent_id"] is None]
    children: dict[str, list[dict]] = {}
    for r in runs.values():
        if r["parent_id"] is not None:
            children.setdefault(r["parent_id"], []).append(r)
    prices = {p["model"]: p for p in t["prices"]}
    llm = [r for r in runs.values() if r["run_type"] == "llm"]
    cost = sum(
        r["prompt_tokens"] * prices[r["model"]]["input_per_million"] / 1e6
        + r["completion_tokens"] * prices[r["model"]]["output_per_million"] / 1e6
        for r in llm
    )
    slowest = min(llm, key=lambda r: (-(r["end_ms"] - r["start_ms"]), r["id"]))["id"] if llm else ""

    path, node = [root["id"]], root
    while children.get(node["id"]):
        node = min(children[node["id"]], key=lambda r: (-r["end_ms"], r["id"]))
        path.append(node["id"])

    def depth(r: dict) -> int:
        return 0 if r["parent_id"] is None else 1 + depth(runs[r["parent_id"]])

    return AuditOutput(
        trace_id=t["trace_id"],
        root_run_id=root["id"],
        run_count=len(runs),
        total_latency_ms=root["end_ms"] - root["start_ms"],
        llm_calls=len(llm),
        failed_run_ids=sorted(r["id"] for r in runs.values() if r["status"] == "error"),
        prompt_tokens=sum(r["prompt_tokens"] for r in llm),
        completion_tokens=sum(r["completion_tokens"] for r in llm),
        cost_usd=round(cost, 6),
        slowest_llm_run_id=slowest,
        critical_path=path,
        max_depth=max(depth(r) for r in runs.values()),
    )


def analyze_trace(given: TraceInput, workspace: Path) -> AuditOutput:
    """The audit-trace reference: the audit computed from the trace, pinned before the agent runs."""
    return audit(trace(given.trace_id))
