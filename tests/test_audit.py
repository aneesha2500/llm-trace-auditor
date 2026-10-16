from __future__ import annotations

from pathlib import Path

import pytest

from contract import ListTracesInput, TraceInput
from plugin import get_trace, list_traces
from plugin.audit import audit
from plugin.tracegen import DATA, render


def test_the_bundled_traces_are_reproducible() -> None:
    assert DATA.read_text(encoding="utf-8") == render()


def test_audit_on_a_hand_built_trace() -> None:
    trace = {
        "trace_id": "trace-99-hand",
        "prices": [{"model": "m", "input_per_million": 2.0, "output_per_million": 4.0}],
        "runs": [
            {"id": "b", "parent_id": None, "run_type": "chain", "start_ms": 0, "end_ms": 100, "status": "error",
             "prompt_tokens": 0, "completion_tokens": 0, "model": ""},
            {"id": "a", "parent_id": "b", "run_type": "llm", "start_ms": 1, "end_ms": 40, "status": "error",
             "prompt_tokens": 1000, "completion_tokens": 0, "model": "m"},
            {"id": "c", "parent_id": "b", "run_type": "llm", "start_ms": 50, "end_ms": 99, "status": "success",
             "prompt_tokens": 1000, "completion_tokens": 500, "model": "m"},
        ],
    }
    a = audit(trace)
    assert a.root_run_id == "b" and a.total_latency_ms == 100 and a.llm_calls == 2
    assert a.failed_run_ids == ["a", "b"] and a.prompt_tokens == 2000 and a.completion_tokens == 500
    assert a.cost_usd == round(2000 * 2 / 1e6 + 500 * 4 / 1e6, 6)
    assert a.slowest_llm_run_id == "c" and a.critical_path == ["b", "c"] and a.max_depth == 1


def test_tools() -> None:
    assert len(list_traces(ListTracesInput(contains=""), Path(".")).traces) == 8
    assert [t.trace_id for t in list_traces(ListTracesInput(contains="retry"), Path(".")).traces] == ["trace-02-retry"]
    with pytest.raises(ValueError):
        get_trace(TraceInput(trace_id="trace-99-none"), Path("."))
