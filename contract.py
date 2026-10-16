"""Each tool's and task's typed input and output (see nanoagent.plugin.contract): declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    LIST_TRACES = "list_traces"
    GET_TRACE = "get_trace"
    ANALYZE_TRACE = "analyze_trace"


class Task(Enum):
    AUDIT_TRACE = "audit-trace"


TRACE_ID = "A trace id, as list_traces gives it, e.g. trace-02-retry"
RUN_ID = "A run id within the trace, e.g. run-07"


@dataclass
class ListTracesInput(Output):
    contains: Annotated[str, "Only traces whose id or description contains this text; empty for every trace"]


@dataclass
class TraceSummary(Output):
    trace_id: Annotated[str, TRACE_ID]
    description: Annotated[str, "What the trace shows"]
    runs: Annotated[int, "How many runs the trace holds"]


@dataclass
class ListTracesOutput(Output):
    traces: Annotated[list[TraceSummary], "The matching traces, by trace id"]


@dataclass
class Price(Output):
    model: Annotated[str, "The model name, as llm runs give it"]
    input_per_million: Annotated[float, "USD per million prompt tokens"]
    output_per_million: Annotated[float, "USD per million completion tokens"]


@dataclass
class TraceRun(Output):
    id: Annotated[str, RUN_ID]
    parent_id: Annotated[str | None, "The parent run's id, or null for the root"]
    name: Annotated[str, "The runnable's name"]
    run_type: Annotated[str, "chain, llm, retriever, prompt, parser or tool"]
    start_ms: Annotated[int, "Start, in ms from the trace start"]
    end_ms: Annotated[int, "End, in ms from the trace start"]
    status: Annotated[str, "success or error"]
    error: Annotated[str, "The error, or empty"]
    model: Annotated[str, "The model of an llm run, or empty"]
    prompt_tokens: Annotated[int, "Prompt tokens of an llm run, else 0"]
    completion_tokens: Annotated[int, "Completion tokens of an llm run, else 0"]


@dataclass
class TraceInput(Output):
    trace_id: Annotated[str, TRACE_ID]


@dataclass
class TraceOutput(Output):
    trace_id: Annotated[str, TRACE_ID]
    description: Annotated[str, "What the trace shows"]
    prices: Annotated[list[Price], "The price table recorded with the trace"]
    runs: Annotated[list[TraceRun], "Every run, sorted by id (not in tree order)"]


@dataclass
class AuditOutput(Output):
    trace_id: Annotated[str, TRACE_ID]
    root_run_id: Annotated[str, "The run with no parent"]
    run_count: Annotated[int, "How many runs the trace holds"]
    total_latency_ms: Annotated[int, "The root run's end_ms minus its start_ms"]
    llm_calls: Annotated[int, "How many runs have run_type llm, failed ones included"]
    failed_run_ids: Annotated[list[str], "The ids of every run with status error, sorted"]
    prompt_tokens: Annotated[int, "Prompt tokens summed over every llm run"]
    completion_tokens: Annotated[int, "Completion tokens summed over every llm run"]
    cost_usd: Annotated[float, "Sum over llm runs of the tokens times their model's price per million, rounded to 6 decimals"]
    slowest_llm_run_id: Annotated[str, "The llm run with the longest end_ms - start_ms; on a tie, the smallest id"]
    critical_path: Annotated[
        list[str],
        "Run ids from the root down: each step goes to the child that ends last (on a tie, the smallest id),"
        " until a run with no children",
    ]
    max_depth: Annotated[int, "The deepest run's depth, the root being 0"]


TOOLS = {
    Tool.LIST_TRACES: (ListTracesInput, ListTracesOutput),
    Tool.GET_TRACE: (TraceInput, TraceOutput),
    Tool.ANALYZE_TRACE: (TraceInput, AuditOutput),
}
TASKS = {Task.AUDIT_TRACE: (TraceInput, AuditOutput)}
