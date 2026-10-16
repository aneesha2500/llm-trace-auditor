"""list_traces and get_trace: the traces returned are the ones asked for, and well-formed trees."""

from .checks import result, trace_id_problem

RUN_TYPES = {"chain", "llm", "retriever", "prompt", "parser", "tool"}


def verify_list_input(value):
    return result(True, "input is valid")


def verify_list_output(input_value, output_value, workspace):
    ids = [trace.trace_id for trace in output_value.traces]
    if ids != sorted(set(ids)):
        return result(False, "traces must be unique and sorted by trace id")
    for trace in output_value.traces:
        if input_value.contains not in trace.trace_id and input_value.contains not in trace.description:
            return result(False, f"{trace.trace_id} does not match the filter")
    return result(True, f"{len(ids)} trace(s) listed")


def verify_trace_input(value):
    problem = trace_id_problem(value.trace_id)
    return result(False, problem) if problem else result(True, "input is valid")


def tree_problem(runs, prices):
    """What is wrong with a list of contract TraceRuns as a run tree, or None."""
    by_id = {run.id: run for run in runs}
    if len(by_id) != len(runs):
        return "two runs share an id"
    roots = [run for run in runs if run.parent_id is None]
    if len(roots) != 1:
        return "a trace has exactly one root run"
    models = {price.model for price in prices}
    for run in runs:
        if run.run_type not in RUN_TYPES or run.status not in ("success", "error"):
            return f"{run.id} has an unknown run_type or status"
        if run.end_ms < run.start_ms:
            return f"{run.id} ends before it starts"
        if run.parent_id is not None:
            parent = by_id.get(run.parent_id)
            if parent is None:
                return f"{run.id}'s parent {run.parent_id} is not in the trace"
            if run.start_ms < parent.start_ms or run.end_ms > parent.end_ms:
                return f"{run.id} runs outside its parent {parent.id}"
        if run.run_type == "llm" and run.model not in models:
            return f"{run.id} uses a model with no price"
        if run.run_type != "llm" and (run.prompt_tokens or run.completion_tokens):
            return f"{run.id} is not an llm run but has tokens"
    return None


def verify_trace_output(input_value, output_value, workspace):
    if output_value.trace_id != input_value.trace_id:
        return result(False, "output is for another trace")
    problem = tree_problem(output_value.runs, output_value.prices)
    if problem:
        return result(False, problem)
    return result(True, f"trace {output_value.trace_id} is a well-formed tree of {len(output_value.runs)} runs")
