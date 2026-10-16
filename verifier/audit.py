"""analyze_trace (the audit-trace reference) and the audit-trace task's outcome check.

The agent's audit is compared, field by field, with the one analyze_trace computed from the same
trace, pinned before the agent started. Counts, ids and paths must match exactly (lists in
order), and cost within 0.0000005 USD. The score is the fraction of the eleven audited fields that
are right; the task is valid only when all are.
"""

import math

from .checks import COST_TOLERANCE, fields, result, trace_id_problem


def verify_input(value):
    problem = trace_id_problem(value.trace_id)
    return result(False, problem) if problem else result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    if output_value.trace_id != input_value.trace_id:
        return result(False, "audit is for another trace")
    if not output_value.critical_path or output_value.critical_path[0] != output_value.root_run_id:
        return result(False, "the critical path must start at the root run")
    if list(output_value.failed_run_ids) != sorted(set(output_value.failed_run_ids)):
        return result(False, "failed_run_ids must be unique and sorted")
    if output_value.max_depth < len(output_value.critical_path) - 1:
        return result(False, "max_depth is shallower than the critical path")
    if not math.isfinite(output_value.cost_usd) or output_value.cost_usd < 0:
        return result(False, "cost must be a non-negative number")
    return result(True, "audit is well formed")


def _right(name, got, want):
    if name == "cost_usd":
        return math.isfinite(got) and abs(got - want) <= COST_TOLERANCE + 1e-12
    return got == want


def verify_task(arguments, output, reference, workspace):
    """``output`` and ``reference``: contract AuditOutputs for ``arguments.trace_id``."""
    if output.trace_id != arguments.trace_id:
        return result(False, f"the audit is for {output.trace_id}, not {arguments.trace_id}", score=0.0)
    got, want = fields(output), fields(reference)
    wrong = {name: {"submitted": got[name], "expected": want[name]} for name in want if not _right(name, got[name], want[name])}
    score = (len(want) - len(wrong)) / len(want)
    if wrong:
        return result(False, f"{len(wrong)} of {len(want)} audited field(s) are wrong", score=score, wrong=wrong)
    return result(True, f"all {len(want)} audited fields match", score=1.0)
