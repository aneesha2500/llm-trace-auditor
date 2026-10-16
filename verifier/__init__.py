"""The deterministic verifier: each tool's checks live in its own module, on contract instances."""

from contract import Task, Tool

from . import audit, traces


class _List:
    verify_input = staticmethod(traces.verify_list_input)
    verify_output = staticmethod(traces.verify_list_output)


class _Trace:
    verify_input = staticmethod(traces.verify_trace_input)
    verify_output = staticmethod(traces.verify_trace_output)


TOOLS = {Tool.LIST_TRACES: _List, Tool.GET_TRACE: _Trace, Tool.ANALYZE_TRACE: audit}
# task -> its output's check, verify_task(arguments, output, reference, workspace)
TASKS = {Task.AUDIT_TRACE: audit.verify_task}


def verify_input(tool, value):
    return TOOLS[Tool(tool)].verify_input(value)


def verify_output(tool, input_value, output_value, workspace):
    return TOOLS[Tool(tool)].verify_output(input_value, output_value, workspace)


def verify_task(task, arguments, output, reference, workspace):
    return TASKS[Task(task)](arguments, output, reference, workspace)
