"""How data/traces.json was made: eight LangChain-style run traces of a code-explanation agent.

Each trace is a tree of runs as LangChain's tracer records them (a root chain, retrievers, prompt
templates, chat-model calls, tools, output parsers), with millisecond start/end times relative to
the trace start, a status, and token counts on chat-model runs. The shapes cover what an audit
must handle: retries after a failed call, a tool-calling agent loop, parallel branches, an
LLM-as-judge call on a second model, a failed trace, deep nesting, and ties.

The generator is seeded, so `python -m plugin.tracegen` rewrites the bundled file byte for byte
(tests/test_audit.py checks this). Run ids are shuffled so the list order says nothing about the
tree.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA = Path(__file__).with_name("data") / "traces.json"
EXPLAINER = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
JUDGE = "Qwen/Qwen2.5-Coder-7B-Instruct"
# The price table recorded with every trace, USD per million tokens.
PRICES = [
    {"model": EXPLAINER, "input_per_million": 0.5, "output_per_million": 1.5},
    {"model": JUDGE, "input_per_million": 3.0, "output_per_million": 15.0},
]


class Builder:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.runs: list[dict] = []

    def run(self, name, run_type, parent, start, duration, *, status="success", error="", model="", tokens=(0, 0)):
        run = {
            "key": len(self.runs),
            "parent": parent,
            "name": name,
            "run_type": run_type,
            "start_ms": start,
            "end_ms": start + duration,
            "status": status,
            "error": error,
            "model": model,
            "prompt_tokens": tokens[0],
            "completion_tokens": tokens[1],
        }
        self.runs.append(run)
        return run

    def llm(self, parent, start, *, model=EXPLAINER, fail="", prompt=None):
        rng = self.rng
        prompt = prompt or rng.randint(350, 900)
        if fail:
            return self.run("ChatHuggingFace", "llm", parent["key"], start, rng.randint(400, 1500),
                            status="error", error=fail, model=model, tokens=(prompt, 0))
        completion = rng.randint(120, 420)
        return self.run("ChatHuggingFace", "llm", parent["key"], start, rng.randint(900, 4200), model=model,
                        tokens=(prompt, completion))

    def step(self, name, run_type, parent, start, low=0, high=3):
        return self.run(name, run_type, parent["key"], start, self.rng.randint(low, high))

    def finish(self, root, trace_id, description) -> dict:
        root["end_ms"] = max(r["end_ms"] for r in self.runs) + self.rng.randint(0, 2)
        order = list(range(len(self.runs)))
        self.rng.shuffle(order)
        ids = {key: f"run-{index + 1:02d}" for index, key in enumerate(order)}
        runs = [
            {
                "id": ids[r["key"]],
                "parent_id": ids[r["parent"]] if r["parent"] is not None else None,
                **{k: v for k, v in r.items() if k not in ("key", "parent")},
            }
            for r in self.runs
        ]
        runs.sort(key=lambda r: r["id"])
        return {"trace_id": trace_id, "description": description, "prices": PRICES, "runs": runs}


def _rag_chain(b: Builder, root, t: int, *, retry: bool = False) -> int:
    retrieve = b.step("lsh_retrieve", "retriever", root, t, 1, 6)
    t = retrieve["end_ms"] + 1
    prompt = b.step("ChatPromptTemplate", "prompt", root, t, 0, 2)
    t = prompt["end_ms"] + 1
    if retry:
        failed = b.llm(root, t, fail="TimeoutError: inference endpoint did not answer")
        t = failed["end_ms"] + b.rng.randint(200, 600)  # back-off before the retry
    call = b.llm(root, t)
    t = call["end_ms"] + 1
    parser = b.step("StrOutputParser", "parser", root, t, 0, 1)
    return parser["end_ms"]


def trace_simple(rng):
    b = Builder(rng)
    root = b.run("CodeExplainerAgent", "chain", None, 0, 0)
    _rag_chain(b, root, 1)
    return b.finish(root, "trace-01-simple", "One retrieval-augmented explanation: retrieve, prompt, LLM, parse.")


def trace_retry(rng):
    b = Builder(rng)
    root = b.run("CodeExplainerAgent", "chain", None, 0, 0)
    _rag_chain(b, root, 1, retry=True)
    return b.finish(root, "trace-02-retry", "The first LLM call times out and is retried after a back-off.")


def trace_tool_agent(rng):
    b = Builder(rng)
    root = b.run("CodeExplainerAgent[tool]", "chain", None, 0, 0)
    first = b.llm(root, 2)
    t = first["end_ms"] + 1
    tool = b.run("search_similar_examples", "tool", root["key"], t, 0)
    inner = b.step("lsh_retrieve", "retriever", tool, t + 1, 1, 5)
    fmt = b.step("format_examples", "chain", tool, inner["end_ms"] + 1, 0, 2)
    tool["end_ms"] = fmt["end_ms"] + 1
    second = b.llm(root, tool["end_ms"] + 2, prompt=rng.randint(900, 1600))
    b.step("StrOutputParser", "parser", root, second["end_ms"] + 1, 0, 1)
    return b.finish(root, "trace-03-tool-agent", "Tool-calling mode: the model asks for similar examples, then answers.")


def trace_parallel(rng):
    b = Builder(rng)
    root = b.run("CompareRetrievers", "chain", None, 0, 0)
    parallel = b.run("RunnableParallel<lsh,full_scan>", "chain", root["key"], 1, 0)
    lsh = b.step("lsh_retrieve", "retriever", parallel, 2, 1, 3)
    full = b.step("full_scan_retrieve", "retriever", parallel, 2, 8, 20)
    parallel["end_ms"] = max(lsh["end_ms"], full["end_ms"]) + 1
    t = parallel["end_ms"] + 1
    prompt = b.step("ChatPromptTemplate", "prompt", root, t, 0, 2)
    call = b.llm(root, prompt["end_ms"] + 1)
    b.step("StrOutputParser", "parser", root, call["end_ms"] + 1, 0, 1)
    return b.finish(root, "trace-04-parallel", "Two retrievers run in parallel before one LLM call.")


def trace_judge(rng):
    b = Builder(rng)
    root = b.run("ExplainAndGrade", "chain", None, 0, 0)
    explain = b.run("CodeExplainerAgent", "chain", root["key"], 1, 0)
    explain["end_ms"] = _rag_chain(b, explain, 2) + 1
    judge = b.run("JudgeChain", "chain", root["key"], explain["end_ms"] + 1, 0)
    prompt = b.step("ChatPromptTemplate", "prompt", judge, judge["start_ms"] + 1, 0, 2)
    call = b.llm(judge, prompt["end_ms"] + 1, model=JUDGE, prompt=rng.randint(700, 1200))
    parse = b.step("JsonOutputParser", "parser", judge, call["end_ms"] + 1, 0, 2)
    judge["end_ms"] = parse["end_ms"] + 1
    return b.finish(root, "trace-05-judge", "An explanation, then a second model grades it (two models, two prices).")


def trace_failed(rng):
    b = Builder(rng)
    root = b.run("CodeExplainerAgent", "chain", None, 0, 0, status="error", error="RuntimeError: retries exhausted")
    retrieve = b.step("lsh_retrieve", "retriever", root, 1, 1, 4)
    prompt = b.step("ChatPromptTemplate", "prompt", root, retrieve["end_ms"] + 1, 0, 2)
    t = prompt["end_ms"] + 1
    for attempt in range(3):
        failed = b.llm(root, t, fail=f"HTTPError: 503 Service Unavailable (attempt {attempt + 1})")
        t = failed["end_ms"] + rng.randint(200, 900)
    return b.finish(root, "trace-06-failed", "Every LLM call fails; the trace itself ends in error.")


def trace_nested(rng):
    b = Builder(rng)
    root = b.run("Course pipeline", "chain", None, 0, 0)
    batch = b.run("ExplainBatch", "chain", root["key"], 1, 0)
    t = 2
    for _ in range(2):
        item = b.run("CodeExplainerAgent", "chain", batch["key"], t, 0)
        item["end_ms"] = _rag_chain(b, item, t + 1) + 1
        t = item["end_ms"] + 1
    batch["end_ms"] = t
    b.step("write_report", "tool", root, t + 1, 2, 9)
    return b.finish(root, "trace-07-nested", "A batch of two explanations nested two chains deep, then a report.")


def trace_tie(rng):
    b = Builder(rng)
    root = b.run("CompareModels", "chain", None, 0, 0)
    left = b.run("ExplainA", "chain", root["key"], 1, 2000)
    right = b.run("ExplainB", "chain", root["key"], 1, 2000)  # ends at the same time as ExplainA
    for branch in (left, right):
        call = b.run("ChatHuggingFace", "llm", branch["key"], 2, 1999, model=EXPLAINER, tokens=(500, 200))
        assert call["end_ms"] == branch["end_ms"]
    return b.finish(root, "trace-08-tie", "Two branches and two LLM calls end at the same millisecond (tie rules).")


TRACES = (trace_simple, trace_retry, trace_tool_agent, trace_parallel, trace_judge, trace_failed, trace_nested, trace_tie)


def generate() -> dict:
    return {
        "description": "LangChain-style run traces of a retrieval-augmented code-explanation agent; times in ms "
        "from the trace start; prices in USD per million tokens.",
        "traces": [make(random.Random(480 + index)) for index, make in enumerate(TRACES)],
    }


def render() -> str:
    return json.dumps(generate(), indent=1) + "\n"


if __name__ == "__main__":
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(render(), encoding="utf-8")
