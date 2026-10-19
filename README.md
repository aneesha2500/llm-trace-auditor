# LLM Trace Auditor

A `nanoagent.plugin/v1` plugin, built with the structure of NanoAgent's
reference plugin, [arxiv-source](https://github.com/zeyuyang8/arxiv-source).
An agent audits a LangChain-style run trace of a retrieval-augmented
code-explanation agent. It rebuilds the run tree from parent ids, then reports
latency, LLM calls, failures, tokens, cost and the critical path. The verifier
checks every field against an exact computation over the same trace.

## Tools

| Tool | Does | Code |
|---|---|---|
| `list_traces` | Lists the bundled traces (id, description, number of runs), optionally filtered | `src/plugin/audit.py` |
| `get_trace` | Returns a trace's runs (id, parent, type, start/end ms, status, model, tokens) and its price table | `src/plugin/audit.py` |
| `analyze_trace` | Computes the trace's audit by the rules below; the `audit-trace` reference | `src/plugin/audit.py` |

### Tasks

| Task | Does | Runner |
|---|---|---|
| `audit-trace` | Audits a trace's latency, LLM calls, failures, tokens, cost and critical path | `agent`, with `list_traces` and `get_trace`; reference `analyze_trace` |

### Audit rules

| Field | Rule |
|---|---|
| `root_run_id` | the run with no parent |
| `run_count` | number of runs |
| `total_latency_ms` | root `end_ms − start_ms` |
| `llm_calls` | runs with `run_type` `llm`, failed ones included |
| `failed_run_ids` | every run with status `error`, sorted |
| `prompt_tokens`, `completion_tokens` | summed over llm runs |
| `cost_usd` | Σ over llm runs of tokens × the model's price per million, rounded to 6 decimals |
| `slowest_llm_run_id` | llm run with the longest duration; ties go to the smallest id |
| `critical_path` | from the root, repeatedly step to the child that ends last (ties go to the smallest id) until a leaf |
| `max_depth` | deepest run, with the root at depth 0 |

Before the agent starts, NanoAgent runs `analyze_trace` and pins its output.
`verifier/audit.py` compares the agent's eleven fields with it:

- ids, counts and lists must match exactly, lists in order;
- cost must be within 0.0000005 USD.

The score is the fraction of fields that are right, and the verdict lists each
wrong one with its expected value. `get_trace`'s output is checked to be a
well-formed tree: one root, known parents, children inside their parent's time
span, a price for every model, and tokens only on llm runs.

### The data

`src/plugin/data/traces.json` holds eight traces of the code-explanation
agent (retrieve, prompt, chat model, parse) as LangChain's tracer records runs:

| Trace | What it shows |
|---|---|
| `trace-01-simple` | one chain |
| `trace-02-retry` | a timed-out call retried after back-off |
| `trace-03-tool-agent` | a tool-calling loop whose tool nests a retriever |
| `trace-04-parallel` | two retrievers in a `RunnableParallel` |
| `trace-05-judge` | a second, pricier judge model |
| `trace-06-failed` | every call fails and the root errors |
| `trace-07-nested` | two explanations nested three levels deep |
| `trace-08-tie` | branches and calls that tie, which tests the tie rules |

Run ids are shuffled, so list order says nothing about the tree.
`src/plugin/tracegen.py` is the seeded generator that wrote the file, and a test
checks that it reproduces the file byte for byte. The prices are the table
recorded with each trace, for the audit's arithmetic; they are not a claim about
any provider's current prices.

## Layout

```text
.
├── nanoagent.yaml   # manifest (a slimconfig ManifestConfig): id, runtime, permissions, tools, tasks
├── contract.py      # each tool's and task's typed input and output: slimconfig Output classes
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env             # environment variable NAMEs this plugin needs (none)
├── src/plugin/      # the MCP server (__main__.py, IMPLEMENTATIONS), audit.py, tracegen.py, data/
├── tests/           # test_verifier.py (correct, wrong, missing, malformed, boundary) and test_audit.py
├── ui/              # outputs drawn from their JSON Schema: renderers.js and renderers.css
├── product/         # the product page a catalog shows: product.yaml and icon.svg
└── verifier/        # deterministic verifier package: traces.py, audit.py, checks.py
```

## Develop and run

```bash
uv sync --extra dev
uv run pytest
```

On RUSH Lab, ask the plugin's chat, for example "audit trace-05-judge".
