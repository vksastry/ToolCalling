# Tool Calling Benchmarks

A runnable benchmark suite that characterizes any OpenAI-compatible LLM endpoint on tool-calling behavior. Tests are organized as small Python scripts ("probes") that exercise one specific feature each — single tool call, parallel tool calls, streaming, multi-turn agent loop, structured output, the older convention-A history format, etc. — and report pass / degraded / fail with the actual server response shape.

## What's here

```
ToolCalling/
├── README.md                       this file
├── TOOL_CALLING_BENCHMARK.md       format catalog and reference examples
├── endpoints.yaml                  list of endpoints to test and which probes to run
├── run_suite.py                    orchestrator — runs probes × endpoints, writes report
├── probes/                         one file per probe; each is independently runnable
│   ├── _endpoint.py                shared helper: reads .env, builds the OpenAI client
│   ├── probe_chat.py
│   ├── probe_tools_single.py
│   ├── probe_tools_multi_turn.py   the diagnostic probe for harmony-format servers
│   ├── probe_convention_a.py       reproduces the SambaStack gpt-oss bug
│   ├── probe_streaming_tools.py
│   ├── probe_parallel_tools.py
│   ├── probe_forced_tool_choice.py
│   ├── probe_structured_output.py
│   ├── probe_responses_api.py
│   └── probe_embeddings.py
├── examples/
│   └── tool_calling_examples.py    worked OpenAI reference examples
├── results/                        dated suite reports, committed to git
└── .env-example                    template for SAMBANOVA_API_KEY + SAMBANOVA_API_BASE
```

## Setup

```bash
cp .env-example .env
# edit .env: paste your API key (Globus token for ALCF; static key for SambaCloud)

python3.11 -m venv .venv
source .venv/bin/activate
pip install openai pyyaml
```

## Run

```bash
# Everything (every endpoint × every probe)
python run_suite.py

# Just one endpoint
python run_suite.py --endpoint sophia-gpt-oss

# Just one probe across all endpoints
python run_suite.py --probe probe_convention_a.py

# See what's configured
python run_suite.py --list

# Run a single probe directly (no suite)
SAMBANOVA_API_BASE="..." MODEL="..." python probes/probe_chat.py
```

Each suite run writes a dated markdown report under `results/`.

## Adding a probe

1. Create `probes/probe_<name>.py`. Use `_endpoint.py` for client setup:
   ```python
   from _endpoint import get_client, get_model
   def main() -> int:
       client = get_client()
       model = get_model("default-model")
       ...
   ```
2. Pick an exit code (0 = OK, 2 = transport/auth, 3 = degraded, 4 = fail).
3. Add the filename to `endpoints.yaml` under `probes:`.

## Adding an endpoint

Append a block under `endpoints:` in `endpoints.yaml`:

```yaml
my-endpoint:
  url: https://example.com/v1
  model: some-model-name
  notes: free-text description
```

If it uses a different auth key, the suite currently reads `SAMBANOVA_API_KEY` from `.env`. Per-endpoint auth is a planned addition.

## See also

- [TOOL_CALLING_BENCHMARK.md](./TOOL_CALLING_BENCHMARK.md) — catalog of tool-calling formats with worked examples. Acts as the spec; the probes are runnable implementations of subsets of this catalog.
- [examples/tool_calling_examples.py](./examples/tool_calling_examples.py) — runnable OpenAI reference for the formats covered.
