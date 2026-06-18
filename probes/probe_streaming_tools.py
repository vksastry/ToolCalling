"""Probe streaming chat completions with tool calls.

Streams a response that should produce a tool call, watching how deltas arrive
and reassembling the full tool_call from chunks. Validates two things:
  1. Server emits tool_call deltas with correct indexing.
  2. Reassembled tool_call has parseable JSON arguments.

Reads SAMBANOVA_API_KEY and SAMBANOVA_API_BASE from .env.
Set MODEL env var to test a specific model. Defaults to gpt-oss-120b.

Run: MODEL=Llama-4-Maverick-17B-128E-Instruct python endpoint_probes/probe_streaming_tools.py
"""

from __future__ import annotations

import json

from _endpoint import get_client, get_model


NOTES = """
INPUT (request body):

  {
    "model": "<MODEL>",
    "messages": [{"role": "user", "content": "What is 17 + 25? Use the calculator tool."}],
    "tools": [<calculator>],
    "tool_choice": "auto",
    "stream": true,
    "max_tokens": 512
  }

------------

EXPECTED (PASS — sequence of SSE deltas; arguments reassemble to valid JSON):

  data: {"choices":[{"delta":{"role":"assistant"}}]}
  data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_X",
                       "function":{"name":"calculator","arguments":""}}]}}]}
  data: {"choices":[{"delta":{"tool_calls":[{"index":0,
                       "function":{"arguments":"{\\"a\\":"}}]}}]}
  data: {"choices":[{"delta":{"tool_calls":[{"index":0,
                       "function":{"arguments":"17,\\"b\\":25}"}}]}}]}
  data: [DONE]

  Concatenating all "arguments" deltas at index 0 → '{"a":17,"b":25}' (valid JSON).

------------

RECEIVED (FAIL — concatenated arguments don't parse as JSON):

  data: {"choices":[{"delta":{"tool_calls":[{"index":0,
                       "function":{"arguments":"{\\"a\\":17"}}]}}]}
  data: [DONE]   (stream cut off mid-JSON, no closing brace)
"""


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Adds two numbers and returns the sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
}


def main() -> int:
    client = get_client()
    model = get_model("gpt-oss-120b")

    print("--- streaming chunks (raw deltas) ---")
    # Reassembly buffers, indexed by tool_call index.
    tool_buffers: dict[int, dict[str, str]] = {}
    text_buffer: list[str] = []
    chunk_count = 0

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 17 + 25? Use the calculator tool."}],
            tools=[CALCULATOR_TOOL],
            tool_choice="auto",
            stream=True,
            max_tokens=512,
        )
        for chunk in stream:
            chunk_count += 1
            delta = chunk.choices[0].delta
            # Print the raw delta object compactly so we can see what arrives.
            d = delta.model_dump(exclude_none=True)
            print(f"  chunk {chunk_count:>3}: {d}")

            if delta.content:
                text_buffer.append(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    buf = tool_buffers.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        buf["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            buf["name"] += tc.function.name
                        if tc.function.arguments:
                            buf["arguments"] += tc.function.arguments
    except Exception as e:
        print(f"\nFAIL during stream: {type(e).__name__}: {str(e)[:240]}")
        return 2

    print(f"\nTotal chunks: {chunk_count}")
    if text_buffer:
        print(f"Reassembled content: {''.join(text_buffer)!r}")

    print("\n--- reassembled tool calls ---")
    if not tool_buffers:
        print("DEGRADED: no tool_calls deltas received during stream.")
        return 3

    all_ok = True
    for idx, buf in sorted(tool_buffers.items()):
        try:
            parsed = json.loads(buf["arguments"]) if buf["arguments"] else {}
            print(f"  [{idx}] id={buf['id']!r}  name={buf['name']!r}  args={parsed}  -> JSON OK")
        except json.JSONDecodeError as e:
            print(f"  [{idx}] id={buf['id']!r}  name={buf['name']!r}  args={buf['arguments']!r}")
            print(f"        FAIL: arguments not parseable JSON: {e}")
            all_ok = False

    print("\n--- summary ---")
    if all_ok and tool_buffers:
        print("OK: server streams tool_calls correctly; arguments reassemble to valid JSON.")
        return 0
    print("DEGRADED: streaming tool_calls had problems (see above).")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
