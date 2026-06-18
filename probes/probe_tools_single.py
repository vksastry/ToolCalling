"""Probe a single native tool call (no follow-up).

Verifies that the endpoint:
  - Accepts a `tools=[...]` request.
  - Returns a structured `tool_calls` entry with id, name, and JSON-parseable
    arguments (i.e. the server's chat template + tool-call parser are working
    correctly for the request side).

This is the most basic agent-readiness check. If this fails, no kit that
needs function calling will work against this endpoint regardless of how
the conversation is structured.

Exit codes:
  0 = OK
  2 = transport/auth failure
  3 = no tool_calls returned (model decided to answer in text)
  4 = tool_calls returned but malformed (bad JSON, wrong tool, etc.)
"""

from __future__ import annotations

import json

from _endpoint import get_client, get_model


NOTES = """
INPUT (request body):

  {
    "model": "<MODEL>",
    "messages": [{"role": "user", "content": "What is 17 + 25? Use the calculator tool."}],
    "tools": [{"type":"function","function":{"name":"calculator",
              "parameters":{"type":"object",
                            "properties":{"a":{"type":"number"},"b":{"type":"number"}},
                            "required":["a","b"]}}}],
    "tool_choice": "auto",
    "max_tokens": 512
  }

------------

EXPECTED (PASS — server returns a structured tool_call with parseable args):

  HTTP 200
  {"choices": [{"message": {
     "role": "assistant",
     "content": null,
     "tool_calls": [{
       "id": "call_X",
       "type": "function",
       "function": {"name": "calculator", "arguments": "{\\"a\\":17,\\"b\\":25}"}
     }]}}]}

------------

RECEIVED (DEGRADED — model answered in text instead of calling the tool):

  HTTP 200
  {"choices": [{"message": {
     "role": "assistant",
     "content": "17 + 25 equals 42.",
     "tool_calls": []
  }}]}

RECEIVED (FAIL — wrong tool name or unparseable arguments):

  HTTP 200
  {"choices": [{"message": {
     "tool_calls": [{"function": {"name": "other_tool", "arguments": "<not JSON>"}}]
  }}]}
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

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 17 + 25? Use the calculator tool."}],
            tools=[CALCULATOR_TOOL],
            tool_choice="auto",
            max_tokens=512,
        )
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {str(e)[:240]}")
        return 2

    msg = r.choices[0].message
    tcs = msg.tool_calls or []
    if not tcs:
        print("\nDEGRADED: no tool_calls returned.")
        print("  expected: tool_calls with at least 1 entry calling 'calculator'")
        print(f"  got     : content={msg.content!r}  tool_calls=[]")
        return 3

    tc = tcs[0]
    print(f"tool_call    : id={tc.id} name={tc.function.name}")
    print(f"raw arguments: {tc.function.arguments!r}")

    if tc.function.name != "calculator":
        print("\nFAIL: wrong tool called.")
        print("  expected: tool_call name='calculator'")
        print(f"  got     : tool_call name={tc.function.name!r}")
        return 4

    try:
        args = json.loads(tc.function.arguments)
    except json.JSONDecodeError as e:
        print("\nFAIL: arguments are not parseable JSON.")
        print("  expected: parseable JSON object with keys 'a' and 'b'")
        print(f"  got     : {tc.function.arguments!r}  (parse error: {e})")
        return 4

    print(f"parsed args  : {args}")
    print("\nOK: native tool_call returned with parseable JSON args")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
