"""Probe whether the model emits multiple tool_calls in a single assistant turn."""

from __future__ import annotations

NOTES = """
INPUT (request body):

  {
    "model": "<MODEL>",
    "messages": [{"role": "user", "content":
       "Call the calculator tool ONCE for 17 + 25 and ONCE for 100 / 4, in the same response."}],
    "tools": [<calculator>],
    "tool_choice": "auto",
    "max_tokens": 1024
  }

------------

EXPECTED (PASS — model parallelizes):

  HTTP 200
  {
    "choices": [{
      "message": {
        "tool_calls": [
          {"id":"call_A","function":{"name":"calculator","arguments":"{\\"expression\\":\\"17 + 25\\"}"}},
          {"id":"call_B","function":{"name":"calculator","arguments":"{\\"expression\\":\\"100 / 4\\"}"}}
        ]
      }
    }]
  }

------------

RECEIVED (DEGRADED — model picks one call at a time):

  HTTP 200
  {
    "choices": [{
      "message": {
        "tool_calls": [
          {"id":"call_X","function":{"name":"calculator","arguments":"{\\"expression\\":\\"17 + 25\\"}"}}
        ]
      }
    }]
  }
"""

import json

from _endpoint import get_client, get_model


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Computes the result of a basic arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression like '17 + 25' or '100 / 4'.",
                },
            },
            "required": ["expression"],
        },
    },
}

GET_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Returns the current date and time.",
        "parameters": {"type": "object", "properties": {}},
    },
}


TEST_CASES = [
    {
        "name": "same tool twice",
        "prompt": (
            "I need two calculations done in parallel. Call the calculator tool "
            "ONCE for 17 + 25 and ONCE for 100 / 4, in the same response."
        ),
        "tools": [CALCULATOR_TOOL],
        "expected_n": 2,
    },
    {
        "name": "two different tools",
        "prompt": (
            "Do both of these in parallel in your next response: "
            "(a) call get_time to get the current date, "
            "(b) call calculator with the expression '7 * 6'."
        ),
        "tools": [CALCULATOR_TOOL, GET_TIME_TOOL],
        "expected_n": 2,
    },
]


def main() -> int:
    client = get_client()
    model = get_model("gpt-oss-120b")

    any_parallel = False
    summaries: list[str] = []

    for case in TEST_CASES:
        print(f"--- case: {case['name']} ---")
        print(f"prompt: {case['prompt']}")
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": case["prompt"]}],
                tools=case["tools"],
                tool_choice="auto",
                max_tokens=1024,
            )
        except Exception as e:
            line = f"FAIL: {type(e).__name__}: {str(e)[:240]}"
            print(line)
            summaries.append(f"  [{case['name']}] {line}")
            continue

        msg = r.choices[0].message
        tcs = msg.tool_calls or []
        n = len(tcs)
        print(f"tool_calls returned: {n}")
        for i, tc in enumerate(tcs):
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = tc.function.arguments
            print(f"  [{i}] id={tc.id}  name={tc.function.name}  args={args}")
        if msg.content:
            print(f"content: {msg.content[:200]!r}")

        if n >= 2:
            verdict = "PARALLEL"
            any_parallel = True
        elif n == 1:
            verdict = "SEQUENTIAL (would need another turn for the second call)"
        else:
            verdict = "NO TOOL USE (model answered in text or refused)"
        summaries.append(f"  [{case['name']}] {verdict} — {n} tool_calls")
        print(f"verdict: {verdict}\n")

    print("--- summary ---")
    for s in summaries:
        print(s)
    if any_parallel:
        print("OK: model emits parallel tool calls when asked.")
        return 0
    print("\nDEGRADED: model never emits parallel tool calls in these cases.")
    print("  expected: tool_calls array with length >= 2 (the two requested calls in one response)")
    print("  got     : tool_calls array with length 1 (model chose sequential planning)")
    print("  -> Agent loops still work; they just take more round trips.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
