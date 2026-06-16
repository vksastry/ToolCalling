"""Probe `tool_choice: "required"` — model must call SOME tool.

Different from `tool_choice: "auto"` (model decides freely) and from
`tool_choice: {"type":"function","function":{"name":"X"}}` (forced specific
function). "required" forces a tool call but lets the model pick which one.

We give the model a chatty user query that it would normally answer with text,
plus a calculator tool. With "required", the model must call the calculator
(or another tool if we provided more); without, it would answer directly.

Exit codes:
  0 = OK — response has tool_calls
  2 = transport/auth failure
  3 = DEGRADED — server accepted the request, model ignored "required" and
                 returned text content instead of a tool_call
  4 = FAIL — server rejected the request (tool_choice="required" not supported)
"""

from __future__ import annotations

from _endpoint import get_client, get_model


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluates an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


def main() -> int:
    client = get_client()
    model = get_model("gpt-oss-120b")

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Just say hello, no tools needed."}],
            tools=[CALCULATOR_TOOL],
            tool_choice="required",
            max_tokens=512,
        )
    except Exception as e:
        msg = str(e)[:240]
        print(f"FAIL: server rejected tool_choice='required'")
        print(f"  {type(e).__name__}: {msg}")
        return 4

    msg = r.choices[0].message
    tcs = msg.tool_calls or []
    if tcs:
        tc = tcs[0]
        print(f"tool_call: name={tc.function.name} args={tc.function.arguments}")
        print("\nOK: server forced a tool_call as required")
        return 0

    print(f"content: {msg.content!r}")
    print("\nDEGRADED: server accepted tool_choice='required' but model returned text")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
