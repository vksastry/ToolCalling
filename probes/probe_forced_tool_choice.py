"""Probe forced tool_choice — endpoint must call a specific named tool.

Sends two tools but forces the model to call only one of them via
`tool_choice={"type":"function","function":{"name":"calculator"}}`. Validates
that the model returns a tool_call for the forced tool, ignoring the other.

Exit codes:
  0 = OK
  2 = transport/auth failure
  3 = endpoint accepted the request but model called wrong tool / no tool
  4 = endpoint rejected the forced tool_choice (parameter not supported)
"""

from __future__ import annotations

from _endpoint import get_client, get_model


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Adds two numbers.",
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
GET_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Returns the current time.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def main() -> int:
    client = get_client()
    model = get_model("gpt-oss-120b")

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is the time?"}],
            tools=[CALCULATOR_TOOL, GET_TIME_TOOL],
            tool_choice={"type": "function", "function": {"name": "calculator"}},
            max_tokens=512,
        )
    except Exception as e:
        print(f"FAIL: endpoint rejected forced tool_choice: {type(e).__name__}: {str(e)[:240]}")
        return 4

    msg = r.choices[0].message
    tcs = msg.tool_calls or []
    if not tcs:
        print(f"DEGRADED: no tool_calls returned even though one was forced. content={msg.content!r}")
        return 3

    tc = tcs[0]
    print(f"tool_call: name={tc.function.name} args={tc.function.arguments}")
    if tc.function.name != "calculator":
        print(f"\nDEGRADED: expected 'calculator' (forced), got '{tc.function.name}'")
        return 3

    print("\nOK: endpoint honored tool_choice and called the forced tool")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
