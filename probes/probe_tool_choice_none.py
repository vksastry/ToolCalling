"""Probe `tool_choice: "none"` — model must NOT call any tool.

The opposite of "required". We send a user query that would normally trigger a
tool call (calculator question) and a calculator tool, but pass
`tool_choice: "none"`. The server should suppress all tool calls and force a
text reply.

Exit codes:
  0 = OK — response has no tool_calls, has content
  2 = transport/auth failure
  3 = DEGRADED — server accepted the request, model ignored "none" and called a tool anyway
  4 = FAIL — server rejected the request (tool_choice="none" not supported)
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
            messages=[{"role": "user", "content": "What is 17 + 25? Just answer in text."}],
            tools=[CALCULATOR_TOOL],
            tool_choice="none",
            max_tokens=512,
        )
    except Exception as e:
        msg = str(e)[:240]
        print(f"FAIL: server rejected tool_choice='none'")
        print(f"  {type(e).__name__}: {msg}")
        return 4

    msg = r.choices[0].message
    tcs = msg.tool_calls or []
    if tcs:
        tc = tcs[0]
        print(f"unexpected tool_call: name={tc.function.name} args={tc.function.arguments}")
        print("\nDEGRADED: server accepted tool_choice='none' but model called a tool anyway")
        return 3

    if not msg.content:
        print("\nDEGRADED: response had neither tool_calls nor text content")
        return 3

    print(f"content: {msg.content!r}")
    print("\nOK: server suppressed tool calls and returned text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
