"""Probe a two-turn agent loop using a well-formed (convention-B) history.

This is the diagnostic probe. After the model emits a tool_call in turn 1, we
build turn 2's request with:
  - the original user message
  - an assistant message echoing the tool_calls array verbatim (with id, name, args)
  - a tool message whose tool_call_id matches the id from the response

If turn 2 succeeds, the server's chat template handles correct multi-turn
histories — i.e. an agent loop that uses bind_tools() / native tool_calls
(convention B) will work against this endpoint.

If turn 2 returns 500, the server's chat template is rejecting even a
well-formed history — that's a serious bug at the server level (not a kit
bug). Worth reporting upstream.

Note: this probe does NOT test convention-A misuse (text-JSON tool calls
with empty `tool_calls` plus orphan ToolMessages). That's a kit-side bug
class — see the function_calling/ rewrite for the fix. To test convention-A
behavior specifically, write a separate probe.

Exit codes:
  0 = OK
  2 = transport/auth failure (network/auth, before logic starts)
  3 = turn 1 didn't produce a tool_call (model declined to use the tool)
  4 = turn 2 failed (the server rejected the multi-turn history)
"""

from __future__ import annotations

from _endpoint import get_client, get_model


NOTES = """
INPUT (turn 2 request body — well-formed convention-B history):

  {
    "model": "<MODEL>",
    "messages": [
      {"role": "user", "content": "What is 17 + 25? Use the calculator tool."},
      {"role": "assistant",
       "content": "",
       "tool_calls": [{
         "id": "call_X",
         "type": "function",
         "function": {"name": "calculator", "arguments": "{\\"a\\":17,\\"b\\":25}"}
       }]},
      {"role": "tool", "tool_call_id": "call_X", "content": "42"}
    ],
    "tools": [<calculator>],
    "max_tokens": 512
  }

  Note: tool message references the SAME id ("call_X") as the assistant's
  tool_calls entry — that's what makes the history "well-formed".

------------

EXPECTED (PASS — server returns final answer):

  HTTP 200
  {"choices": [{"message": {
     "role": "assistant",
     "content": "17 + 25 = 42."
  }, "finish_reason": "stop"}]}

------------

RECEIVED (FAIL — server rejects turn 2):

  HTTP 400 or 500
  {"error": {"message": "<chat-template renderer rejected the history>"}}
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

    messages: list[dict] = [
        {"role": "user", "content": "What is 17 + 25? Use the calculator tool."}
    ]

    print("--- turn 1: ask model to emit a tool call ---")
    try:
        r1 = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[CALCULATOR_TOOL],
            tool_choice="auto",
            max_tokens=512,
        )
    except Exception as e:
        print(f"FAIL on turn 1: {type(e).__name__}: {str(e)[:240]}")
        return 2

    msg1 = r1.choices[0].message
    tcs = msg1.tool_calls or []
    if not tcs:
        print(f"DEGRADED: turn 1 produced no tool_calls. content={msg1.content!r}")
        return 3
    tc = tcs[0]
    print(f"turn 1 tool_call: id={tc.id} name={tc.function.name} args={tc.function.arguments}")

    # Build turn 2's history. Note the strict shape: an assistant message
    # carrying a *structured* tool_calls field (not text JSON), followed by a
    # tool message linked by tool_call_id. This is the format harmony's chat
    # template expects.
    messages.append(
        {
            "role": "assistant",
            "content": msg1.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": "42",  # 17 + 25 — the answer we pretend to compute
        }
    )

    print("\n--- turn 2: send tool result back, expect final answer ---")
    try:
        r2 = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[CALCULATOR_TOOL],
            max_tokens=512,
        )
    except Exception as e:
        print(f"FAIL on turn 2: {type(e).__name__}: {str(e)[:240]}")
        print("\nDiagnostic: server's chat template rejected a correctly-formed history.")
        print("This is a server-side bug. The exact same conversation shape is what")
        print("LangChain's bind_tools() + ToolMessage produce — so any kit that uses")
        print("the modern OpenAI tool-calling convention would also fail here.")
        return 4

    msg2 = r2.choices[0].message
    print(f"turn 2 content    : {msg2.content!r}")
    print(f"turn 2 tool_calls : {len(msg2.tool_calls or [])} additional calls requested")

    print("\nOK: server handled multi-turn (convention-B) history correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
