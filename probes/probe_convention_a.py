"""Reproduce the SambaStack harmony bug: orphan ToolMessage in convention-A history.

Sends a deliberately malformed turn-2 history that mimics what older agent
kits produce (tool call expressed as text JSON in `content`, empty
`tool_calls=[]`, followed by a `role=tool` message with an unlinked
`tool_call_id`).

Expected behavior on a tolerant server (e.g. stock vLLM serving gpt-oss-120b
on Sophia): server renders the history loosely and returns 200 with a final
answer.

Observed behavior on SambaStack serving gpt-oss-120b on Metis: server returns
HTTP 500 with "Internal error when running prediction service". The strict
harmony chat template implementation cannot render a tool message that does
not link to a structured tool_calls entry.

Run:
  export SAMBANOVA_API_BASE=...
  export MODEL=gpt-oss-120b
  python endpoint_probes/probe_convention_a.py

Exit codes:
  0 = server tolerated the malformed history (bug not present)
  4 = server rejected the malformed history (bug reproduced — Metis behavior)
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


def main() -> int:
    client = get_client()
    model = get_model("gpt-oss-120b")

    # --- turn 1: produce a real tool_call (any tolerant + correct path) ---
    print("--- turn 1: get a real tool_call ---")
    try:
        r1 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 17 + 25? Use the calculator tool."}],
            tools=[CALCULATOR_TOOL],
            tool_choice="auto",
            max_tokens=512,
        )
    except Exception as e:
        print(f"FAIL on turn 1 (transport/auth): {type(e).__name__}: {str(e)[:240]}")
        return 2
    msg1 = r1.choices[0].message
    if not (msg1.tool_calls and msg1.tool_calls[0].function.name == "calculator"):
        print(f"DEGRADED: turn 1 didn't produce a calculator tool_call. msg={msg1}")
        return 3
    print(f"turn 1 ok: real tool_call id={msg1.tool_calls[0].id}")

    # --- turn 2: send a DELIBERATELY MALFORMED history ---
    # This is what the SambaNova function_calling kit does internally:
    #   - assistant.content contains the tool call serialized as text JSON
    #   - assistant.tool_calls is empty (so no id for the tool message to link to)
    #   - tool message claims tool_call_id="0", a kit-side counter that
    #     references nothing in the assistant turn
    malformed_history = [
        {"role": "user", "content": "What is 17 + 25? Use the calculator tool."},
        {
            "role": "assistant",
            "content": '[{"tool":"calculator","tool_input":{"a":17,"b":25}}]',
            "tool_calls": [],   # <-- the key malformation
        },
        {
            "role": "tool",
            "tool_call_id": "0",            # <-- orphan; nothing to link to
            "content": "Tool 'calculator' response: 42",
        },
    ]

    print("\n--- turn 2: send the malformed convention-A history ---")
    try:
        r2 = client.chat.completions.create(
            model=model,
            messages=malformed_history,
            tools=[CALCULATOR_TOOL],
            max_tokens=512,
        )
    except Exception as e:
        print(f"\nBUG REPRODUCED: server rejected the malformed history.")
        print(f"  {type(e).__name__}: {str(e)[:300]}")
        print("\nThis is the SambaStack harmony bug. A tolerant server (stock vLLM)")
        print("accepts the same history and continues; SambaStack's strict harmony")
        print("renderer cannot template a role=tool message that does not link to a")
        print("structured tool_calls entry on the prior assistant turn.")
        return 4

    msg2 = r2.choices[0].message
    print("\nBUG NOT PRESENT: server tolerated the malformed history.")
    print(f"  turn 2 content: {msg2.content!r}")
    print("  This is the expected vLLM behavior (lenient harmony rendering).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
