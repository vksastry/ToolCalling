"""Verbose inspector for forced tool_choice behavior.

Prints the full request body, the full server response, and the model's
reasoning trace (if exposed). Use it to compare side-by-side how SambaStack
vs vLLM handle the same forced-tool_choice request.

Run on each endpoint and diff the outputs:

    python probes/inspect_forced_tool_choice.py            # uses .env + MODEL
    MODEL=... SAMBANOVA_API_BASE=... python probes/inspect_forced_tool_choice.py

Not part of the suite (filename doesn't start with probe_).
"""

from __future__ import annotations

import json

from _endpoint import get_client, get_model


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Adds two numbers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
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


def banner(s: str) -> None:
    print(f"\n=== {s} ===")


def main() -> int:
    client = get_client()
    model = get_model("gpt-oss-120b")

    request_body = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the time?"}],
        "tools": [CALCULATOR_TOOL, GET_TIME_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "calculator"}},
        "max_tokens": 512,
    }

    banner("REQUEST BODY (what we send)")
    print(json.dumps(request_body, indent=2))

    banner("Annotations on key fields")
    print("messages[0].content    : The user's question — naturally fits get_time.")
    print("tools[]                : Two functions registered; model can pick either or none.")
    print('tool_choice            : Forced — server *should* constrain output to "calculator".')

    r = client.chat.completions.create(**request_body)

    banner("FULL RESPONSE (what came back)")
    print(json.dumps(r.model_dump(), indent=2, default=str))

    msg = r.choices[0].message
    tcs = msg.tool_calls or []
    reasoning = (
        getattr(msg, "reasoning", None)
        or getattr(msg, "reasoning_content", None)
        or msg.model_dump().get("reasoning_content")
        or msg.model_dump().get("reasoning")
    )

    banner("INTERPRETATION")
    print(f"content              : {msg.content!r}")
    print(f"tool_calls count     : {len(tcs)}")
    if tcs:
        tc = tcs[0]
        print(f"tool_calls[0].name   : {tc.function.name}")
        print(f"tool_calls[0].args   : {tc.function.arguments}")
    print(f"finish_reason        : {r.choices[0].finish_reason}")
    if reasoning:
        print(f"reasoning length     : {len(reasoning)} chars")
        print("reasoning content    :")
        for line in reasoning.split("\n"):
            print(f"  | {line}")

    banner("VERDICT")
    if tcs and tcs[0].function.name == "calculator":
        print("Server enforced the forced tool_choice. The model emitted 'calculator' as required.")
        print("This means the server used constrained decoding (or equivalent) to make the")
        print("'calculator' function name the only legal output. The model had no escape.")
    elif tcs and tcs[0].function.name != "calculator":
        print(f"Server did NOT enforce the forced tool_choice — model picked '{tcs[0].function.name}' instead.")
        print("This means the server passed `tool_choice` as guidance only, not as a hard constraint.")
        print("If reasoning_content is present, you can see whether the model was even *aware*")
        print("it was supposed to call calculator. If the reasoning says 'should use get_time' with")
        print("no mention of the constraint, the server didn't communicate the constraint at all.")
    else:
        print("Server returned no tool_call. Model defied the directive entirely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
