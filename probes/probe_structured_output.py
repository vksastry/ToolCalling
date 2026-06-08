"""Probe structured output via response_format=json_schema.

Asks the model to extract event info and return it as a strictly-typed JSON
object. Tests whether the endpoint honors `response_format` with a JSON schema
(modern OpenAI Chat Completions feature). Uses Chat Completions API (widely
supported) rather than the Responses API (OpenAI-only).

Exit codes:
  0 = OK
  2 = transport/auth failure
  3 = endpoint accepted the request but returned malformed/unschemed output
  4 = endpoint rejected the request (response_format not supported)
"""

from __future__ import annotations

import json

from _endpoint import get_client, get_model


SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "date": {"type": "string"},
        "participants": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "date", "participants"],
    "additionalProperties": False,
}


def main() -> int:
    client = get_client()
    model = get_model("gpt-oss-120b")

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Extract the event into JSON."},
                {"role": "user", "content": "Alice meets Bob on Friday."},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "event",
                    "schema": SCHEMA,
                    "strict": True,
                },
            },
            max_tokens=512,
        )
    except Exception as e:
        msg = str(e)[:200]
        if "response_format" in msg or "json_schema" in msg or "unsupported" in msg.lower():
            print(f"FAIL: endpoint does not support response_format json_schema")
        print(f"FAIL: {type(e).__name__}: {msg}")
        return 4

    content = r.choices[0].message.content
    print(f"raw content: {content!r}")

    if not content:
        print("\nDEGRADED: response had no content")
        return 3

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"\nDEGRADED: content is not valid JSON: {e}")
        return 3

    missing = [k for k in SCHEMA["required"] if k not in parsed]
    if missing:
        print(f"\nDEGRADED: parsed JSON missing required fields: {missing}")
        return 3

    print(f"\nparsed: {parsed}")
    print("\nOK: endpoint honored response_format and returned valid JSON matching schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
