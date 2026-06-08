"""Probe plain chat completions. Sanity baseline.

Sends one user turn, expects content back. Confirms auth + URL + model align
before more complex probes are worth running.

Exit codes:
  0 = OK
  2 = transport/auth failure
  3 = empty/unexpected response
"""

from __future__ import annotations

from _endpoint import get_client, get_model


def main() -> int:
    client = get_client()
    model = get_model("gpt-oss-120b")

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
            max_tokens=512,
        )
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {str(e)[:240]}")
        return 2

    msg = r.choices[0].message
    print(f"content      : {msg.content!r}")
    print(f"finish_reason: {r.choices[0].finish_reason}")
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    if reasoning:
        print(f"reasoning    : {len(reasoning)} chars (model emits separate analysis channel)")

    if msg.content and "pong" in msg.content.lower():
        print("\nOK: model responded with the expected token")
        return 0
    print(f"\nDEGRADED: response content was empty or unexpected. content={msg.content!r}")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
