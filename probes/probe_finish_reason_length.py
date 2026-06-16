"""Probe `finish_reason: "length"` — server reports truncation correctly.

Set a very small max_tokens and ask a question that requires more. The server
must return `finish_reason: "length"` (not "stop") on the truncated response.

Two failure modes worth distinguishing:
  - server reports wrong finish_reason (correctness bug)
  - server returns empty content with finish_reason="length" — common with
    reasoning models like gpt-oss and MiniMax that consume the budget on
    hidden reasoning before any visible token (informational; not a fail)

Exit codes:
  0 = OK — finish_reason == "length"
  2 = transport/auth failure
  3 = DEGRADED — response fit within max_tokens (budget too generous; bump and retry)
  4 = FAIL — finish_reason was something else (wrong report)
"""

from __future__ import annotations

from _endpoint import get_client, get_model


def main() -> int:
    client = get_client()
    model = get_model("gpt-oss-120b")

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": "Write a detailed 500-word essay about the history of the printing press.",
            }],
            max_tokens=20,
        )
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {str(e)[:240]}")
        return 2

    choice = r.choices[0]
    msg = choice.message
    fr = choice.finish_reason
    content_len = len(msg.content or "")
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    reasoning_len = len(reasoning or "")

    print(f"finish_reason : {fr}")
    print(f"content       : {(msg.content or '')[:80]!r}{' …' if content_len > 80 else ''}")
    print(f"content_len   : {content_len} chars")
    if reasoning is not None:
        print(f"reasoning_len : {reasoning_len} chars (reasoning channel emitted)")

    if fr == "length":
        if not msg.content and reasoning_len > 0:
            print("\nOK: finish_reason=length. Reasoning model consumed budget before final text.")
        else:
            print("\nOK: finish_reason=length as expected.")
        return 0

    if fr == "stop" and content_len < 100:
        print(f"\nDEGRADED: finished with 'stop' but content is short ({content_len} chars). "
              "Likely the model gave a brief answer that fit in 20 tokens.")
        return 3

    print(f"\nFAIL: expected finish_reason='length', got {fr!r}.")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
