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


NOTES = """
INPUT (request body):

  {
    "model": "<MODEL>",
    "messages": [{"role": "user", "content":
       "Write a detailed 500-word essay about the history of the printing press."}],
    "max_tokens": 20
  }

  Note: max_tokens is intentionally tiny — the answer cannot fit.

------------

EXPECTED (PASS — server reports truncation correctly):

  HTTP 200
  {"choices": [{
     "message": {"content": "The printing press, invented by Johannes"},
     "finish_reason": "length"
  }]}

  Or for reasoning models (gpt-oss, MiniMax) the budget is consumed by hidden
  reasoning and content can be empty — finish_reason="length" is still PASS:

  {"choices": [{
     "message": {"content": null, "reasoning_content": "<partial trace>"},
     "finish_reason": "length"
  }]}

------------

RECEIVED (FAIL — server reports wrong finish_reason):

  HTTP 200
  {"choices": [{
     "message": {"content": "..."},
     "finish_reason": "stop"          # wrong — response was clearly truncated
  }]}

RECEIVED (DEGRADED — response fit in 20 tokens; not a real truncation test):

  {"choices": [{
     "message": {"content": "Done."},
     "finish_reason": "stop"
  }]}
"""


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
        print("\nDEGRADED: response fit within the 20-token budget.")
        print("  expected: finish_reason='length' (truncated at max_tokens=20)")
        print(f"  got     : finish_reason='stop' with content_len={content_len} chars")
        print("  -> model gave a terse answer that didn't trigger truncation. Bump max_tokens lower and retry to force length.")
        return 3

    print("\nFAIL: server reported the wrong finish_reason.")
    print("  expected: 'length' (response was truncated at max_tokens=20)")
    print(f"  got     : {fr!r}")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
