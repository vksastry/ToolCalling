"""Probe the Responses API (/v1/responses).

The Responses API is OpenAI's newer agent-oriented API. Most non-OpenAI
endpoints (stock vLLM, SambaStack, Together, etc.) implement only Chat
Completions and not Responses. This probe attempts a Responses call and
reports whether the endpoint supports it.

Exit codes:
  0 = OK — endpoint supports Responses API
  2 = transport/auth failure
  3 = endpoint returned 404/405 (no Responses API support — expected for vLLM/SambaStack)
  4 = endpoint returned an unexpected error shape
"""

from __future__ import annotations

from _endpoint import get_client, get_model


def main() -> int:
    client = get_client()
    model = get_model("gpt-oss-120b")

    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": "Reply with the word: pong"}],
            max_output_tokens=64,
        )
    except Exception as e:
        err = type(e).__name__
        msg = str(e)[:240]
        # NotFoundError or Method Not Allowed → endpoint just doesn't implement /v1/responses.
        if "NotFoundError" in err or "404" in msg or "method not allowed" in msg.lower():
            print(f"NO RESPONSES API: {err}: {msg}")
            print("\nExpected on stock vLLM and SambaStack — they implement Chat Completions only.")
            return 3
        print(f"FAIL: {err}: {msg}")
        return 4

    text = getattr(resp, "output_text", None) or str(resp.output)
    print(f"output: {text!r}")
    print("\nOK: endpoint supports the Responses API")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
