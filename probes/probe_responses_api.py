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


NOTES = """
INPUT (POST to /v1/responses):

  {
    "model": "<MODEL>",
    "input": [{"role": "user", "content": "Reply with the word: pong"}],
    "max_output_tokens": 64
  }

------------

EXPECTED (PASS — server implements Responses API):

  HTTP 200
  {
    "id": "resp_X",
    "output_text": "pong",
    "output": [{"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "pong"}]}]
  }

------------

RECEIVED (DEGRADED — endpoint doesn't implement /v1/responses):

  HTTP 404 or 405
  <html>Not Found</html>

RECEIVED (DEGRADED — endpoint exists but this model isn't on the allowlist):

  HTTP 400
  {
    "error": {
      "code": "unsupported_model",
      "message": "Unsupported model <X> on Response API",
      "param": "model",
      "type": "invalid_request_error"
    }
  }

  Server-side allowlist gates which models can use /v1/responses. Chat
  Completions still works for these models — only the Responses API path
  is restricted.
"""


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
        # 404 / Method Not Allowed → endpoint doesn't implement /v1/responses at all.
        if "NotFoundError" in err or "404" in msg or "method not allowed" in msg.lower():
            print(f"NO RESPONSES API: {err}: {msg}")
            print("\nEndpoint doesn't implement /v1/responses (Chat Completions only).")
            return 3
        # Structured 400 with `unsupported_model` → endpoint exists, but this
        # specific model isn't routable through it (per-model allowlist).
        if "unsupported_model" in msg or "Unsupported model" in msg or (
            "BadRequestError" in err and "Response API" in msg
        ):
            print(f"MODEL NOT ON RESPONSES ALLOWLIST: {err}: {msg}")
            print("\nEndpoint serves /v1/responses but this model isn't allowed on it.")
            print("Try another model. Chat Completions still works for this model.")
            return 3
        print(f"FAIL: {err}: {msg}")
        return 4

    text = getattr(resp, "output_text", None) or str(resp.output)
    print(f"output: {text!r}")
    print("\nOK: endpoint supports the Responses API")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
