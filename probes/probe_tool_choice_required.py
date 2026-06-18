"""Probe `tool_choice: "required"` in a routing-agent scenario.

Models a customer-support dispatcher: an incoming user message must be routed
to one of three handlers. Free-form text replies are not acceptable — they
mean the dispatcher failed. `tool_choice: "required"` is the guardrail that
should force the model to pick a route.

Exit codes:
  0 = OK — server forced a route (any of the three is acceptable)
  2 = transport/auth failure
  3 = DEGRADED — server accepted the request, model returned text instead
                 of routing (dispatcher would break in production)
  4 = FAIL — server rejected the request (tool_choice="required" not supported)
"""

from __future__ import annotations

from _endpoint import get_client, get_model


NOTES = """
INPUT (request body):

  {
    "model": "<MODEL>",
    "messages": [{"role": "user", "content": "Hi, my account isn't working"}],
    "tools": [
      {"type":"function","function":{"name":"route_to_billing",
         "description":"Route to billing team","parameters":{...}}},
      {"type":"function","function":{"name":"route_to_tech_support",
         "description":"Route to technical support","parameters":{...}}},
      {"type":"function","function":{"name":"route_to_sales",
         "description":"Route to sales team","parameters":{...}}}
    ],
    "tool_choice": "required",
    "max_tokens": 512
  }

  Realistic dispatch scenario: every message must be routed somewhere. Free-form
  text replies break the downstream pipeline.

------------

EXPECTED (PASS — server forces a route):

  HTTP 200
  {"choices": [{"message": {
     "tool_calls": [{"function": {
        "name": "route_to_tech_support",     # or billing — any of the 3 is OK
        "arguments": "{\\"reason\\":\\"account access issue\\"}"
     }}]
  }}]}

------------

RECEIVED (DEGRADED — server didn't enforce, dispatcher fails in production):

  HTTP 200
  {"choices": [{"message": {
     "content": "I'm sorry to hear that. Can you tell me more about what's wrong?",
     "tool_calls": []
  }}]}
"""


ROUTE_BILLING = {
    "type": "function",
    "function": {
        "name": "route_to_billing",
        "description": "Route the user's request to the billing team. Use for payment, invoice, subscription, or refund questions.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Why this routing was chosen."}},
            "required": ["reason"],
        },
    },
}
ROUTE_TECH_SUPPORT = {
    "type": "function",
    "function": {
        "name": "route_to_tech_support",
        "description": "Route the user's request to technical support. Use for login problems, bugs, account access, or product errors.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Why this routing was chosen."}},
            "required": ["reason"],
        },
    },
}
ROUTE_SALES = {
    "type": "function",
    "function": {
        "name": "route_to_sales",
        "description": "Route the user's request to the sales team. Use for product inquiries, upgrades, or pricing questions.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Why this routing was chosen."}},
            "required": ["reason"],
        },
    },
}

VALID_ROUTES = {"route_to_billing", "route_to_tech_support", "route_to_sales"}


def main() -> int:
    client = get_client()
    model = get_model("gpt-oss-120b")

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi, my account isn't working"}],
            tools=[ROUTE_BILLING, ROUTE_TECH_SUPPORT, ROUTE_SALES],
            tool_choice="required",
            max_tokens=512,
        )
    except Exception as e:
        msg = str(e)[:240]
        print(f"FAIL: server rejected tool_choice='required'")
        print(f"  {type(e).__name__}: {msg}")
        return 4

    msg = r.choices[0].message
    tcs = msg.tool_calls or []
    if tcs:
        tc = tcs[0]
        print(f"tool_call: name={tc.function.name} args={tc.function.arguments}")
        if tc.function.name in VALID_ROUTES:
            print(f"\nOK: server forced a route ({tc.function.name}). Dispatcher would succeed.")
            return 0
        print("\nDEGRADED: server forced a tool, but not one of the registered routes.")
        print(f"  expected: name in {sorted(VALID_ROUTES)}")
        print(f"  got     : name={tc.function.name!r}")
        return 3

    print("\nDEGRADED: model returned text instead of routing. Dispatcher would break.")
    print("  expected: tool_call to one of the route_to_* functions")
    print(f"  got     : content={msg.content!r}  tool_calls=[]")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
