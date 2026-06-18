"""Probe forced tool_choice — endpoint must call a specific named tool."""

from __future__ import annotations

from _endpoint import get_client, get_model, verbose_chat_completion


NOTES = """
INPUT (request body):

  {
    "model": "<MODEL>",
    "messages": [{"role": "user", "content": "What is the time?"}],
    "tools": [
      {"type":"function","function":{"name":"get_time_local","parameters":{...}}},
      {"type":"function","function":{"name":"get_time_utc","parameters":{...}}}
    ],
    "tool_choice": {"type":"function","function":{"name":"get_time_utc"}},
    "max_tokens": 512
  }

  Both tools are valid answers to "What is the time?". Force is on get_time_utc.

------------

EXPECTED (PASS — server constrains decoding):

  HTTP 200
  {
    "choices": [{
      "message": {
        "tool_calls": [{
          "id": "call_X",
          "function": {"name": "get_time_utc", "arguments": "{}"}
        }]
      }
    }]
  }

------------

RECEIVED (DEGRADED — server treats tool_choice as a hint, model picks the other):

  HTTP 200
  {
    "choices": [{
      "message": {
        "tool_calls": [{
          "id": "call_X",
          "function": {"name": "get_time_local", "arguments": "{}"}
        }]
      }
    }]
  }
"""


GET_TIME_LOCAL_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time_local",
        "description": "Returns the current time in the user's local timezone.",
        "parameters": {"type": "object", "properties": {}},
    },
}
GET_TIME_UTC_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time_utc",
        "description": "Returns the current time in UTC.",
        "parameters": {"type": "object", "properties": {}},
    },
}

FORCED_NAME = "get_time_utc"


def main() -> int:
    client = get_client()
    model = get_model("gpt-oss-120b")

    try:
        r = verbose_chat_completion(
            client,
            model=model,
            messages=[{"role": "user", "content": "What is the time?"}],
            tools=[GET_TIME_LOCAL_TOOL, GET_TIME_UTC_TOOL],
            tool_choice={"type": "function", "function": {"name": FORCED_NAME}},
            max_tokens=512,
        )
    except Exception as e:
        print(f"FAIL: endpoint rejected forced tool_choice: {type(e).__name__}: {str(e)[:240]}")
        return 4

    msg = r.choices[0].message
    tcs = msg.tool_calls or []
    if not tcs:
        print("\nDEGRADED: no tool_calls returned even though one was forced.")
        print(f"  expected: tool_call with name={FORCED_NAME!r} (the forced function)")
        print(f"  got     : content={msg.content!r}  tool_calls=[]")
        return 3

    tc = tcs[0]
    print(f"tool_call: name={tc.function.name} args={tc.function.arguments}")
    if tc.function.name != FORCED_NAME:
        print("\nDEGRADED: model ignored the forced tool_choice.")
        print(f"  expected: tool_call with name={FORCED_NAME!r}")
        print(f"  got     : tool_call with name={tc.function.name!r}")
        return 3

    print("\nOK: endpoint honored tool_choice and called the forced tool")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
