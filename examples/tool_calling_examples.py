from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from openai import OpenAI


def require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")


def get_weather_tool() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a city",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        }
    ]


def run_responses_function_call(client: OpenAI) -> None:
    tools = get_weather_tool()
    input_items: List[Dict[str, Any]] = [{"role": "user", "content": "Weather in Tokyo?"}]

    first = client.responses.create(
        model="gpt-5.2",
        tools=tools,
        input=input_items,
    )

    input_items += first.output

    for item in first.output:
        if item.type == "function_call" and item.name == "get_weather":
            args = json.loads(item.arguments)
            result = {"temp": 22, "unit": "C", "city": args["city"]}
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result),
                }
            )

    final = client.responses.create(
        model="gpt-5.2",
        tools=tools,
        input=input_items,
    )

    print(final.output_text)


def run_responses_parallel_tools(client: OpenAI) -> None:
    resp = client.responses.create(
        model="gpt-5.2",
        input="Get weather in Tokyo and Paris",
        parallel_tool_calls=True,
        tools=get_weather_tool(),
    )

    print(resp.output)


def run_responses_structured_output(client: OpenAI) -> None:
    response = client.responses.create(
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "system", "content": "Extract the event."},
            {"role": "user", "content": "Alice meets Bob on Friday."},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "event",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "date": {"type": "string"},
                        "participants": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "date", "participants"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        },
    )

    print(response.output_text)


def run_chat_tool_auto(client: OpenAI) -> None:
    completion = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="auto",
    )

    msg = completion.choices[0].message
    if not msg.tool_calls:
        print(msg)
        return

    tool_call = msg.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result = {"temp": 22, "unit": "C", "city": args["city"]}

    followup = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "user", "content": "Weather in Tokyo?"},
            msg,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )

    print(followup.choices[0].message)


def run_chat_forced_tool(client: OpenAI) -> None:
    completion = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": "Weather in Tokyo"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )

    print(completion.choices[0].message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tool-calling examples using the OpenAI SDK")
    parser.add_argument(
        "--example",
        choices=[
            "responses_function_call",
            "responses_parallel_tools",
            "responses_structured_output",
            "chat_tool_auto",
            "chat_forced_tool",
        ],
        required=True,
        help="Which example to run",
    )
    return parser.parse_args()


def main() -> None:
    require_api_key()
    client = OpenAI()
    args = parse_args()

    if args.example == "responses_function_call":
        run_responses_function_call(client)
    elif args.example == "responses_parallel_tools":
        run_responses_parallel_tools(client)
    elif args.example == "responses_structured_output":
        run_responses_structured_output(client)
    elif args.example == "chat_tool_auto":
        run_chat_tool_auto(client)
    elif args.example == "chat_forced_tool":
        run_chat_forced_tool(client)


if __name__ == "__main__":
    main()
