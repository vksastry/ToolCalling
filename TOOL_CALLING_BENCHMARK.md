# Tool Calling Benchmark

This document provides minimal, copy/paste-ready examples that cover multiple tool-calling
formats and API shapes: Harmony, ChatML, Responses API, Chat Completions API, Structured
Outputs, and Custom Tools with CFG.

## Coverage Matrix

| Area | Covered | Notes |
| --- | --- | --- |
| Harmony | Yes | Multi-channel + tool call + tool output |
| ChatML | Yes | Equivalent function calling flow |
| Responses API | Yes | Function tools + tool output + reasoning items |
| Chat Completions API | Yes | tools + tool_choice |
| Structured Outputs | Yes | json_schema strict mode |
| Custom Tools + CFG | Yes | Lark and Regex examples |
| Parallel tool calls | Yes | Responses and Chat examples |
| Forced tool choice | Yes | Responses and Chat examples |

## Conventions
- Example tool: `get_weather`
- Temperature omitted for brevity
- Python examples use the OpenAI SDK
- Use `$OPENAI_API_KEY` in your environment

## Harmony Example (Tool Call + Tool Output)

```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|>
<|start|>developer<|message|># Instructions

Be concise.

# Tools

## functions
namespace functions {
// Get weather for a city
type get_weather = (_: {
  city: string,
  unit?: "celsius" | "fahrenheit", // default: celsius
}) => any;
} // namespace functions<|end|>
<|start|>user<|message|>Weather in Tokyo?<|end|>
<|start|>assistant<|channel|>analysis<|message|>Need get_weather for Tokyo.<|end|>
<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Tokyo"}<|call|>
<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>{"temp":22,"unit":"C"}<|end|>
<|start|>assistant<|channel|>final<|message|>Tokyo is 22C right now.<|return|>
```

## ChatML Example (Function Calling)

```
<|im_start|>system
# Tools
<tools>
{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
</tools>
For each function call, return a JSON object within <tool_call></tool_call> tags.
<|im_end|>
<|im_start|>user
Weather in Tokyo?
<|im_end|>
<|im_start|>assistant
<tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>{"temp":22,"unit":"C"}</tool_response>
<|im_end|>
<|im_start|>assistant
Tokyo is 22C right now.
<|im_end|>
```

## Responses API (Function Tool, Tool Output, Final Response)

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [
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

input_items = [{"role": "user", "content": "Weather in Tokyo?"}]

resp = client.responses.create(
  model="gpt-5.2",
  tools=tools,
  input=input_items,
)

input_items += resp.output

for item in resp.output:
  if item.type == "function_call" and item.name == "get_weather":
    args = json.loads(item.arguments)
    result = {"temp": 22, "unit": "C", "city": args["city"]}
    input_items.append({
      "type": "function_call_output",
      "call_id": item.call_id,
      "output": json.dumps(result),
    })

final = client.responses.create(
  model="gpt-5.2",
  tools=tools,
  input=input_items,
)

print(final.output_text)
```

## Chat Completions API (Tools + Tool Choice)

```python
from openai import OpenAI

client = OpenAI()

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

print(completion.choices[0].message)
```

## Structured Outputs (json_schema, Strict)

```python
from openai import OpenAI

client = OpenAI()

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
```

## Custom Tool (Freeform Input)

```json
{
  "model": "gpt-5",
  "input": "Use code_exec to print hello",
  "tools": [
    {
      "type": "custom",
      "name": "code_exec",
      "description": "Executes arbitrary Python code"
    }
  ]
}
```

## Custom Tool with Lark CFG

```python
grammar = """
start: expr
expr: term (" + " term)*
term: INT
%import common.INT
"""

request = {
  "model": "gpt-5",
  "input": "Use math_exp to add four plus four.",
  "tools": [
    {
      "type": "custom",
      "name": "math_exp",
      "description": "Creates valid math expressions",
      "format": {"type": "grammar", "syntax": "lark", "definition": grammar}
    }
  ]
}
```

## Custom Tool with Regex CFG

```python
regex = r"^(January|February|March)\\s+\\d{1,2},\\s+\\d{4}$"

request = {
  "model": "gpt-5",
  "input": "Save the date March 3, 2025.",
  "tools": [
    {
      "type": "custom",
      "name": "save_date",
      "description": "Saves a date",
      "format": {"type": "grammar", "syntax": "regex", "definition": regex}
    }
  ]
}
```

## Parallel Tool Calls (Responses API)

```python
from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
  model="gpt-5.2",
  input="Get weather in Tokyo and Paris",
  parallel_tool_calls=True,
  tools=[
    {
      "type": "function",
      "name": "get_weather",
      "description": "Get weather",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
      },
    }
  ],
)

print(resp.output)
```

## Forced Tool Choice (Chat Completions)

```python
from openai import OpenAI

client = OpenAI()

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
```

## Validation Checklist (Smoke Test)
- Tools appear in the right format for the chosen API.
- Tool calls include a `call_id` and valid JSON arguments (function tools).
- Tool outputs are returned with matching `call_id`.
- For reasoning models, reasoning items are carried across tool-call turns.
- Structured outputs adhere to schema (strict mode on).
- Parallel calls only when explicitly allowed.
- Tool forcing honored when configured.
