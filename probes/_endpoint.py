"""Shared endpoint helper for the probe scripts.

Lets every probe run against any OpenAI-compatible endpoint without editing
.env each time. Resolution order:

    1. os.environ (set inline on the command line)
    2. .env at the repo root

Vars used:
    SAMBANOVA_API_BASE       base URL of the chat endpoint
    SAMBANOVA_API_KEY        bearer token (vLLM ignores it, but the SDK
                             requires a non-empty string; defaults to "EMPTY")
    INSECURE_SKIP_VERIFY=1   skip TLS verification (self-signed certs)
    SSL_CERT_FILE            path to a CA bundle; Python's stdlib also reads this

Example: probe a self-hosted vLLM on the same machine with a self-signed cert.

    SAMBANOVA_API_BASE="https://127.0.0.1:8000/v1" \
    SAMBANOVA_API_KEY="EMPTY" \
    INSECURE_SKIP_VERIFY=1 \
    MODEL="gpt-oss-120b" \
    python endpoint_probes/probe_streaming_tools.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from openai import OpenAI


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def get_client() -> OpenAI:
    """Build an OpenAI client, env > .env > defaults."""
    repo_env = _load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    base_url = os.environ.get("SAMBANOVA_API_BASE") or repo_env.get("SAMBANOVA_API_BASE")
    api_key = (
        os.environ.get("SAMBANOVA_API_KEY")
        or repo_env.get("SAMBANOVA_API_KEY")
        or "EMPTY"
    )
    if not base_url:
        raise SystemExit("FAIL: SAMBANOVA_API_BASE not set (env or .env)")

    http_client: Optional[object] = None
    if os.environ.get("INSECURE_SKIP_VERIFY") == "1":
        import httpx  # local import — only needed for this path

        http_client = httpx.Client(verify=False)

    print(f"endpoint = {base_url}")
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


def get_model(default: str) -> str:
    """Picks the model name; MODEL env var overrides the default."""
    model = os.environ.get("MODEL", default)
    print(f"model    = {model}\n")
    return model
