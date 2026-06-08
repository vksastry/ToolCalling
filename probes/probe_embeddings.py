"""Probe whether the chat endpoint also exposes /v1/embeddings.

Reads SAMBANOVA_API_KEY and SAMBANOVA_API_BASE from .env.
Tries a few likely embedding model names. Records what's served, what isn't.

Run: python endpoint_probes/probe_embeddings.py
"""

from __future__ import annotations

from _endpoint import get_client


# Common embedding model names to probe. Order matters — we want the first hit
# to be the canonical SambaNova one, then HF-style, then OpenAI-compat fallback.
CANDIDATES = [
    "E5-Mistral-7B-Instruct",          # SambaNova-catalog style
    "intfloat/e5-large-v2",             # HF-style (kits' default)
    "intfloat/e5-mistral-7b-instruct",
    "BAAI/bge-large-en-v1.5",
    "text-embedding-3-small",           # OpenAI-compat
    "text-embedding-ada-002",
]


def main() -> int:
    client = get_client()

    findings: list[str] = []
    for model in CANDIDATES:
        try:
            r = client.embeddings.create(model=model, input=["the quick brown fox"])
            vec = r.data[0].embedding
            line = f"OK   {model}  ({len(vec)} dims)"
            print(line)
            findings.append(line)
        except Exception as e:
            err = type(e).__name__
            msg = str(e)[:140].replace("\n", " ")
            line = f"FAIL {model}  -> {err}: {msg}"
            print(line)
            findings.append(line)

    served = [f for f in findings if f.startswith("OK")]
    print("\n--- summary ---")
    if served:
        print(f"Embedding endpoint reachable. Served models: {len(served)}")
        for s in served:
            print(f"  {s}")
        print("Implication: kits that need embeddings can use server-side embedding.")
        return 0
    print("No embedding model reachable.")
    print("Implication: kits will fall back to local HF model (intfloat/e5-large-v2).")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
