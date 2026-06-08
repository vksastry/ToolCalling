"""Test suite runner for OpenAI-compatible endpoints.

Runs every probe against every endpoint, captures results, writes a dated
markdown report under results/, and exits 0 only if all probes returned 0.

CLI:
    python run_suite.py                                    # everything from endpoints.yaml
    python run_suite.py --endpoint sophia-gpt-oss          # one named endpoint
    python run_suite.py --device metis --model gpt-oss-120b  # ad-hoc, no yaml edit
    python run_suite.py --probe probe_parallel_tools.py    # one probe across endpoints
    python run_suite.py --list                             # print config and exit
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROBES_DIR = HERE / "probes"
CONFIG_PATH = HERE / "endpoints.yaml"

EXIT_MEANING = {
    0: "OK",
    1: "MISCONFIG",
    2: "TRANSPORT/AUTH FAIL",
    3: "DEGRADED",
    4: "FAIL",
}

# Short device names → ALCF base URLs. Used by --device.
DEVICES = {
    "metis":        "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1",
    "metis-direct": "https://metis.alcf.anl.gov/v1",  # bypasses Globus gateway; needs a SambaNova-style API key
    "sophia":       "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    "sophia-sam3":  "https://inference-api.alcf.anl.gov/resource_server/sophia/sam3service/v1",
}


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def run_probe(probe: str, endpoint_name: str, cfg: dict) -> dict:
    """Run one probe as a subprocess with the endpoint's env vars set."""
    env = os.environ.copy()
    env["SAMBANOVA_API_BASE"] = cfg["url"]
    env["MODEL"] = cfg["model"]
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(PROBES_DIR / probe)],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as e:
        exit_code = 124
        stdout = (e.stdout or "") + "\n[timed out after 180s]"
        stderr = e.stderr or ""
    elapsed = time.time() - started
    return {
        "endpoint": endpoint_name,
        "probe": probe,
        "model": cfg["model"],
        "url": cfg["url"],
        "exit_code": exit_code,
        "verdict": EXIT_MEANING.get(exit_code, f"unknown({exit_code})"),
        "elapsed_s": round(elapsed, 1),
        "stdout": stdout,
        "stderr": stderr,
    }


def write_report(results: list, path: Path) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Endpoint test suite — {now}",
        "",
        "## Summary",
        "",
        "| Endpoint | Probe | Verdict | Exit | Time |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['endpoint']} | {r['probe']} | {r['verdict']} | {r['exit_code']} | {r['elapsed_s']}s |"
        )
    lines += ["", "## Detail", ""]
    for r in results:
        lines += [
            f"### {r['endpoint']} × {r['probe']}",
            "",
            f"- url: `{r['url']}`",
            f"- model: `{r['model']}`",
            f"- exit_code: {r['exit_code']} ({r['verdict']})",
            f"- elapsed: {r['elapsed_s']}s",
            "",
            "```",
            r["stdout"].rstrip() or "(no stdout)",
        ]
        if r["stderr"].strip():
            lines += ["--- stderr ---", r["stderr"].rstrip()]
        lines += ["```", ""]
    path.write_text("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", help="run only this named endpoint (from endpoints.yaml)")
    p.add_argument("--device", choices=list(DEVICES), help=f"ad-hoc: pick a base URL ({', '.join(DEVICES)})")
    p.add_argument("--model", help="ad-hoc: model name or comma-separated list; used with --device")
    p.add_argument("--probe", help="run only this probe (filename)")
    p.add_argument("--list", action="store_true", help="show config and exit")
    p.add_argument("--output", type=Path, help="override report path")
    args = p.parse_args()

    cfg = load_config()
    endpoints: dict = cfg["endpoints"]
    probes: list = cfg["probes"]

    if args.list:
        print("Endpoints:")
        for name, e in endpoints.items():
            print(f"  {name:24s}  model={e['model']}")
            print(f"  {'':24s}  url={e['url']}")
        print("\nDevices (for --device):")
        for d, url in DEVICES.items():
            print(f"  {d:24s}  url={url}")
        print("\nProbes:")
        for probe in probes:
            print(f"  {probe}")
        return 0

    # --device/--model is mutually exclusive with --endpoint.
    if args.device or args.model:
        if args.endpoint:
            print("ERROR: --endpoint and --device/--model are mutually exclusive.", file=sys.stderr)
            return 1
        if not (args.device and args.model):
            print("ERROR: --device and --model must be given together.", file=sys.stderr)
            return 1
        # --model can be comma-separated to run multiple models on one device.
        endpoints = {}
        for m in [s.strip() for s in args.model.split(",") if s.strip()]:
            name = f"{args.device}-{m}"
            endpoints[name] = {"url": DEVICES[args.device], "model": m, "notes": "ad-hoc from CLI"}
    elif args.endpoint:
        if args.endpoint not in endpoints:
            print(f"Unknown endpoint: {args.endpoint}. See --list.", file=sys.stderr)
            return 1
        endpoints = {args.endpoint: endpoints[args.endpoint]}

    if args.probe:
        if args.probe not in probes:
            print(f"Unknown probe: {args.probe}. See --list.", file=sys.stderr)
            return 1
        probes = [args.probe]

    results: list = []
    for ename, ecfg in endpoints.items():
        for probe in probes:
            print(f"\n>>> {ename} × {probe}")
            r = run_probe(probe, ename, ecfg)
            print(f"    {r['verdict']} (exit {r['exit_code']}) in {r['elapsed_s']}s")
            results.append(r)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    report_path = args.output or (results_dir / f"results-suite-{timestamp}.md")
    write_report(results, report_path)
    print(f"\nReport written: {report_path}")

    all_ok = all(r["exit_code"] == 0 for r in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
