#!/usr/bin/env python3
"""
concurrency_test.py - Multi-stream concurrency evaluator for MeshLLM / Skippy.
Zero dependencies (uses standard Python 3.x urllib and concurrent.futures).
Automatically saves results to concurrency_results.json and concurrency_results.md.

Usage:
    python3 concurrency_test.py
    TARGET_URL=http://<ip>:9337 python3 concurrency_test.py
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

BASE_URL = os.environ.get("TARGET_URL", "http://127.0.0.1:9337")
MODEL = os.environ.get("MODEL", "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL")
OUT_JSON = os.environ.get("OUT_JSON", "concurrency_results.json")
OUT_MD = os.environ.get("OUT_MD", "concurrency_results.md")

PROMPTS = [
    "Write a concise 3-bullet point summary of quantum computing principles.",
    "Write a concise 3-bullet point summary of artificial neural network architectures.",
    "Write a concise 3-bullet point summary of distributed ledger consensus mechanisms.",
    "Write a concise 3-bullet point summary of CRISPR Cas9 genetic engineering techniques."
]


def send_request(req_id, prompt, max_tokens=50):
    t0 = time.perf_counter()
    print(f"[Req {req_id}] Dispatched: \"{prompt[:45]}...\"")

    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
            wall_clock = time.perf_counter() - t0
            json_data = json.loads(body)
            timings = json_data.get("timings", {})
            usage = json_data.get("usage", {})

            completion_tokens = usage.get("completion_tokens", 0)
            decode_speed = timings.get("predicted_per_second", 0.0)
            queue_wait = timings.get("queue_wait_ms", 0.0)

            print(f"[Req {req_id}] Finished in {wall_clock:.2f}s | Generated: {completion_tokens} tok | Queue Wait: {queue_wait:.1f}ms | Decode: {decode_speed:.2f} tok/s")
            return {
                "id": req_id,
                "wall_clock": wall_clock,
                "completion_tokens": completion_tokens,
                "decode_speed": decode_speed,
                "queue_wait_ms": queue_wait
            }
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        print(f"[Req {req_id}] HTTP Error {e.code}: {err}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[Req {req_id}] Failed: {e}", file=sys.stderr)
        return None


def run_concurrency(n, max_tokens=50):
    print(f"\n================ RUNNING {n} CONCURRENT STREAMS ================")
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [
            executor.submit(send_request, i + 1, PROMPTS[i % len(PROMPTS)], max_tokens)
            for i in range(n)
        ]
        results = [f.result() for f in futures]

    total_wall_sec = time.perf_counter() - t_start
    valid = [r for r in results if r is not None]

    if not valid:
        print(f"\n[ERROR] All {n} streams failed.")
        return {
            "n": n,
            "success_count": 0,
            "total_tokens": 0,
            "total_wall_sec": total_wall_sec,
            "avg_queue_ms": 0.0,
            "avg_decode_speed": 0.0,
            "aggregate_tok_s": 0.0
        }

    total_tokens = sum(r["completion_tokens"] for r in valid)
    avg_queue_ms = sum(r["queue_wait_ms"] for r in valid) / len(valid)
    avg_decode_speed = sum(r["decode_speed"] for r in valid) / len(valid)
    aggregate_tok_per_sec = total_tokens / total_wall_sec if total_wall_sec > 0 else 0.0

    print(f"\n---------------- METRICS FOR {n} CONCURRENT STREAMS ----------------")
    print(f"Streams Executed:          {len(valid)} / {n}")
    print(f"Total Generated Tokens:    {total_tokens} tokens")
    print(f"Total Elapsed Wall Time:   {total_wall_sec:.2f} seconds")
    print(f"Average Queue Wait Time:   {avg_queue_ms:.1f} ms")
    print(f"Average Stream Speed:      {avg_decode_speed:.2f} tok/s per stream")
    print(f"AGGREGATE CLUSTER TOK/SEC: {aggregate_tok_per_sec:.2f} tok/s")
    print("-------------------------------------------------------------------\n")

    return {
        "n": n,
        "success_count": len(valid),
        "total_tokens": total_tokens,
        "total_wall_sec": total_wall_sec,
        "avg_queue_ms": avg_queue_ms,
        "avg_decode_speed": avg_decode_speed,
        "aggregate_tok_s": aggregate_tok_per_sec
    }


def save_reports(r2, r4):
    timestamp = datetime.now().isoformat()
    record = {
        "timestamp": timestamp,
        "endpoint": BASE_URL,
        "model": MODEL,
        "two_streams": r2,
        "four_streams": r4
    }

    # 1. Append/Write JSON
    existing = []
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
        except Exception:
            existing = []
    existing.append(record)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    # 2. Append Markdown report
    with open(OUT_MD, "a", encoding="utf-8") as f:
        f.write(f"\n## Concurrency Run: {timestamp}\n\n")
        f.write(f"- **Endpoint**: `{BASE_URL}`\n")
        f.write(f"- **Model**: `{MODEL}`\n\n")
        f.write("| Test | Executed | Tokens | Wall Time | Queue Wait | Avg Speed/Stream | Aggregate Speed |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        if r2:
            f.write(f"| **2 Streams** | {r2['success_count']}/2 | {r2['total_tokens']} | {r2['total_wall_sec']:.2f}s | {r2['avg_queue_ms']:.1f}ms | {r2['avg_decode_speed']:.2f} tok/s | **{r2['aggregate_tok_s']:.2f} tok/s** |\n")
        if r4:
            f.write(f"| **4 Streams** | {r4['success_count']}/4 | {r4['total_tokens']} | {r4['total_wall_sec']:.2f}s | {r4['avg_queue_ms']:.1f}ms | {r4['avg_decode_speed']:.2f} tok/s | **{r4['aggregate_tok_s']:.2f} tok/s** |\n")
        f.write("\n---\n")

    print(f"💾 Results saved to:")
    print(f"   • JSON: {OUT_JSON}")
    print(f"   • Markdown: {OUT_MD}\n")


def main():
    print("Starting Concurrency Evaluation...")
    print(f"Model:        {MODEL}")
    print(f"Target:       {BASE_URL}")
    print(f"Output Files: {OUT_JSON} & {OUT_MD}")

    # Test 1: 2 Concurrent Streams
    r2 = run_concurrency(2, 50)

    # Test 2: 4 Concurrent Streams
    r4 = run_concurrency(4, 50)

    print("\n================ SUMMARY COMPARISON ================")
    print(f"2 Streams Concurrent: {r2['aggregate_tok_s']:.2f} tok/s aggregate ({r2['total_wall_sec']:.2f}s wall time)")
    print(f"4 Streams Concurrent: {r4['aggregate_tok_s']:.2f} tok/s aggregate ({r4['total_wall_sec']:.2f}s wall time)")
    print("====================================================\n")

    save_reports(r2, r4)


if __name__ == "__main__":
    main()
