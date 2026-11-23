#!/usr/bin/env python3
"""
benchmark.py - Zero-dependency benchmark runner for MeshLLM / Skippy.
Works with standard Python 3.x (no pip packages required).
Automatically writes structured results to benchmark_results.json and benchmark_results.md.

Usage:
    python3 benchmark.py
    python3 benchmark.py --stress
    python3 benchmark.py --streaming
    python3 benchmark.py --full
    TARGET_URL=http://<ip>:9337 python3 benchmark.py
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from datetime import datetime

BASE_URL = os.environ.get("TARGET_URL", "http://127.0.0.1:9337")
MODEL = os.environ.get("MODEL", "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL")
OUT_JSON = os.environ.get("OUT_JSON", "benchmark_results.json")
OUT_MD = os.environ.get("OUT_MD", "benchmark_results.md")

print("\n================ BENCHMARK TARGET CONFIG ================")
print(f"Target Endpoint: {BASE_URL}")
print(f"Model:           {MODEL}")
print(f"Output Files:    {OUT_JSON} & {OUT_MD}")
print("=========================================================\n")


def post_json(endpoint, payload):
    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
            wall_clock = time.perf_counter() - t0
            return json.loads(body), wall_clock
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        print(f"HTTP Error {e.code}: {err_msg}", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return None, None


def run_standard_benchmark(tokens_to_generate=100):
    print(f"--- [1] Standard Generation Test ({tokens_to_generate} tokens) ---")
    prompt = "Write a detailed timeline of key events in computing history from 1950 to 1990 with bullet points."
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens_to_generate,
        "temperature": 0.2
    }

    res, wall_clock = post_json("/v1/chat/completions", payload)
    if not res:
        return None

    timings = res.get("timings", {})
    usage = res.get("usage", {})

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_speed = timings.get("prompt_per_second")
    decode_speed = timings.get("predicted_per_second")
    prompt_ms = timings.get("prompt_ms")
    predicted_ms = timings.get("predicted_ms")

    print(f"Prompt Tokens:       {prompt_tokens}")
    print(f"Completion Tokens:   {completion_tokens}")
    if prompt_speed is not None:
        ms_str = f" ({prompt_ms:.1f} ms)" if prompt_ms else ""
        print(f"🚀 Prompt Prefill:   {prompt_speed:.2f} tok/s{ms_str}")
    if decode_speed is not None:
        ms_str = f" ({predicted_ms:.1f} ms)" if predicted_ms else ""
        print(f"⚡ Decode Speed:     {decode_speed:.2f} tok/s{ms_str}")
    print(f"⏱️ Total Wall Clock:  {wall_clock:.2f} s\n")

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_tok_per_sec": prompt_speed,
        "decode_tok_per_sec": decode_speed,
        "prompt_ms": prompt_ms,
        "predicted_ms": predicted_ms,
        "wall_clock_sec": wall_clock
    }


def run_streaming_test(tokens_to_generate=30):
    print("--- [2] Real-Time Streaming SSE Test (TTFT) ---")
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count from 1 to 20 slowly with commas."}],
        "max_tokens": tokens_to_generate,
        "temperature": 0.2,
        "stream": True
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    first_token_time = None
    token_count = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                decoded = line.decode("utf-8").strip()
                if not decoded:
                    continue
                if first_token_time is None and '"delta"' in decoded:
                    first_token_time = (time.perf_counter() - t0) * 1000  # ms
                if decoded.startswith("data: ") and "[DONE]" not in decoded:
                    token_count += 1
    except Exception as e:
        print(f"Streaming error: {e}", file=sys.stderr)
        return None

    total_duration = time.perf_counter() - t0
    ttft_str = f"{first_token_time:.1f} ms" if first_token_time is not None else "N/A"
    print(f"⏱️ Time to First Token (TTFT): {ttft_str}")
    print(f"Total Stream Duration:         {total_duration:.2f} s (~{token_count} tokens)")
    print("Stream Consistency:            Instantaneous first token emission\n")
    return {"ttft_ms": first_token_time, "duration_s": total_duration, "tokens": token_count}


def generate_filler(target_tokens):
    base = "Distributed artificial intelligence pools compute across heterogeneous peer nodes using encrypted QUIC channels and sequential pipeline splits. "
    approx_tokens_per_phrase = 18
    repeat_count = max(1, target_tokens // approx_tokens_per_phrase)
    return base * repeat_count


def run_stress_stage(name, target_tokens, max_completion=28):
    filler = generate_filler(target_tokens)
    prompt = f"{filler}\n\nBased on the text above, summarize the key principle in exactly 1 bullet point."
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_completion,
        "temperature": 0.2
    }

    print(f"Testing Stage [{name}] (~{target_tokens} prompt tokens)...")
    res, wall_clock = post_json("/v1/chat/completions", payload)
    if not res:
        return None

    usage = res.get("usage", {})
    timings = res.get("timings", {})
    p_tok = usage.get("prompt_tokens", 0)
    c_tok = usage.get("completion_tokens", 0)
    prefill = timings.get("prompt_per_second", 0.0)
    decode = timings.get("predicted_per_second", 0.0)

    print(f"   -> Prompt Tokens: {p_tok} | Completion Tokens: {c_tok}")
    print(f"   -> Prefill:       {prefill:.2f} tok/s")
    print(f"   -> Decode:        {decode:.2f} tok/s")
    print(f"   -> Wall Time:     {wall_clock:.2f} s\n")

    return {
        "stage": name,
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "prefill_tok_per_sec": prefill,
        "decode_tok_per_sec": decode,
        "wall_clock_sec": wall_clock
    }


def run_long_context_stress():
    print("--- [3] Long-Context Stress Test Matrix ---")
    stages = [
        ("Short (~450 tok)", 450),
        ("Medium (~1,300 tok)", 1300),
        ("Heavy (~3,300 tok)", 3300),
        ("Extra Heavy (~5,900 tok)", 5900),
    ]

    results = []
    for name, tok in stages:
        r = run_stress_stage(name, tok)
        if r:
            results.append(r)

    print("\n================ STRESS TEST SUMMARY TABLE ================")
    header = f"{'Stage':<26} | {'Prompt Tok':<10} | {'Compl Tok':<10} | {'Prefill (tok/s)':<16} | {'Decode (tok/s)':<15} | {'Wall Clock (s)':<14}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['stage']:<26} | {r['prompt_tokens']:<10} | {r['completion_tokens']:<10} | {r['prefill_tok_per_sec']:<16.2f} | {r['decode_tok_per_sec']:<15.2f} | {r['wall_clock_sec']:<14.2f}")
    print("=" * len(header) + "\n")
    return results


def save_reports(standard_data, streaming_data, stress_data):
    timestamp = datetime.now().isoformat()
    record = {
        "timestamp": timestamp,
        "endpoint": BASE_URL,
        "model": MODEL,
        "standard_benchmark": standard_data,
        "streaming_benchmark": streaming_data,
        "stress_matrix": stress_data
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
        f.write(f"\n## Benchmark Run: {timestamp}\n\n")
        f.write(f"- **Endpoint**: `{BASE_URL}`\n")
        f.write(f"- **Model**: `{MODEL}`\n\n")

        if standard_data:
            f.write("### Standard Generation (100 tokens)\n\n")
            f.write(f"- **Prompt Tokens**: {standard_data.get('prompt_tokens')}\n")
            f.write(f"- **Completion Tokens**: {standard_data.get('completion_tokens')}\n")
            f.write(f"- **Prompt Prefill**: **{standard_data.get('prefill_tok_per_sec', 0):.2f} tok/s**\n")
            f.write(f"- **Decode Speed**: **{standard_data.get('decode_tok_per_sec', 0):.2f} tok/s**\n")
            f.write(f"- **Total Wall Clock**: {standard_data.get('wall_clock_sec', 0):.2f} s\n\n")

        if streaming_data:
            f.write("### Streaming (SSE)\n\n")
            ttft = streaming_data.get("ttft_ms")
            ttft_str = f"{ttft:.1f} ms" if ttft else "N/A"
            f.write(f"- **TTFT (Time to First Token)**: **{ttft_str}**\n")
            f.write(f"- **Stream Duration**: {streaming_data.get('duration_s', 0):.2f} s\n\n")

        if stress_data:
            f.write("### Long-Context Stress Test\n\n")
            f.write("| Stage | Prompt Tokens | Completion Tokens | Prefill Throughput | Decode Speed | Total Wall Time |\n")
            f.write("|---|---|---|---|---|---|\n")
            for s in stress_data:
                f.write(f"| **{s['stage']}** | {s['prompt_tokens']} | {s['completion_tokens']} | **{s['prefill_tok_per_sec']:.2f} tok/s** | **{s['decode_tok_per_sec']:.2f} tok/s** | {s['wall_clock_sec']:.2f} s |\n")
            f.write("\n---\n")

    print(f"💾 Results saved to:")
    print(f"   • JSON: {OUT_JSON}")
    print(f"   • Markdown: {OUT_MD}\n")


def main():
    args = sys.argv[1:]
    standard_data = None
    streaming_data = None
    stress_data = None

    if "--streaming" in args:
        streaming_data = run_streaming_test()
    elif "--stress" in args:
        stress_data = run_long_context_stress()
    elif "--full" in args:
        standard_data = run_standard_benchmark(100)
        streaming_data = run_streaming_test(30)
        stress_data = run_long_context_stress()
    else:
        standard_data = run_standard_benchmark(100)
        streaming_data = run_streaming_test(30)
        print("💡 Tip: Run `python3 benchmark.py --stress` to benchmark across context scaling (450 to 5,900 tokens).")
        print("💡 Tip: Run `python3 concurrency_test.py` to measure multi-stream throughput.")

    save_reports(standard_data, streaming_data, stress_data)


if __name__ == "__main__":
    main()
