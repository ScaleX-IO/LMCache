#!/usr/bin/env python3
"""Decode throughput benchmark: measures output token rate during cold requests
where LMCache STORE (GPU→SSD write) runs concurrently with decode.

Usage:
    # Start lmcache server + vllm first (see bench_e2e.py), then:
    python bench_decode_throughput.py

    # Or let it start services automatically:
    python bench_decode_throughput.py --backend ugds --device /dev/ugds_drv0
    python bench_decode_throughput.py --backend cufile --slab-dir /mnt/ugds_test
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

MODEL = "/tmp/Qwen3-0.6B"
VLLM_PORT = 8000
COMPLETIONS_URL = f"http://127.0.0.1:{VLLM_PORT}/v1/completions"

PROMPT_TOKENS = [512, 1024, 2048]
MAX_TOKENS = 256
REPEATS = 3


def _unique_prompt(token_count, seed):
    return [100 + (seed + i * 104729) % 140000 for i in range(token_count)]


def request_streaming(prompt, max_tokens):
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        COMPLETIONS_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft_ms = None
    output_tokens = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            text = line.decode().strip()
            if not text.startswith("data: "):
                continue
            data_str = text[len("data: "):]
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            choices = chunk.get("choices", [])
            if choices:
                output_tokens += 1
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
    total_ms = (time.perf_counter() - t0) * 1000
    decode_ms = total_ms - ttft_ms if ttft_ms else total_ms
    decode_tokens = output_tokens - 1 if output_tokens > 1 else 1
    decode_tok_s = decode_tokens / (decode_ms / 1000)
    return {
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "output_tokens": output_tokens,
        "decode_tok_s": decode_tok_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cufile", "ugds"])
    parser.add_argument("--slab-dir", default="/mnt/ugds_test")
    parser.add_argument("--device", default="/dev/ugds_drv0")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--standalone", action="store_true",
                        help="connect to already-running vLLM (no service management)")
    args = parser.parse_args()

    manage_services = args.backend is not None and not args.standalone
    procs = []

    if manage_services:
        bench_e2e_path = os.path.join(os.path.dirname(__file__), "bench_e2e.py")
        sys.path.insert(0, os.path.dirname(bench_e2e_path))
        import bench_e2e

        env = {**os.environ}
        from pathlib import Path
        log_dir = Path("/tmp/lmcache-e2e-logs")

        if args.backend == "ugds":
            lmc = bench_e2e.start_lmcache_server(
                "ugds", args.device, env, log_dir / "lmcache-ugds.log")
        else:
            lmc = bench_e2e.start_lmcache_server(
                "cufile", args.slab_dir, env, log_dir / "lmcache-cufile.log")
        procs.append(lmc)
        vllm = bench_e2e.start_vllm(env, log_dir / "vllm-decode.log")
        procs.append(vllm)

        print("Warmup...", flush=True)
        request_streaming(_unique_prompt(64, 1), 1)
        time.sleep(1)

    try:
        print(f"\n=== Decode Throughput (max_tokens={args.max_tokens}) ===")
        print(f"{'Prompt':>8s}  {'Type':>5s}  {'TTFT(ms)':>10s}  {'Decode(ms)':>12s}  "
              f"{'Tokens':>6s}  {'Tok/s':>8s}")
        print("-" * 62)

        for prompt_len in PROMPT_TOKENS:
            for rep in range(args.repeats):
                seed = 50000 + prompt_len * 100 + rep * 7
                prompt = _unique_prompt(prompt_len, seed)

                r = request_streaming(prompt, args.max_tokens)
                tag = "cold"
                print(f"{prompt_len:>8d}  {tag:>5s}  {r['ttft_ms']:10.1f}  "
                      f"{r['decode_ms']:12.1f}  {r['output_tokens']:>6d}  "
                      f"{r['decode_tok_s']:8.1f}", flush=True)

                time.sleep(0.5)

    finally:
        if manage_services:
            import bench_e2e
            print("\nShutting down...", flush=True)
            for p in reversed(procs):
                bench_e2e.kill_proc(p)

    print("\nDone.")


if __name__ == "__main__":
    main()
