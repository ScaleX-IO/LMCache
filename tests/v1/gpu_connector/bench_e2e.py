#!/usr/bin/env python3
"""End-to-end vLLM + LMCache benchmark: cuFile GDS vs uGDS.

Starts lmcache server and vllm serve, runs sequential cold/hot and concurrent
hot cache-hit tests, saves results to JSON.

Usage:
    # uGDS backend (990 PRO bound to ugds_drv)
    export LD_LIBRARY_PATH=/root/uGDS-workspace/uGDS/build
    python dev-docs/bench_lmcache_e2e.py --backend ugds --device /dev/ugds_drv0

    # cuFile GDS backend (990 PRO bound to nvme, mounted at /mnt/ugds_test)
    python dev-docs/bench_lmcache_e2e.py --backend cufile --slab-dir /mnt/ugds_test
"""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODEL = "/tmp/Qwen3-0.6B"
VLLM_PORT = 8000
LMCACHE_PORT = 6555
LMCACHE_HTTP_PORT = 8080
COMPLETIONS_URL = f"http://127.0.0.1:{VLLM_PORT}/v1/completions"

SEQ_TOKEN_COUNTS = [256, 512, 1024, 2048, 3840]
SEQ_HOT_REPEATS = 5
CONC_TOKEN_COUNT = 1024
CONC_LEVELS = [1, 2, 4, 8]
CONC_ROUNDS = 3
STORE_SETTLE_SEC = 1.5


def _unique_prompt(token_count: int, seed: int) -> list[int]:
    return [100 + (seed + i * 104729) % 140000 for i in range(token_count)]


class _RequestResult:
    __slots__ = ("latency_ms", "ttft_ms")

    def __init__(self, latency_ms: float, ttft_ms: float):
        self.latency_ms = latency_ms
        self.ttft_ms = ttft_ms


def _request(prompt: list[int]) -> _RequestResult:
    """Send a streaming completion request; measure TTFT and total latency."""
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        COMPLETIONS_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft_ms = None
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            text = line.decode().strip()
            if not text.startswith("data: "):
                continue
            data_str = text[len("data: ") :]
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            choices = chunk.get("choices", [])
            if choices and ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000
    latency_ms = (time.perf_counter() - t0) * 1000
    if ttft_ms is None:
        raise RuntimeError("no token received in streaming response")
    return _RequestResult(latency_ms, ttft_ms)


def _log_tail(log_path: Path, lines: int = 80) -> str:
    try:
        return "".join(
            log_path.read_text(errors="replace").splitlines(keepends=True)[-lines:]
        )
    except OSError as exc:
        return f"<unable to read {log_path}: {exc}>"


def _wait_for_service(
    url: str,
    label: str,
    proc: subprocess.Popen,
    log_path: Path,
    timeout: int = 180,
):
    deadline = time.time() + timeout
    while time.time() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            raise RuntimeError(
                f"{label} exited with status {returncode} during startup\n"
                f"--- {log_path} (tail) ---\n{_log_tail(log_path)}"
            )
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5):
                print(f"  {label} ready", flush=True)
                return
        except (urllib.error.URLError, OSError, ConnectionError):
            time.sleep(2)
    raise TimeoutError(
        f"{label} not ready after {timeout}s\n"
        f"--- {log_path} (tail) ---\n{_log_tail(log_path)}"
    )


def _start_logged_process(
    cmd: list[str], env: dict, log_path: Path
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        return subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def start_lmcache_server(
    backend: str, path: str, env: dict, log_path: Path
) -> subprocess.Popen:
    lmcache_bin = os.path.join(os.path.dirname(sys.executable), "lmcache")
    cmd = [
        lmcache_bin,
        "server",
        "--host",
        "localhost",
        "--port",
        str(LMCACHE_PORT),
        "--http-port",
        str(LMCACHE_HTTP_PORT),
        "--chunk-size",
        "256",
        "--max-gpu-workers",
        "4",
        "--l1-size-gb",
        "8",
        "--eviction-policy",
        "LRU",
        "--gds-l1-backend",
        backend,
        "--gds-l1-path",
        path,
        "--disable-metrics",
    ]
    print(
        f"Starting lmcache server ({backend}, {path}); log: {log_path}",
        flush=True,
    )
    proc = _start_logged_process(cmd, env, log_path)
    try:
        _wait_for_service(
            f"http://127.0.0.1:{LMCACHE_HTTP_PORT}/healthcheck",
            "lmcache server",
            proc,
            log_path,
        )
    except Exception:
        kill_proc(proc)
        raise
    return proc


def start_vllm(env: dict, log_path: Path) -> subprocess.Popen:
    kv_config = json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {"lmcache.mp.port": LMCACHE_PORT},
        }
    )
    vllm_bin = os.path.join(os.path.dirname(sys.executable), "vllm")
    cmd = [
        vllm_bin,
        "serve",
        MODEL,
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.8",
        "--no-enable-prefix-caching",
        "--kv-transfer-config",
        kv_config,
    ]
    env = {
        **env,
        "PYTHONUNBUFFERED": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    print(f"Starting vLLM; log: {log_path}", flush=True)
    proc = _start_logged_process(cmd, env, log_path)
    try:
        _wait_for_service(
            f"http://127.0.0.1:{VLLM_PORT}/health",
            "vLLM",
            proc,
            log_path,
            timeout=300,
        )
    except Exception:
        kill_proc(proc)
        raise
    return proc


def kill_proc(proc: subprocess.Popen):
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)


def run_sequential(seed_base: int = 42000) -> dict:
    print("\n=== Sequential cold/hot benchmark ===", flush=True)
    results = []
    for token_count in SEQ_TOKEN_COUNTS:
        seed = seed_base + token_count * 31
        prompt = _unique_prompt(token_count, seed)

        cold = _request(prompt)
        print(
            f"  {token_count} tokens  cold={cold.latency_ms:.1f}ms  "
            f"ttft={cold.ttft_ms:.1f}ms",
            flush=True,
        )

        time.sleep(STORE_SETTLE_SEC)

        hot_results = []
        for _ in range(SEQ_HOT_REPEATS):
            r = _request(prompt)
            hot_results.append(r)
        hot_latencies = [r.latency_ms for r in hot_results]
        hot_ttfts = [r.ttft_ms for r in hot_results]
        hot_p50 = statistics.median(hot_latencies)
        hot_ttft_p50 = statistics.median(hot_ttfts)
        print(
            f"  {token_count} tokens  hot_p50={hot_p50:.1f}ms  "
            f"ttft_p50={hot_ttft_p50:.1f}ms  "
            f"latencies={[f'{s:.1f}' for s in hot_latencies]}",
            flush=True,
        )

        results.append(
            {
                "tokens": token_count,
                "cold_ms": cold.latency_ms,
                "cold_ttft_ms": cold.ttft_ms,
                "hot_latency_samples_ms": hot_latencies,
                "hot_ttft_samples_ms": hot_ttfts,
                "hot_p50_ms": hot_p50,
                "hot_ttft_p50_ms": hot_ttft_p50,
            }
        )
    return {"sequential": results}


def run_concurrent(seed_base: int = 90000) -> dict:
    print("\n=== Concurrent hot benchmark ===", flush=True)
    results = []
    for level in CONC_LEVELS:
        prompts = []
        for i in range(level):
            seed = seed_base + level * 1000 + i * 7919
            p = _unique_prompt(CONC_TOKEN_COUNT, seed)
            cold = _request(p)
            print(
                f"  warm c={level} slot={i}  cold={cold.latency_ms:.1f}ms", flush=True
            )
            prompts.append(p)
        time.sleep(STORE_SETTLE_SEC)

        round_results = []
        for rnd in range(CONC_ROUNDS):
            with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
                t0 = time.perf_counter()
                futures = [pool.submit(_request, p) for p in prompts]
                req_results = [f.result() for f in futures]
                wall_ms = (time.perf_counter() - t0) * 1000

            latencies = [r.latency_ms for r in req_results]
            ttfts = [r.ttft_ms for r in req_results]
            latencies_sorted = sorted(latencies)
            ttfts_sorted = sorted(ttfts)
            p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)
            p95 = latencies_sorted[p95_idx]
            ttft_p95 = ttfts_sorted[p95_idx]
            throughput_ktok = (level * CONC_TOKEN_COUNT) / (wall_ms / 1000) / 1000
            round_results.append(
                {
                    "wall_ms": wall_ms,
                    "latencies_ms": latencies,
                    "ttfts_ms": ttfts,
                    "p95_ms": p95,
                    "ttft_p95_ms": ttft_p95,
                    "throughput_ktok_s": throughput_ktok,
                }
            )
            print(
                f"  c={level} round={rnd}  wall={wall_ms:.1f}ms  "
                f"p95={p95:.1f}ms  ttft_p95={ttft_p95:.1f}ms  "
                f"throughput={throughput_ktok:.1f}kTok/s",
                flush=True,
            )

        med_throughput = statistics.median(
            [r["throughput_ktok_s"] for r in round_results]
        )
        med_p95 = statistics.median([r["p95_ms"] for r in round_results])
        med_ttft_p95 = statistics.median([r["ttft_p95_ms"] for r in round_results])
        results.append(
            {
                "concurrency": level,
                "tokens_per_request": CONC_TOKEN_COUNT,
                "rounds": round_results,
                "median_throughput_ktok_s": med_throughput,
                "median_p95_ms": med_p95,
                "median_ttft_p95_ms": med_ttft_p95,
            }
        )
    return {"concurrent": results}


def main():
    parser = argparse.ArgumentParser(
        description="LMCache E2E benchmark: cuFile GDS vs uGDS"
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["cufile", "ugds"],
    )
    parser.add_argument(
        "--slab-dir",
        default="/mnt/ugds_test",
        help="slab directory for cuFile backend",
    )
    parser.add_argument(
        "--device",
        default="/dev/ugds_drv0",
        help="raw device for uGDS backend",
    )
    parser.add_argument(
        "--output",
        default="",
        help="output JSON path (default: bench_e2e_{backend}.json)",
    )
    parser.add_argument(
        "--log-dir",
        default="/tmp/lmcache-e2e-logs",
        help="directory for lmcache and vLLM service logs",
    )
    args = parser.parse_args()

    if not args.output:
        args.output = f"bench_e2e_{args.backend}.json"

    env = {**os.environ}
    log_dir = Path(args.log_dir)
    if args.backend == "ugds":
        lmcache_backend = "ugds"
        lmcache_path = args.device
    else:
        lmcache_backend = "cufile"
        lmcache_path = args.slab_dir

    procs = []
    try:
        lmc = start_lmcache_server(
            lmcache_backend,
            lmcache_path,
            env,
            log_dir / f"lmcache-{args.backend}.log",
        )
        procs.append(lmc)
        vllm = start_vllm(env, log_dir / f"vllm-{args.backend}.log")
        procs.append(vllm)

        # Warmup: one throwaway request to trigger JIT/compile
        print("\nWarmup request...", flush=True)
        _request(_unique_prompt(64, 1))
        time.sleep(1)

        seq = run_sequential()
        conc = run_concurrent()

        output = {
            "backend": args.backend,
            "model": MODEL,
            **seq,
            **conc,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}", flush=True)

    finally:
        print("\nShutting down services...", flush=True)
        for p in reversed(procs):
            kill_proc(p)
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
