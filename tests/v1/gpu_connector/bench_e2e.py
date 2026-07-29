#!/usr/bin/env python3
"""End-to-end vLLM + LMCache cache-hit benchmark.

Starts a standalone LMCache server and vLLM with ``LMCacheMPConnector``. Runs
sequential cold/hot and concurrent hot cache-hit tests, then saves JSON results.

Usage:
    # MP connector with uGDS (990 PRO bound to ugds_drv)
    export LD_LIBRARY_PATH=/root/uGDS-workspace/uGDS/build
    python tests/v1/gpu_connector/bench_e2e.py \
        --backend ugds --device /dev/ugds_drv0

"""

# Standard
import argparse
import concurrent.futures
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import re
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
LOCAL_URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

SEQ_TOKEN_COUNTS = [256, 512, 1024, 2048, 3840]
SEQ_HOT_REPEATS = 5
CONC_TOKEN_COUNT = 1024
CONC_LEVELS = [1, 2, 4, 8]
CONC_ROUNDS = 3
STORE_SETTLE_SEC = 1.5
STORE_PATTERN = re.compile(r"Stored (\d+) tokens")
RETRIEVE_PATTERN = re.compile(r"Retrieved (\d+) tokens")


def build_vllm_command(
    model: str = MODEL,
    max_model_len: int = 4096,
    max_num_seqs: int | None = None,
) -> list[str]:
    """Build the vLLM 0.20.1 server command for LMCache MP mode.

    Args:
        model: Model name or local model path served by vLLM.
        max_model_len: Maximum model context length accepted by vLLM.
        max_num_seqs: Optional maximum number of concurrent sequences.

    Returns:
        The command and arguments used to start ``vllm serve``.

    Notes:
        vLLM 0.20.1 requires the hybrid KV cache manager to be disabled for
        ``LMCacheMPConnector``.
    """
    kv_config = json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {"lmcache.mp.port": LMCACHE_PORT},
        }
    )
    vllm_bin = os.path.join(os.path.dirname(sys.executable), "vllm")
    command = [
        vllm_bin,
        "serve",
        model,
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        "0.8",
        "--no-enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--kv-transfer-config",
        kv_config,
        "--disable-hybrid-kv-cache-manager",
    ]
    if max_num_seqs is not None:
        command.extend(["--max-num-seqs", str(max_num_seqs)])
    return command


def _unique_prompt(token_count: int, seed: int) -> list[int]:
    return [100 + (seed + i * 104729) % 140000 for i in range(token_count)]


class _RequestResult:
    __slots__ = ("cached_tokens", "latency_ms", "ttft_ms")

    def __init__(
        self,
        latency_ms: float,
        ttft_ms: float,
        cached_tokens: int | None = None,
    ) -> None:
        self.latency_ms = latency_ms
        self.ttft_ms = ttft_ms
        self.cached_tokens = cached_tokens


def _request(prompt: list[int], model: str = MODEL) -> _RequestResult:
    """Send a streaming completion request; measure TTFT and total latency."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        COMPLETIONS_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft_ms = None
    cached_tokens = None
    with LOCAL_URL_OPENER.open(req, timeout=120) as resp:
        for line in resp:
            text = line.decode().strip()
            if not text.startswith("data: "):
                continue
            data_str = text[len("data: ") :]
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            usage = chunk.get("usage")
            if usage is not None:
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = details.get("cached_tokens")
            choices = chunk.get("choices", [])
            if choices and ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000
    latency_ms = (time.perf_counter() - t0) * 1000
    if ttft_ms is None:
        raise RuntimeError("no token received in streaming response")
    return _RequestResult(latency_ms, ttft_ms, cached_tokens)


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
    proc: subprocess.Popen[bytes],
    log_path: Path,
    timeout: int = 180,
) -> None:
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
            with LOCAL_URL_OPENER.open(req, timeout=5):
                print(f"  {label} ready", flush=True)
                return
        except (urllib.error.URLError, OSError, ConnectionError):
            time.sleep(2)
    raise TimeoutError(
        f"{label} not ready after {timeout}s\n"
        f"--- {log_path} (tail) ---\n{_log_tail(log_path)}"
    )


def _start_logged_process(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
) -> subprocess.Popen[bytes]:
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
    backend: str,
    path: str,
    env: dict[str, str],
    log_path: Path,
    l1_size_gb: float = 8,
) -> subprocess.Popen[bytes]:
    """Start the standalone LMCache server and wait until it is healthy.

    Args:
        backend: GDS backend name, either ``cufile`` or ``ugds``.
        path: Filesystem slab directory or raw uGDS device path.
        env: Child process environment.
        log_path: Destination for combined stdout and stderr.
        l1_size_gb: SSD L1 capacity in GiB.

    Returns:
        Running LMCache server process.

    Raises:
        RuntimeError: If the process exits or does not become healthy in time.
    """
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
        str(l1_size_gb),
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


def start_vllm(
    env: dict[str, str],
    log_path: Path,
    model: str = MODEL,
    max_model_len: int = 4096,
    max_num_seqs: int | None = None,
) -> subprocess.Popen[bytes]:
    """Start vLLM with the selected LMCache connector.

    Args:
        env: Child process environment.
        log_path: Destination for combined stdout and stderr.
        model: Model name or local path.
        max_model_len: Maximum model context length.
        max_num_seqs: Optional maximum concurrent sequence count.

    Returns:
        Running vLLM server process.

    Raises:
        RuntimeError: If the process exits or does not become healthy in time.
    """
    cmd = build_vllm_command(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    env = {
        **env,
        "PYTHONUNBUFFERED": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    print(
        f"Starting vLLM (LMCacheMPConnector); log: {log_path}",
        flush=True,
    )
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


def kill_proc(proc: subprocess.Popen[bytes]) -> None:
    """Terminate a managed process group, escalating to SIGKILL if needed.

    Args:
        proc: Managed subprocess whose process group should be terminated.
    """
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)


def _parse_positive_int_list(raw: str, argument: str) -> list[int]:
    """Parse a comma-separated list of unique positive integers.

    Args:
        raw: Comma-separated integer values.
        argument: CLI argument name used in validation errors.

    Returns:
        Parsed integers in their original order.

    Raises:
        ValueError: If a value is missing, non-integer, non-positive, or repeated.
    """
    try:
        values = [int(value.strip()) for value in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"{argument} must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{argument} values must be positive")
    if len(set(values)) != len(values):
        raise ValueError(f"{argument} values must be unique")
    return values


def _log_token_total(log_path: Path, pattern: re.Pattern[str]) -> int:
    """Return the token total for matching LMCache operation log entries.

    Args:
        log_path: LMCache server log to scan.
        pattern: Compiled pattern containing the token count in group one.

    Returns:
        Sum of all matched token counts, or zero when the log does not exist.
    """
    if not log_path.exists():
        return 0
    return sum(
        int(match.group(1))
        for match in pattern.finditer(log_path.read_text(errors="replace"))
    )


def _wait_for_token_total(
    log_path: Path,
    pattern: re.Pattern[str],
    baseline: int,
    expected_delta: int,
    label: str,
    timeout: int = 300,
) -> int:
    """Wait until LMCache logs the expected STORE or RETRIEVE token total.

    Args:
        log_path: LMCache server log to scan.
        pattern: STORE or RETRIEVE token pattern.
        baseline: Token total before the measured operations.
        expected_delta: Required additional token total.
        label: Operation label used in errors.
        timeout: Maximum wait time in seconds.

    Returns:
        Observed token delta once it reaches the expectation.

    Raises:
        TimeoutError: If the expected total is not observed before the timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        observed = _log_token_total(log_path, pattern) - baseline
        if observed >= expected_delta:
            return observed
        time.sleep(0.2)
    observed = _log_token_total(log_path, pattern) - baseline
    raise TimeoutError(
        f"{label} logged {observed} tokens; expected at least {expected_delta}\n"
        f"--- {log_path} (tail) ---\n{_log_tail(log_path)}"
    )


def validate_external_cache_hits(
    cached_tokens: list[int | None], expected_tokens: int, label: str
) -> None:
    """Require every hot request to report a full external cache hit.

    Args:
        cached_tokens: Cached-token count reported by each hot request.
        expected_tokens: Full prompt token count expected from LMCache.
        label: Request group label used in errors.

    Raises:
        RuntimeError: If any request omits or under-reports cached tokens.

    Notes:
        vLLM APC is disabled, so reported cached tokens can only come from the
        external KV connector.
    """
    if any(tokens != expected_tokens for tokens in cached_tokens):
        raise RuntimeError(
            f"{label} expected {expected_tokens} external cached tokens per request; "
            f"observed {cached_tokens}"
        )


def run_sequential(
    token_counts: list[int] | None = None,
    hot_repeats: int = SEQ_HOT_REPEATS,
    model: str = MODEL,
    seed_base: int = 42000,
) -> dict[str, object]:
    """Run the original interleaved cold/hot sequential benchmark.

    Args:
        token_counts: Prompt lengths to benchmark.
        hot_repeats: Number of hot requests per prompt.
        model: Model name or path used in completion requests.
        seed_base: Base seed for deterministic token IDs.

    Returns:
        Sequential benchmark samples and p50 values.
    """
    print("\n=== Sequential cold/hot benchmark ===", flush=True)
    if token_counts is None:
        token_counts = SEQ_TOKEN_COUNTS
    results = []
    for token_count in token_counts:
        seed = seed_base + token_count * 31
        prompt = _unique_prompt(token_count, seed)

        cold = _request(prompt, model)
        print(
            f"  {token_count} tokens  cold={cold.latency_ms:.1f}ms  "
            f"ttft={cold.ttft_ms:.1f}ms",
            flush=True,
        )

        time.sleep(STORE_SETTLE_SEC)

        hot_results = []
        for _ in range(hot_repeats):
            r = _request(prompt, model)
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
                "hot_cached_token_samples": [
                    request.cached_tokens for request in hot_results
                ],
                "hot_p50_ms": hot_p50,
                "hot_ttft_p50_ms": hot_ttft_p50,
            }
        )
    return {"sequential": results}


def run_concurrent(
    token_count: int = CONC_TOKEN_COUNT,
    levels: list[int] | None = None,
    rounds: int = CONC_ROUNDS,
    model: str = MODEL,
    seed_base: int = 90000,
) -> dict[str, object]:
    """Run the original interleaved cold/hot concurrent benchmark.

    Args:
        token_count: Prompt tokens per concurrent request.
        levels: Concurrent request counts to benchmark.
        rounds: Hot rounds per concurrency level.
        model: Model name or path used in completion requests.
        seed_base: Base seed for deterministic token IDs.

    Returns:
        Concurrent benchmark samples and median summary values.
    """
    print("\n=== Concurrent hot benchmark ===", flush=True)
    if levels is None:
        levels = CONC_LEVELS
    results = []
    for level in levels:
        prompts = []
        for i in range(level):
            seed = seed_base + level * 1000 + i * 7919
            p = _unique_prompt(token_count, seed)
            cold = _request(p, model)
            print(
                f"  warm c={level} slot={i}  cold={cold.latency_ms:.1f}ms", flush=True
            )
            prompts.append(p)
        time.sleep(STORE_SETTLE_SEC)

        round_results = []
        for rnd in range(rounds):
            with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
                t0 = time.perf_counter()
                futures = [pool.submit(_request, p, model) for p in prompts]
                req_results = [f.result() for f in futures]
                wall_ms = (time.perf_counter() - t0) * 1000

            latencies = [r.latency_ms for r in req_results]
            ttfts = [r.ttft_ms for r in req_results]
            latencies_sorted = sorted(latencies)
            ttfts_sorted = sorted(ttfts)
            p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)
            p95 = latencies_sorted[p95_idx]
            ttft_p95 = ttfts_sorted[p95_idx]
            throughput_ktok = (level * token_count) / (wall_ms / 1000) / 1000
            round_results.append(
                {
                    "wall_ms": wall_ms,
                    "latencies_ms": latencies,
                    "ttfts_ms": ttfts,
                    "cached_tokens": [
                        request.cached_tokens for request in req_results
                    ],
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
                "tokens_per_request": token_count,
                "rounds": round_results,
                "median_throughput_ktok_s": med_throughput,
                "median_p95_ms": med_p95,
                "median_ttft_p95_ms": med_ttft_p95,
            }
        )
    return {"concurrent": results}


def _run_ssd_only_sequential(
    prompts: list[tuple[int, list[int]]],
    hot_repeats: int,
    model: str,
    lmcache_log: Path,
) -> dict[str, object]:
    """Preload sequential prompts, then measure SSD-only hot TTFT.

    Args:
        prompts: Prompt lengths paired with deterministic token IDs.
        hot_repeats: Hot samples per prompt.
        model: Model name or path used in completion requests.
        lmcache_log: LMCache log used to verify SSD operations.

    Returns:
        Cold samples, hot p50 values, and operation evidence.
    """
    print("\n=== Sequential SSD preload ===", flush=True)
    store_baseline = _log_token_total(lmcache_log, STORE_PATTERN)
    cold_by_tokens: dict[int, _RequestResult] = {}
    for token_count, prompt in prompts:
        cold = _request(prompt, model)
        cold_by_tokens[token_count] = cold
        print(
            f"  {token_count} tokens  cold={cold.latency_ms:.1f}ms  "
            f"ttft={cold.ttft_ms:.1f}ms",
            flush=True,
        )

    expected_store_tokens = sum(token_count for token_count, _ in prompts)
    stored_tokens = _wait_for_token_total(
        lmcache_log,
        STORE_PATTERN,
        store_baseline,
        expected_store_tokens,
        "sequential STORE",
    )
    time.sleep(STORE_SETTLE_SEC)
    print(f"  verified STORE tokens={stored_tokens}", flush=True)
    return {
        "cold_by_tokens": cold_by_tokens,
        "store_tokens": stored_tokens,
    }


def _measure_ssd_only_sequential(
    prompts: list[tuple[int, list[int]]],
    cold_by_tokens: dict[int, _RequestResult],
    hot_repeats: int,
    model: str,
    lmcache_log: Path,
) -> dict[str, object]:
    """Measure sequential hot requests after the SSD STORE barrier.

    Args:
        prompts: Prompt lengths paired with deterministic token IDs.
        cold_by_tokens: Cold preload samples keyed by prompt length.
        hot_repeats: Hot samples per prompt.
        model: Model name or path used in completion requests.
        lmcache_log: LMCache log used to verify SSD operations.

    Returns:
        Sequential hot samples, p50 values, and retrieve evidence.
    """
    print("\n=== Sequential SSD-only hot benchmark ===", flush=True)
    retrieve_baseline = _log_token_total(lmcache_log, RETRIEVE_PATTERN)
    results = []
    for token_count, prompt in prompts:
        hot_results = [_request(prompt, model) for _ in range(hot_repeats)]
        validate_external_cache_hits(
            [request.cached_tokens for request in hot_results],
            token_count,
            f"sequential {token_count}",
        )
        hot_latencies = [request.latency_ms for request in hot_results]
        hot_ttfts = [request.ttft_ms for request in hot_results]
        hot_p50 = statistics.median(hot_latencies)
        hot_ttft_p50 = statistics.median(hot_ttfts)
        cold = cold_by_tokens[token_count]
        print(
            f"  {token_count} tokens  hot_p50={hot_p50:.1f}ms  "
            f"ttft_p50={hot_ttft_p50:.1f}ms  "
            f"cached={[request.cached_tokens for request in hot_results]}",
            flush=True,
        )
        results.append(
            {
                "tokens": token_count,
                "cold_ms": cold.latency_ms,
                "cold_ttft_ms": cold.ttft_ms,
                "hot_latency_samples_ms": hot_latencies,
                "hot_ttft_samples_ms": hot_ttfts,
                "hot_cached_token_samples": [
                    request.cached_tokens for request in hot_results
                ],
                "hot_p50_ms": hot_p50,
                "hot_ttft_p50_ms": hot_ttft_p50,
            }
        )

    expected_retrieve_tokens = sum(
        token_count * hot_repeats for token_count, _ in prompts
    )
    retrieved_tokens = _wait_for_token_total(
        lmcache_log,
        RETRIEVE_PATTERN,
        retrieve_baseline,
        expected_retrieve_tokens,
        "sequential RETRIEVE",
    )
    return {
        "sequential": results,
        "evidence": {
            "retrieve_tokens": retrieved_tokens,
            "expected_retrieve_tokens": expected_retrieve_tokens,
        },
    }


def _run_ssd_only_concurrent_preload(
    prompts: list[list[int]],
    token_count: int,
    model: str,
    lmcache_log: Path,
) -> dict[str, int]:
    """Preload the fixed concurrent prompt pool into the SSD slab.

    Args:
        prompts: Fixed prompt pool sized for maximum concurrency.
        token_count: Tokens in each prompt.
        model: Model name or path used in completion requests.
        lmcache_log: LMCache log used to verify SSD operations.

    Returns:
        Observed and expected STORE token totals.
    """
    print("\n=== Concurrent SSD preload ===", flush=True)
    store_baseline = _log_token_total(lmcache_log, STORE_PATTERN)
    for index, prompt in enumerate(prompts, start=1):
        _request(prompt, model)
        if index == 1 or index % 16 == 0 or index == len(prompts):
            print(f"  preloaded {index}/{len(prompts)} prompts", flush=True)
    expected_store_tokens = len(prompts) * token_count
    stored_tokens = _wait_for_token_total(
        lmcache_log,
        STORE_PATTERN,
        store_baseline,
        expected_store_tokens,
        "concurrent STORE",
        timeout=600,
    )
    time.sleep(STORE_SETTLE_SEC)
    print(f"  verified STORE tokens={stored_tokens}", flush=True)
    return {
        "store_tokens": stored_tokens,
        "expected_store_tokens": expected_store_tokens,
    }


def _measure_ssd_only_concurrent(
    prompts: list[list[int]],
    token_count: int,
    levels: list[int],
    rounds: int,
    model: str,
    lmcache_log: Path,
) -> dict[str, object]:
    """Measure concurrent SSD-only throughput after the SSD STORE barrier.

    Args:
        prompts: Fixed preloaded prompt pool.
        token_count: Tokens in each prompt.
        levels: Concurrent request counts to benchmark.
        rounds: Hot rounds per concurrency level.
        model: Model name or path used in completion requests.
        lmcache_log: LMCache log used to verify SSD operations.

    Returns:
        Successful concurrency samples, upper bound, and optional failure.
    """
    print("\n=== Concurrent SSD-only hot benchmark ===", flush=True)
    results = []
    failure = None
    for level in levels:
        retrieve_baseline = _log_token_total(lmcache_log, RETRIEVE_PATTERN)
        round_results = []
        try:
            for round_index in range(rounds):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=level
                ) as pool:
                    t0 = time.perf_counter()
                    futures = [
                        pool.submit(_request, prompt, model)
                        for prompt in prompts[:level]
                    ]
                    request_results = [future.result() for future in futures]
                    wall_ms = (time.perf_counter() - t0) * 1000

                validate_external_cache_hits(
                    [request.cached_tokens for request in request_results],
                    token_count,
                    f"concurrency {level}",
                )
                latencies = [request.latency_ms for request in request_results]
                ttfts = [request.ttft_ms for request in request_results]
                p95_index = max(0, int(len(latencies) * 0.95) - 1)
                p95 = sorted(latencies)[p95_index]
                ttft_p95 = sorted(ttfts)[p95_index]
                throughput = (level * token_count) / (wall_ms / 1000) / 1000
                round_results.append(
                    {
                        "wall_ms": wall_ms,
                        "latencies_ms": latencies,
                        "ttfts_ms": ttfts,
                        "cached_tokens": [
                            request.cached_tokens for request in request_results
                        ],
                        "p95_ms": p95,
                        "ttft_p95_ms": ttft_p95,
                        "throughput_ktok_s": throughput,
                    }
                )
                print(
                    f"  c={level} round={round_index} wall={wall_ms:.1f}ms "
                    f"ttft_p95={ttft_p95:.1f}ms "
                    f"throughput={throughput:.1f}kTok/s",
                    flush=True,
                )

            expected_retrieve_tokens = level * token_count * rounds
            retrieved_tokens = _wait_for_token_total(
                lmcache_log,
                RETRIEVE_PATTERN,
                retrieve_baseline,
                expected_retrieve_tokens,
                f"concurrency {level} RETRIEVE",
            )
            results.append(
                {
                    "concurrency": level,
                    "tokens_per_request": token_count,
                    "rounds": round_results,
                    "median_throughput_ktok_s": statistics.median(
                        result["throughput_ktok_s"] for result in round_results
                    ),
                    "median_p95_ms": statistics.median(
                        result["p95_ms"] for result in round_results
                    ),
                    "median_ttft_p95_ms": statistics.median(
                        result["ttft_p95_ms"] for result in round_results
                    ),
                    "retrieve_tokens": retrieved_tokens,
                    "expected_retrieve_tokens": expected_retrieve_tokens,
                }
            )
        except Exception as exc:
            failure = {"concurrency": level, "error": str(exc)}
            print(f"  stopping at c={level}: {exc}", flush=True)
            break

    return {
        "concurrent": results,
        "max_successful_concurrency": (
            results[-1]["concurrency"] if results else None
        ),
        "failure": failure,
    }


def run_ssd_only_phase(
    phase: str,
    backend: str,
    lmcache_path: str,
    env: dict[str, str],
    log_dir: Path,
    l1_size_gb: float,
    model: str,
    max_model_len: int,
    max_num_seqs: int | None,
    seq_token_counts: list[int],
    seq_hot_repeats: int,
    conc_token_count: int,
    conc_levels: list[int],
    conc_rounds: int,
) -> dict[str, object]:
    """Run one isolated SSD-only pressure phase.

    Args:
        phase: ``sequential`` or ``concurrent``.
        backend: LMCache GDS backend name.
        lmcache_path: Raw device or filesystem slab directory.
        env: Child process environment.
        log_dir: Directory for phase-specific service logs.
        l1_size_gb: SSD slab capacity in GiB.
        model: Model name or local path.
        max_model_len: Maximum model context length.
        max_num_seqs: Optional maximum concurrent sequences.
        seq_token_counts: Sequential prompt lengths.
        seq_hot_repeats: Hot samples per sequential prompt.
        conc_token_count: Tokens in each concurrent prompt.
        conc_levels: Concurrent request counts.
        conc_rounds: Hot rounds per concurrency level.

    Returns:
        Phase results and SSD operation evidence.

    Raises:
        ValueError: If ``phase`` is unsupported.
    """
    lmcache_log = log_dir / f"lmcache-{backend}-{phase}.log"
    vllm_log = log_dir / f"vllm-{backend}-{phase}.log"
    lmcache_proc = start_lmcache_server(
        backend, lmcache_path, env, lmcache_log, l1_size_gb
    )
    vllm_proc: subprocess.Popen[bytes] | None = None
    try:
        vllm_proc = start_vllm(
            env,
            vllm_log,
            model,
            max_model_len,
            max_num_seqs,
        )
        print("\nWarmup request before SSD preload...", flush=True)
        _request(_unique_prompt(64, 1), model)
        time.sleep(1)

        if phase == "sequential":
            prompts = [
                (
                    token_count,
                    _unique_prompt(token_count, 42000 + token_count * 31),
                )
                for token_count in seq_token_counts
            ]
            preload = _run_ssd_only_sequential(
                prompts, seq_hot_repeats, model, lmcache_log
            )
            measured = _measure_ssd_only_sequential(
                prompts,
                preload["cold_by_tokens"],
                seq_hot_repeats,
                model,
                lmcache_log,
            )
            measured["evidence"]["store_tokens"] = preload["store_tokens"]
            measured["evidence"]["expected_store_tokens"] = sum(
                seq_token_counts
            )
            return measured

        if phase == "concurrent":
            max_concurrency = max(conc_levels)
            prompts = [
                _unique_prompt(conc_token_count, 90000 + index * 7919)
                for index in range(max_concurrency)
            ]
            preload = _run_ssd_only_concurrent_preload(
                prompts, conc_token_count, model, lmcache_log
            )
            measured = _measure_ssd_only_concurrent(
                prompts,
                conc_token_count,
                conc_levels,
                conc_rounds,
                model,
                lmcache_log,
            )
            measured["evidence"] = preload
            return measured

        raise ValueError(f"unsupported SSD-only phase: {phase}")
    finally:
        if vllm_proc is not None:
            kill_proc(vllm_proc)
        kill_proc(lmcache_proc)


def main() -> None:
    """Parse benchmark configuration, run selected phases, and save JSON."""
    parser = argparse.ArgumentParser(description="LMCache E2E cache-hit benchmark")
    parser.add_argument(
        "--backend",
        choices=["cufile", "ugds"],
        help="standalone LMCache server backend (required for MP connector)",
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
        "--l1-size-gb",
        type=float,
        default=8,
        help="LMCache L1 slab size in GiB (default: 8)",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"model name or local path (default: {MODEL})",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM maximum model length (default: 4096)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="vLLM maximum concurrent sequences (default: vLLM default)",
    )
    parser.add_argument(
        "--seq-token-counts",
        default=",".join(str(value) for value in SEQ_TOKEN_COUNTS),
        help="comma-separated sequential prompt lengths",
    )
    parser.add_argument(
        "--seq-hot-repeats",
        type=int,
        default=SEQ_HOT_REPEATS,
        help=f"hot samples per sequential prompt (default: {SEQ_HOT_REPEATS})",
    )
    parser.add_argument(
        "--conc-token-count",
        type=int,
        default=CONC_TOKEN_COUNT,
        help=f"tokens per concurrent prompt (default: {CONC_TOKEN_COUNT})",
    )
    parser.add_argument(
        "--conc-levels",
        default=",".join(str(value) for value in CONC_LEVELS),
        help="comma-separated concurrency levels",
    )
    parser.add_argument(
        "--conc-rounds",
        type=int,
        default=CONC_ROUNDS,
        help=f"hot rounds per concurrency level (default: {CONC_ROUNDS})",
    )
    parser.add_argument(
        "--run-mode",
        choices=["sequential", "concurrent", "all"],
        default="all",
        help="benchmark phase selection (default: all)",
    )
    parser.add_argument(
        "--ssd-only",
        action="store_true",
        help="preload SSD and require full external cache hits and retrieves",
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

    try:
        seq_token_counts = _parse_positive_int_list(
            args.seq_token_counts, "--seq-token-counts"
        )
        conc_levels = _parse_positive_int_list(args.conc_levels, "--conc-levels")
    except ValueError as exc:
        parser.error(str(exc))

    if args.backend is None:
        parser.error("--backend is required for LMCacheMPConnector")
    if args.l1_size_gb <= 0:
        parser.error("--l1-size-gb must be positive")
    if args.max_model_len <= 0:
        parser.error("--max-model-len must be positive")
    if args.max_num_seqs is not None and args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive")
    if args.seq_hot_repeats <= 0 or args.conc_rounds <= 0:
        parser.error("repeat and round counts must be positive")
    if args.conc_token_count <= 0:
        parser.error("--conc-token-count must be positive")
    if max(seq_token_counts) + 1 > args.max_model_len:
        parser.error("sequential prompt plus output exceeds --max-model-len")
    if args.conc_token_count + 1 > args.max_model_len:
        parser.error("concurrent prompt plus output exceeds --max-model-len")
    if args.max_num_seqs is not None and max(conc_levels) > args.max_num_seqs:
        parser.error("--conc-levels cannot exceed --max-num-seqs")
    if not args.output:
        args.output = f"bench_e2e_{args.backend}.json"

    env: dict[str, str] = {**os.environ}
    log_dir = Path(args.log_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    phases = (
        [args.run_mode]
        if args.run_mode != "all"
        else ["sequential", "concurrent"]
    )
    output: dict[str, object] = {
        "backend": args.backend,
        "model": args.model,
        "kv_connector": "LMCacheMPConnector",
        "managed_lmcache_server": True,
        "ssd_only": args.ssd_only,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "l1_size_gb": args.l1_size_gb,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "seq_token_counts": seq_token_counts,
            "seq_hot_repeats": args.seq_hot_repeats,
            "conc_token_count": args.conc_token_count,
            "conc_levels": conc_levels,
            "conc_rounds": args.conc_rounds,
            "run_mode": args.run_mode,
        },
        "versions": {
            package: metadata.version(package)
            for package in ("lmcache", "torch", "vllm")
        },
    }

    if args.ssd_only:
        assert args.backend is not None
        lmcache_path = args.device if args.backend == "ugds" else args.slab_dir
        evidence: dict[str, object] = {}
        try:
            for phase in phases:
                phase_result = run_ssd_only_phase(
                    phase,
                    args.backend,
                    lmcache_path,
                    env,
                    log_dir,
                    args.l1_size_gb,
                    args.model,
                    args.max_model_len,
                    args.max_num_seqs,
                    seq_token_counts,
                    args.seq_hot_repeats,
                    args.conc_token_count,
                    conc_levels,
                    args.conc_rounds,
                )
                phase_evidence = phase_result.pop("evidence", {})
                evidence[phase] = phase_evidence
                output.update(phase_result)
        finally:
            output["evidence"] = evidence
            output["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            output_path.write_text(json.dumps(output, indent=2))
            print(f"\nResults saved to {output_path}", flush=True)
        return

    procs: list[subprocess.Popen[bytes]] = []
    try:
        assert args.backend is not None
        lmcache_path = args.device if args.backend == "ugds" else args.slab_dir
        lmc = start_lmcache_server(
            args.backend,
            lmcache_path,
            env,
            log_dir / f"lmcache-{args.backend}.log",
            args.l1_size_gb,
        )
        procs.append(lmc)

        vllm_proc = start_vllm(
            env,
            log_dir / f"vllm-{args.backend}.log",
            args.model,
            args.max_model_len,
            args.max_num_seqs,
        )
        procs.append(vllm_proc)

        # Warmup: one throwaway request to trigger JIT/compile
        print("\nWarmup request...", flush=True)
        _request(_unique_prompt(64, 1), args.model)
        time.sleep(1)

        if "sequential" in phases:
            output.update(
                run_sequential(
                    seq_token_counts,
                    args.seq_hot_repeats,
                    args.model,
                )
            )
        if "concurrent" in phases:
            output.update(
                run_concurrent(
                    args.conc_token_count,
                    conc_levels,
                    args.conc_rounds,
                    args.model,
                )
            )
        output["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        output_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {output_path}", flush=True)

    finally:
        print("\nShutting down services...", flush=True)
        for p in reversed(procs):
            kill_proc(p)
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
