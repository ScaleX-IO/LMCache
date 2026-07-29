# uGDS vs. cuFile GDS Under High LLM Inference Load

## Overview

This experiment compares the end-to-end cache-hit performance of uGDS and
NVIDIA cuFile GDS using the same GPU, SSD, model, and LMCache configuration.
It evaluates two independent sources of storage pressure:

1. **Context pressure:** increase prompt length at concurrency 1 and measure
   hot Time to First Token (TTFT).
2. **Concurrency pressure:** increase the number of concurrent requests at a
   fixed prompt length and measure aggregate throughput.

Only one pressure dimension changes at a time. This keeps the experiment small
and makes each result attributable to either context size or concurrency rather
than an interaction between both variables.

The hypothesis was that uGDS would preserve or increase its relative advantage
as storage pressure increased. The hypothesis was not used as a pass criterion;
all valid measurements were retained regardless of the observed trend.

## Scope

The experiment measures retrieval of previously stored KV cache from SSD. It
does not compare cold prefill performance, decode throughput, model quality, or
different software configurations. Torch, vLLM, CUDA, model precision, LMCache
chunk size, and SSD capacity remain fixed across both backends.

## Test System

| Component | Configuration |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, GPU 0 |
| SSD | Samsung 990 PRO, PCI `0000:0f:00.0` |
| Model | Qwen3-0.6B, BF16 |
| Model context limit | 40,960 tokens |
| vLLM | `0.20.1+cu129` |
| PyTorch | `2.11.0+cu129` |
| CUDA toolkit/runtime | 12.9 |
| LMCache | `0.1.dev1881` |
| KV connector | `LMCacheMPConnector` |
| LMCache CUDA extension | Native `lmcache.c_ops`, compiled for `sm_80` |
| LMCache chunk size | 256 tokens |
| SSD L1 capacity | 40 GiB |
| vLLM GPU memory utilization | 0.8 |
| vLLM maximum sequences | 256 |
| Generated tokens per request | 1 |
| NUMA placement | GPU, LMCache, vLLM, and client on NUMA node 0 |
| uGDS hugepages | Eight 2 MiB pages on NUMA node 0 |

GPU 0 had no other compute workload during either run. Before each measured
phase, one 64-token request initialized the model, CUDA graphs, and connector so
startup work was excluded from the samples.

## Ensuring That Hits Come From SSD

### vLLM Automatic Prefix Caching

vLLM Automatic Prefix Caching (APC) was disabled with
`--no-enable-prefix-caching`. With APC enabled, completed requests may leave
reusable KV blocks in vLLM's local GPU block pool. A repeated prompt could then
hit GPU memory directly and bypass LMCache and SSD, invalidating a storage
backend comparison.

vLLM still allocates a GPU KV block pool for active inference. This allocation
does not mean that a completed prompt remains locally cacheable: with APC
disabled, completed blocks return to the free pool and cannot produce a local
prefix-cache hit.

### Hybrid KV Cache Manager

The hybrid KV cache manager was disabled with
`--disable-hybrid-kv-cache-manager`. This setting controls KV block layouts; it
does not control whether KV cache is stored on GPU or SSD.

`LMCacheMPConnector` in vLLM 0.20.1 accepts one block-ID group and does not
advertise hybrid memory allocator support. Qwen3-0.6B uses a single
full-attention KV layout, so disabling the hybrid manager provides the layout
expected by the connector without changing the storage tier being measured.

### Store and Retrieve Barrier

Each phase used the following sequence:

1. Send a unique cold request and let LMCache store its KV cache in the SSD L1
   slab.
2. Wait until LMCache reports the complete expected STORE token count.
3. Send the same prompt as a measured hot request.
4. Accept the sample only if vLLM reports the full prompt as externally cached
   and LMCache reports the corresponding RETRIEVE token count.

The CPU pinned-memory L1 tier was disabled when the GDS L1 tier was active.
Small GPU buffers used to stage I/O were not persistent KV cache. No measured
sample was accepted if it recomputed KV, used a backend fallback, or encountered
a storage read error.

## Capacity Planning

For Qwen3-0.6B, the BF16 KV size per token is:

```text
2 (K and V) x 28 layers x 8 KV heads x 128 head dimension x 2 bytes
= 114,688 bytes
= 112 KiB per token
```

The largest working sets were therefore approximately:

| Workload | KV data size |
|---|---:|
| One 40,704-token request | 4.35 GiB |
| 256 concurrent 1,024-token requests | 28 GiB |

Both backends used a 40 GiB SSD L1 slab. This accommodates the largest working
set with roughly 12 GiB of headroom, preventing LRU eviction from being mistaken
for backend performance.

## Experiment Matrix

### Context Pressure

Concurrency was fixed at 1. The prompt lengths were:

```text
3,840, 8,192, 16,384, 32,768, and 40,704 tokens
```

40,704 is the largest multiple of the 256-token LMCache chunk size below the
model's 40,960-token context limit while leaving room for one output token. It
avoids a partial final chunk that would introduce tail recomputation.

For each prompt length:

1. Generate one deterministic, unique prompt.
2. Send one cold request and wait for the SSD STORE barrier.
3. Send five hot requests, each requiring a complete SSD retrieval.
4. Retain all five raw TTFT measurements and report their median.

The primary metric is:

```text
hot_ttft_p50_ms = median(five hot-request TTFT samples)
TTFT speedup = cuFile GDS hot TTFT p50 / uGDS hot TTFT p50
```

A TTFT speedup greater than 1 means uGDS is faster. Cold-request TTFT is kept
only as diagnostic data and is not included in the reported median.

### Concurrency Pressure

Every request used a unique 1,024-token prompt. The concurrency levels were:

```text
1, 2, 4, 8, 16, 32, 64, 128, and 256
```

Geometric scaling covers low, medium, and saturated load with few points. A
fixed pool of 256 prompts was stored once; concurrency level `N` used the first
`N` prompts from that pool. This avoids consuming additional SSD capacity for
each level.

For each backend:

1. Store all 256 prompts and verify the complete STORE token count.
2. At each concurrency level, launch the selected requests simultaneously.
3. Repeat the level three times.
4. Require a complete SSD retrieval for every request in every round.
5. Retain wall time, per-request TTFT, p95 latency, and throughput for each
   round.

The primary metric is:

```text
throughput_ktok_s = concurrency x 1,024 / wall_time_s / 1,000
median_throughput_ktok_s = median(three round throughputs)
Throughput speedup = uGDS median throughput / cuFile GDS median throughput
```

A throughput speedup greater than 1 means uGDS is faster. Throughput saturation
or decline was not a stopping condition. A concurrency point was excluded only
after a hard failure such as timeout, CUDA out-of-memory, GPU Xid, storage I/O
error, incomplete cache hit, or L1 eviction.

## Backend Fairness

Both backends used the complete Samsung 990 PRO rather than splitting the SSD
between a raw region and a filesystem region.

The uGDS run used the raw SSD through the uGDS driver. After all uGDS phases
completed, every service was stopped, the SSD was rebound to the Linux NVMe
driver, formatted as ext4, and mounted for the cuFile GDS run. The cuFile run
then used the same model, prompts, measurement order, SSD capacity, and validity
checks.

The sequential and concurrent phases used separate LMCache server lifecycles so
their working sets could not compete for slab capacity. Both backend runs were
performed in the same maintenance window to limit environmental drift.

## Validity Criteria

A backend run was valid only if all of the following conditions held:

- GPU 0 had no unrelated compute process.
- Software versions and all fixed benchmark parameters matched.
- LMCache loaded the native `lmcache.c_ops` CUDA extension.
- The SSD L1 tier was active and the CPU pinned-memory L1 tier was disabled.
- uGDS did not use a hugepage fallback, and cuFile did not use a compatibility
  or POSIX fallback.
- vLLM APC was disabled and every measured request reported the complete prompt
  as externally cached.
- LMCache STORE and RETRIEVE token totals matched the expected totals.
- No measured request recomputed KV or reported a backend error.
- No vLLM or LMCache process remained after the run.
- The kernel log contained no GPU Xid, segmentation fault, or storage I/O error.

If a measured sample failed any criterion, the backend phase had to be rerun in
full. Individual favorable samples were not selectively repeated or retained.

## Reproduction

The uGDS run used:

```bash
python tests/v1/gpu_connector/bench_e2e.py \
  --backend ugds \
  --device /dev/ugds_drv0 \
  --l1-size-gb 40 \
  --model /tmp/Qwen3-0.6B \
  --max-model-len 40960 \
  --max-num-seqs 256 \
  --seq-token-counts 3840,8192,16384,32768,40704 \
  --seq-hot-repeats 5 \
  --conc-token-count 1024 \
  --conc-levels 1,2,4,8,16,32,64,128,256 \
  --conc-rounds 3 \
  --run-mode all \
  --ssd-only \
  --output results/high_load/ugds.json
```

The cuFile GDS run changed only the backend-specific storage path and output:

```bash
python tests/v1/gpu_connector/bench_e2e.py \
  --backend cufile \
  --slab-dir /mnt/ugds_test \
  --l1-size-gb 40 \
  --model /tmp/Qwen3-0.6B \
  --max-model-len 40960 \
  --max-num-seqs 256 \
  --seq-token-counts 3840,8192,16384,32768,40704 \
  --seq-hot-repeats 5 \
  --conc-token-count 1024 \
  --conc-levels 1,2,4,8,16,32,64,128,256 \
  --conc-rounds 3 \
  --run-mode all \
  --ssd-only \
  --output results/high_load/gds.json
```

Root privileges, NUMA binding, and backend-specific library paths were supplied
by the host environment and were not part of the benchmark's measurement logic.

## Results

The experiment completed on July 29, 2026. Both backends passed all five context
points and all nine concurrency points, including concurrency 256. Every
measured request reported a complete external-cache hit, and the observed STORE
and RETRIEVE token totals matched their expected values.

### Context Pressure Results

| Prompt tokens | uGDS TTFT p50 | cuFile GDS TTFT p50 | uGDS speedup |
|---:|---:|---:|---:|
| 3,840 | 100.1 ms | 245.4 ms | 2.45x |
| 8,192 | 192.0 ms | 508.7 ms | 2.65x |
| 16,384 | 367.1 ms | 980.6 ms | 2.67x |
| 32,768 | 714.7 ms | 1,915.2 ms | 2.68x |
| 40,704 | 950.4 ms | 2,367.6 ms | 2.49x |

### Concurrency Pressure Results

| Concurrent requests | uGDS throughput | cuFile GDS throughput | uGDS speedup |
|---:|---:|---:|---:|
| 1 | 25.3 kTokens/s | 15.7 kTokens/s | 1.61x |
| 2 | 31.7 kTokens/s | 15.2 kTokens/s | 2.09x |
| 4 | 38.3 kTokens/s | 16.4 kTokens/s | 2.33x |
| 8 | 42.5 kTokens/s | 17.5 kTokens/s | 2.42x |
| 16 | 45.9 kTokens/s | 18.5 kTokens/s | 2.48x |
| 32 | 47.4 kTokens/s | 19.1 kTokens/s | 2.49x |
| 64 | 48.9 kTokens/s | 19.4 kTokens/s | 2.52x |
| 128 | 49.9 kTokens/s | 19.1 kTokens/s | 2.60x |
| 256 | 50.2 kTokens/s | 19.1 kTokens/s | 2.63x |

uGDS reduced hot TTFT by 2.45x to 2.68x across the tested context sizes. Its
relative throughput advantage increased from 1.61x at concurrency 1 to 2.63x
at concurrency 256, supporting the hypothesis that uGDS preserves and increases
its advantage under higher concurrent storage load.

## Artifacts

Raw per-request and per-round samples are retained in:

```text
results/high_load/ugds.json
results/high_load/gds.json
```

Phase-specific LMCache and vLLM logs are retained in:

```text
results/high_load/ugds_logs/
results/high_load/gds_logs/
```
