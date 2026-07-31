# Performance Breakdown Report of uGDS and cuFile GDS

## Abstract

This report pinpoints the performance differences between uGDS and cuFile GDS in the LMCache SSD KV restoration path through four sets of interlocking controlled experiments. The experiments sequentially cover raw asynchronous reads, production-form I/O transactions, request-level breakdown, and a counterfactual matrix of sync/async modes and stream counts.

The results indicate that the performance discrepancy first emerges in the single-stream, large-block asynchronous read layer: in 16 MiB reads, the median bandwidths for uGDS and cuFile are 5.494 GiB/s and 2.204 GiB/s, respectively, making the former 2.49 x that of the latter. This difference is maintained at 2.42 x after passing through the LMCache GDSContext, which corresponds to a 2.80 x difference in request-level SSD-to-temp latency. Conversely, the uGDS/cuFile bandwidth ratio for 16 MiB single-thread synchronous reads is 0.960, with a 90% bootstrap confidence interval of `[0.939, 1.005]`, aligning with the hypothesis of performance parity. Therefore, the available evidence supports the conclusion that the advantage of uGDS stems from the single-stream async serving path currently utilized by LMCache, rather than a generalized difference in PCIe DMA or sequential SSD bandwidth.

## Research Questions and Experimental Scope

The experiments address the following questions:

1. Whether the performance difference already exists outside of LMCache and vLLM;
    
2. Whether GDSContext, production fragmentation, and request-level processing create or amplify the difference;
    
3. Whether the difference is determined by the synchronous read ceiling, asynchronous APIs, or stream parallelism;
    
4. Which conclusions can be extrapolated to LMCache, and which are strictly applicable to the current workload.
    

All formal results were obtained from the same NVIDIA A 100-SXM 4-40 GB, Samsung 990 PRO, and NUMA 0 node. The cuFile version is 1.14.1.1, validated with `compatibility=false` and no POSIX fallback. The raw asynchronous, production transaction, and request-level experiments were subjected to multiple rounds of repetition; each configuration in the sync/async matrix underwent five formal measurement rounds. Bandwidth comparisons utilize the median across rounds, and key ratio reporting is based on a 90% bootstrap confidence interval derived from round resampling.

## Cross-Layer Results

The table below presents the core data required to close the attribution chain. Bandwidth ratios are calculated as uGDS/cuFile; latency ratios are cuFile/uGDS, ensuring that all values greater than 1 represent a relative advantage for uGDS.

|Measurement Level|Control Condition|uGDS|cuFile GDS|Relative Advantage|
|---|---|---|---|---|
|Raw I/O|16 MiB async read|5.494 GiB/s|2.204 GiB/s|2.49 x|
|Production I/O|112 MiB transaction, GDSContext, depth 64|5.543 GiB/s|2.287 GiB/s|2.42 x|
|Request Path|SSD-to-temp p 50, concurrency 256|19.98 ms|56.04 ms|2.80 x|
|Request Path|KV scatter p 50, concurrency 256|0.189 ms|0.193 ms|1.02 x|
|Counterfactual|16 MiB, single-thread sync|5,677 MiB/s|5,916 MiB/s|0.96 x|
|Counterfactual|16 MiB, single-stream async|5,617 MiB/s|2,412 MiB/s|2.33 x|

The data above forms the following chain of evidence:

```
Raw async I/O already exhibits discrepancy
  -> Production-form transaction retains the discrepancy
  -> Request-level differences are concentrated in SSD-to-temp
  -> Lower service rate accumulates into queue wait at high concurrency
```

GDSContext does not significantly alter the backend ranking. At a depth of 64, the context/raw bandwidth ratio for uGDS is $5.543 / 5.457 = 1.016$, while the corresponding ratio for cuFile is $2.287 / 2.316 = 0.987$. The encapsulation overhead changes for both are much smaller than the underlying differences between the backends, indicating that slab lookup, offset calculation, registration area boundary splitting, and submission lifecycle primarily act as conduits for the underlying service rate.

## Asynchronous Submission Path Analysis

LMCache's 1,024-token KV transaction is 112 MiB, consisting of four 28 MiB chunks; each chunk is split into two asynchronous reads of 16 MiB and 12 MiB at the registration area boundary. The submission time proportions at depth 64 are as follows:

|Path|Submit/Wall|Median Bandwidth|
|---|---|---|
|uGDS raw|0.28%|5.457 GiB/s|
|uGDS GDSContext|0.97%|5.543 GiB/s|
|cuFile raw|99.62%|2.316 GiB/s|
|cuFile GDSContext|99.60%|2.287 GiB/s|

The uGDS asynchronous interface submits I/O callbacks to the CUDA stream; the callback subsequently executes the same userspace NVMe I/O as the synchronous path and polls for completion. The calling thread's submission time proportion is low, but this design is not without cost: in a production transaction, the uGDS process consumes approximately two CPU core-equivalents, reflecting a trade-off where low submission latency and a high service rate are achieved at the expense of callback and busy-poll CPU overhead.

In contrast, cuFile's submit/wall ratio approaches 100% as depth increases, while bandwidth remains around 2.3 GiB/s. This phenomenon can be strictly defined as "observable backpressure exists in the current single-stream async submission path." Because cuFile's internal implementation is closed-source, the existing data cannot differentiate between specific causes such as lock contention, internal queues, extent mapping, command splitting, or completion handling; therefore, no further inferences are made regarding its internal mechanisms.

## Request-Level Attribution

At a concurrency of 256, each backend processes 1,280 complete SSD-hit requests. The p 50 results for the request-level breakdown are as follows:

|Stage|uGDS|cuFile GDS|Explanation|
|---|---|---|---|
|SSD-to-temp|19.98 ms|56.04 ms|Primary source of service time difference|
|KV scatter|0.189 ms|0.193 ms|Backends are fundamentally identical|
|Connector|1.413 ms|1.417 ms|Backends are fundamentally identical|
|vLLM residual|15.88 ms|16.77 ms|Difference is minor|
|MP queue|1,319 ms|5,827 ms|Queuing outcome of service rate differences|

MP queue wait and SSD-to-temp do not constitute two mutually independent sources of acceleration. cuFile requests occupy the serving path for a longer duration, resulting in longer queues for subsequent requests. There may also be temporal overlap between stages, so the p 50 values of each stage cannot be directly summed to form an end-to-end latency.

## Sync/Async and Stream Count Counterfactuals

The counterfactual matrix is used to decouple the device bandwidth ceiling from API/stream topology effects. The key results for 16 MiB are:

|I/O Profile|uGDS|cuFile GDS|uGDS/cuFile|
|---|---|---|---|
|1 thread sync|5,677 MiB/s|5,916 MiB/s|0.960|
|1 stream async|5,617 MiB/s|2,412 MiB/s|2.328|
|16 streams async|5,618 MiB/s|4,487 MiB/s|1.252|

Under single-thread conditions, uGDS async retains 98.9% of its sync bandwidth, whereas cuFile retains only 40.8%. The uGDS/cuFile ratio for single-stream async is 2.328, with a 90% bootstrap confidence interval of `[2.271, 2.379]`. Expanding cuFile to 16 streams yields a 1.86 x scaling, indicating that parallel streams can recover most of its asynchronous throughput; uGDS, on the other hand, already plateaus under a single stream.

This matrix also establishes applicability boundaries: at 1 MiB with 16 streams async, the two are nearly identical (uGDS/cuFile is 0.993); at 64 KiB with 16 streams async, the uGDS/cuFile ratio is 0.613. The latter demonstrates that the uGDS host callback design experiences scheduling bottlenecks under small I/O and multi-stream conditions.

## Validity and Limitations

The experiments constrain internal validity through the following checks: matching physical offsets, read-back data verification, full external-cache hits, same-process native-GDS logging for cuFile, NVFS byte closure, hugepage reclamation, and the absence of CUDA/storage errors. All reported figures have been uniformly recalculated from immutable raw archives.

The results remain bound by the following constraints:

- uGDS utilizes raw devices, while cuFile uses ext 4; they share the SSD and topology, but their file system semantics differ.
    
- Some campaigns did not employ a strict AB/BA sequence within every matrix cell, leaving the potential for residual timing perturbations.
    
- The current conclusions cover only the read path and do not apply to writes or mixed read/write scenarios.
    
- Results following changes to SSDs, GPUs, models, KV layouts, or software versions require independent verification.
    
- Small-I/O, multi-stream, and mixed-hit workloads should not be extrapolated from the large-block, single-stream results in this report.
    

## Conclusion

Under LMCache's current workload profile of a single load-stream and 12/16 MiB async reads, uGDS sustains a service rate of approximately 5.5 GiB/s with low submission thread utilization. The equivalent path in cuFile exhibits submission backpressure, resulting in a service rate of about 2.3 GiB/s. This discrepancy forms at the raw I/O layer, is largely preserved through GDSContext, and manifests in the request path as shorter SSD-to-temp times and correspondingly lower derived queue latencies.

The performance parity in synchronous large-block reads, combined with the significant performance recovery of cuFile across multiple streams, collectively rule out the overgeneralization that "uGDS's PCIe DMA is inherently ~2.5 x faster." Accurate conclusions must be strictly scoped to the current async API and stream topology.