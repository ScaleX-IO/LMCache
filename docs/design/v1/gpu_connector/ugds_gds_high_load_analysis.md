# Performance Analysis Report of uGDS vs. cuFile GDS Under High Load

## Abstract

This report evaluates the end-to-end performance of uGDS and cuFile GDS in LMCache SSD KV cache full-hit scenarios. The experiments increase context length and client concurrency, respectively, to measure the hot TTFT and logical KV retrieval throughput when the storage restoration path is under sustained pressure.

Under the conditions of a single GPU, a single LMCache load stream, 100% full SSD KV hits, and 1-token output, uGDS achieves a 2.45–2.68 x TTFT speedup over cuFile GDS. At a concurrency of 256, the logical KV retrieval throughputs are 50.18 kToken/s and 19.07 kToken/s respectively, with uGDS being 2.63 x that of cuFile. Breakdown experiments indicate that this difference is primarily determined by the large-block single-stream async SSD-to-temp service rate, rather than model execution, KV scatter, or connector overhead.

For details on the underlying mechanisms and counterfactuals, please refer to [`ugds_gds_breakdown_findings. md`](ugds_gds_breakdown_findings. md "null").

## Experiment Goals and Metric Definitions

The high-load experiments are designed to answer two end-to-end questions:

1. How the backend service rate affects hot TTFT as the KV restoration data volume increases with context length;
    
2. Whether the end-to-end throughput converges to the backend's asynchronous read capacity when client concurrency continuously saturates the restoration path.
    

The system is fixed to an NVIDIA A 100-SXM 4-40 GB, Samsung 990 PRO, Qwen 3-0.6 B BF 16, a single vLLM instance, and the LMCache MP connector. The KV data volume per token for the model is 112 KiB; an LMCache chunk is 256 tokens, equating to 28 MiB. Each chunk is split into a 16 MiB and a 12 MiB asynchronous read.

The experiments disable vLLM APC and the CPU KV tier. Every prompt completes a STORE first; a sample is accepted only if vLLM reports a full external-cache hit and the LMCache STORE/RETRIEVE token counts match. The 40 GiB SSD L 1 can accommodate a maximum working set of approximately 28 GiB, and no evictions occurred during the experiments.

The experiments include two types of load pressure:

- **Context Load**: Concurrency is fixed at 1, with prompt lengths of 3,840, 8,192, 16,384, 32,768, and 40,704 tokens; each data point executes 5 hot hits, reporting the median TTFT;
    
- **Concurrency Load**: Prompt length is fixed at 1,024 tokens, preloading 256 distinct prompts, with concurrency scaling from 1 to 256; each data point executes 3 rounds, reporting the median throughput.
    

Each request generates only 1 token to minimize the impact of decoding on the measurements. The numerator for kToken/s in concurrency experiments is the prompt/KV tokens restored from the SSD, not the generated output tokens. Therefore, this document explicitly defines this metric as "logical KV retrieval throughput" and refrains from describing it as generic LLM generation throughput.

## Long Context Results

|Prompt Length|KV Data Volume|uGDS TTFT p 50|cuFile GDS TTFT p 50|TTFT Speedup|
|---|---|---|---|---|
|3,840 tokens|0.410 GiB|100.1 ms|245.4 ms|2.45 x|
|8,192 tokens|0.875 GiB|192.0 ms|508.7 ms|2.65 x|
|16,384 tokens|1.750 GiB|367.1 ms|980.6 ms|2.67 x|
|32,768 tokens|3.500 GiB|714.7 ms|1,915.2 ms|2.68 x|
|40,704 tokens|4.348 GiB|950.4 ms|2,367.6 ms|2.49 x|

Under the 1-token output condition, hot TTFT can be approximately decomposed as:

```
TTFT = KV_bytes / async_backend_service_rate + shared_request_overhead
```

40,704 tokens correspond to 4.348 GiB of KV data. Based on the saturated service rate estimated from the breakdown experiments, the pure read times for uGDS and cuFile are approximately 0.81 s and 2.13 s; after adding the common overheads of lookup, scatter, connector, and vLLM, the magnitudes align with the measured 0.95 s and 2.37 s. Request-level profiling further shows that the SSD-to-temp p 50 is 19.98 ms and 56.04 ms, while the scatter p 50 is 0.189 ms and 0.193 ms, supporting the conclusion that the difference primarily lies in the storage restoration phase.

## Concurrency Throughput Results

|Concurrent Requests|uGDS|cuFile GDS|uGDS / cuFile|
|---|---|---|---|
|1|25.32 kToken/s|15.73 kToken/s|1.61 x|
|2|31.73 kToken/s|15.16 kToken/s|2.09 x|
|4|38.26 kToken/s|16.40 kToken/s|2.33 x|
|8|42.53 kToken/s|17.54 kToken/s|2.42 x|
|16|45.95 kToken/s|18.50 kToken/s|2.48 x|
|32|47.38 kToken/s|19.06 kToken/s|2.49 x|
|64|48.88 kToken/s|19.36 kToken/s|2.52 x|
|128|49.88 kToken/s|19.15 kToken/s|2.60 x|
|256|50.18 kToken/s|19.07 kToken/s|2.63 x|

The logical KV throughput at a concurrency of 256 can be converted to an effective data rate using 112 KiB per token:

```
uGDS       50.18 kToken/s * 112 KiB/token = 5.36 GiB/s
cuFile GDS 19.07 kToken/s * 112 KiB/token = 2.04 GiB/s
```

These conversion results are close to the 5.54 GiB/s and 2.29 GiB/s observed in production-like transactions, indicating that high concurrency amortizes the common overhead, and the end-to-end throughput gradually converges to the backend asynchronous read service rate. The number of concurrent requests does not equal the number of GDS streams: currently, requests for a single instance pass through an affinity worker and a fixed load stream; the primary effect of increasing client concurrency is to continuously supply work to the same service path, rather than creating an equal number of independent storage streams.

At a concurrency of 256, the MP queue wait p 50 s are 1.32 s and 5.83 s, respectively. The queue wait is a derivative result of the difference in service time within the queuing system and should not be multiplied or added to the SSD-to-temp difference to form a second independent acceleration factor.

## Cross-Layer Attribution

|Attribution Question|Control Result|Inference|
|---|---|---|
|Is the difference caused by the model or connector?|raw 16 MiB async: 5.494 vs 2.204 GiB/s|No; a 2.49 x difference already appears outside the model.|
|Do production chunks/splits cause the difference?|112 MiB GDSContext: 5.543 vs 2.287 GiB/s|No; the production path primarily preserves the underlying difference.|
|Which phase of the request differs?|SSD-to-temp: 19.98 vs 56.04 ms; scatter is basically identical|The main difference lies in SSD-to-temp.|
|Is it determined by the synchronous bandwidth limit?|16 MiB sync: 5,677 vs 5,916 MiB/s|No; synchronous baselines perform quite consistently.|
|Why is the difference significant under high load?|16 MiB, single-stream async: 5,617 vs 2,412 MiB/s|High-load continuous pressure specifically targets this I/O pattern.|

The evidence above supports the following causal explanation: raw large-block asynchronous reads already exhibit a service rate difference; the LMCache transaction and request path do not significantly alter this difference; long context increases the service demand per request, and high concurrency reduces backend idle time, ultimately driving the TTFT and logical KV throughput to approach the asynchronous service rate of their respective backends.

## Scope of Result Interpretation

This experiment measures the performance upper bound for retrieval-dominated scenarios, not the general benefits for online inference. The conclusions apply to a single GPU, 100% full SSD KV hits, a single LMCache load stream, 1-token output, and the current Qwen 3-0.6 B KV layout. The following scenarios require independent measurement:

- Mixed hits, CPU/GPU cache tier hits, or limited request arrival rates;
    
- Long decodes and workloads optimizing for output-tokens/s;
    
- Larger models, different KV precisions, different chunk sizes, or multiple LMCache load streams;
    
- Different SSDs, PCIe topologies, GPUs, and software versions;
    
- Write-dominated or mixed read-write workloads.
    

uGDS uses raw devices, while cuFile uses ext 4. Both share the same SSD and PCIe topology but do not possess identical file system semantics or physical LBAs. This difference acts as a limitation on external validity.

cuFile is also not slow under all asynchronous configurations. When 16 MiB reads scale from 1 stream to 16 streams, its bandwidth increases from 2,412 MiB/s to 4,487 MiB/s. The high-load results evaluate the current LMCache single load-stream integration and do not represent the upper performance limit of cuFile if refactored with multiple streams.

## Conclusion

Under the SSD-only, retrieval-dominated high load defined in this report, uGDS achieves a 2.45–2.68 x TTFT speedup and a 2.63 x logical KV retrieval throughput at a concurrency of 256. The performance difference can be explained by the service rate of large-block single-stream async reads: approximately 5.5 GiB/s for uGDS and 2.3 GiB/s for cuFile. The model, connector, and KV scatter do not constitute major sources of the difference; the longer cuFile queue wait is a queuing consequence of its lower service rate under high concurrency.