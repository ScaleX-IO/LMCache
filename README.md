# LMCache uGDS Backend

This fork adds a [uGDS](https://github.com/ScaleX-IO/uGDS) storage backend
to the GDS L1 tier of [LMCache](https://github.com/LMCache/LMCache).

uGDS is a user-space GPUDirect Storage library: the NVMe IO path runs
entirely in user space (no kernel driver, no ioctl per IO), and the SSD
DMAs data directly to/from GPU memory. Compared to NVIDIA cuFile GDS,
uGDS delivers up to **2x lower TTFT** on cache-hit requests and **2x
higher throughput** under concurrent load in vLLM end-to-end benchmarks.

---

## Architecture

```
vLLM (scheduler / worker)
    │
    ├─ LMCacheMPConnector          (official vLLM KV connector)
    │      │
    │      └─ LMCache adapter      (ZMQ control + CUDA IPC)
    │              │
    ▼              ▼
vLLM paged KV   LMCache MP server
                       │  gather/scatter kernel
                       ▼
                 GPU staging buffer
                       │
                       ├─ cuFileAsync  → ext4 slab file   (GDS)
                       └─ uGDSAsync    → raw block device  (uGDS)
```

uGDS replaces the cuFile async IO path with a user-space NVMe stack.
Set `--gds-l1-backend ugds` and point `--gds-l1-path` at the raw device
(e.g. `/dev/ugds_drv0`). The rest of LMCache (chunking, hashing, slab
allocator, eviction) is unchanged.

## Performance

vLLM end-to-end on NVIDIA A100-SXM4-40GB + Samsung 990 PRO (PCIe Gen4 x4),
cross-root-port P2P. Qwen3-0.6B, 256-token LMCache chunks, GDS L1 = 4 GiB,
vLLM APC disabled, `max_tokens=1` (pure TTFT measurement).

- **Sequential**: each prompt uses unique token IDs to guarantee a cold miss
  on first request. After the STORE completes, the same prompt is sent 5
  more times (cache hit); report TTFT p50.
- **Concurrent**: each concurrency level (1/2/4/8) uses separate 1024-token
  prompts, pre-warmed with a cold request. All requests then fire
  concurrently for 3 rounds; report median throughput.

![Sequential TTFT](assets/lmcache_e2e_seq_ttft.png)

![Concurrent Throughput](assets/lmcache_e2e_conc_throughput.png)

![Speedup](assets/lmcache_e2e_speedup.png)

## Usage

After environment setup (see below), the only difference from the default
cuFile GDS backend is two CLI flags and `LD_LIBRARY_PATH`:

```bash
# cuFile GDS (default)
lmcache server \
  --gds-l1-backend cufile \
  --gds-l1-path /mnt/nvme \
  ...

# uGDS
export LD_LIBRARY_PATH=/path/to/uGDS/build:$LD_LIBRARY_PATH
lmcache server \
  --gds-l1-backend ugds \
  --gds-l1-path /dev/ugds_drv0 \
  ...
```

All other server flags (`--chunk-size`, `--l1-size-gb`, `--eviction-policy`,
etc.) and the vLLM side (`--kv-transfer-config`) remain the same. No code
changes are needed in the application or vLLM launch command.

## Environment Setup

Requirements: NVIDIA GPU with CUDA, an NVMe SSD dedicated to uGDS, the
[uGDS](https://github.com/ScaleX-IO/uGDS) library built (`libugds.so`)
and its kernel module (`ugds_drv.ko`).

```bash
# Bind the NVMe SSD to the uGDS driver
cd /path/to/uGDS
scripts/env_switch.sh ugds 0000:b8:00.0
ls /dev/ugds_drv*

# Make libugds.so visible to the loader
export LD_LIBRARY_PATH=/path/to/uGDS/build:$LD_LIBRARY_PATH
```

To switch the SSD back to the kernel driver (for cuFile/GDS):

```bash
scripts/env_switch.sh gds 0000:b8:00.0
sudo mount -o data=ordered /dev/nvme0n1 /mnt/ugds_test
```

## Running the Benchmarks

### IO-level

```bash
# uGDS (SSD bound to ugds_drv)
python tests/v1/gpu_connector/bench_ugds_vs_gds.py --backend ugds
python tests/v1/gpu_connector/bench_chunk_read.py ugds
python tests/v1/gpu_connector/bench_chunk_read.py ugds-context

# cuFile GDS (SSD on kernel nvme driver, mounted)
python tests/v1/gpu_connector/bench_ugds_vs_gds.py --backend gds \
    --gds-file /mnt/ugds_test/bench_slab.bin
python tests/v1/gpu_connector/bench_chunk_read.py gds
python tests/v1/gpu_connector/bench_chunk_read.py gds-context
```

### vLLM end-to-end

The E2E benchmark starts an LMCache server and vLLM, runs sequential
cold/hot and concurrent cache-hit tests, and saves results to JSON.

```bash
# uGDS backend
export LD_LIBRARY_PATH=/path/to/uGDS/build
python tests/v1/gpu_connector/bench_e2e.py --backend ugds --device /dev/ugds_drv0

# cuFile GDS backend
python tests/v1/gpu_connector/bench_e2e.py --backend cufile --slab-dir /mnt/ugds_test
```

## Tests

```bash
# uGDS async backend unit + hardware roundtrip
pytest tests/v1/gpu_connector/test_ugds_async.py --noconftest -v
pytest tests/v1/gpu_connector/test_gds_context.py -v

# cuFile roundtrip (needs GDS-capable mount point)
LMCACHE_GDS_TEST_DIR=/mnt/ugds_test \
    pytest tests/v1/gpu_connector/test_gds_context.py -v
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
