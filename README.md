# LMCache uGDS Backend

This fork adds a **uGDS** backend to the GDS L1 tier of
[LMCache](https://github.com/LMCache/LMCache). uGDS is a user-space
GPUDirect Storage library: the NVMe IO path runs entirely in user space
(no kernel NVMe driver, no ioctl per IO), and the SSD DMAs data directly
to/from GPU memory. Set `backend: "ugds"` in the GDS L1 config and point
`file_location` at the raw device (e.g. `/dev/ugds_drv0`).

## Environment setup

Requirements: NVIDIA GPU with CUDA, an NVMe SSD dedicated to uGDS, the
[uGDS](https://github.com/ScaleX-IO/uGDS) library built (`libugds.so`) and
its kernel module (`ugds_drv.ko`).

```bash
# Bind the NVMe SSD to the uGDS driver (replace the PCI address with yours)
cd /path/to/uGDS
scripts/env_switch.sh ugds 0000:b8:00.0
ls /dev/ugds_drv*          # device node index depends on bind order

# Make libugds.so visible to the loader
export LD_LIBRARY_PATH=/path/to/uGDS/build:$LD_LIBRARY_PATH
```

To switch the SSD back to the kernel driver (for cuFile/GDS or regular
file IO):

```bash
scripts/env_switch.sh gds 0000:b8:00.0
sudo mount -o data=ordered /dev/nvme0n1 /mnt/ugds_test
```

## Running the tests

```bash
# uGDS backend unit + hardware roundtrip tests (skipped without hardware)
pytest tests/v1/gpu_connector/test_ugds_async.py --noconftest -v
pytest tests/v1/gpu_connector/test_gds_context.py -v

# cuFile roundtrip tests need a GDS-capable filesystem named explicitly
# (tmp_path may live on LVM, which nvidia-fs cannot register)
LMCACHE_GDS_TEST_DIR=/mnt/ugds_test pytest tests/v1/gpu_connector/test_gds_context.py -v
```

## Benchmarks

`tests/v1/gpu_connector/bench_ugds_vs_gds.py` sweeps IO size (4K to 1M) and
pipeline depth (1 to 64) through the async backend interface:

```bash
# Phase 1: uGDS (SSD bound to ugds_drv)
python tests/v1/gpu_connector/bench_ugds_vs_gds.py --backend ugds

# Phase 2: GDS/cuFile (switch the driver and mount first, see above)
python tests/v1/gpu_connector/bench_ugds_vs_gds.py --backend gds \
    --gds-file /mnt/ugds_test/bench_slab.bin
```

Results are saved to `bench_results_{backend}.json`.

`tests/v1/gpu_connector/bench_chunk_read.py` measures reads at the realistic
32 MB chunk size, either at the raw backend level or through the full
LMCache `GDSContext` path:

```bash
python tests/v1/gpu_connector/bench_chunk_read.py ugds          # raw uGDS
python tests/v1/gpu_connector/bench_chunk_read.py ugds-context  # via GDSContext
python tests/v1/gpu_connector/bench_chunk_read.py gds           # raw cuFile async
python tests/v1/gpu_connector/bench_chunk_read.py gds-sync      # cuFile sync
python tests/v1/gpu_connector/bench_chunk_read.py gds-context   # via GDSContext
```

Iteration count and pipeline depths are tunable via `LMCACHE_BENCH_ITERS`,
`LMCACHE_BENCH_WARMUP`, and `LMCACHE_BENCH_DEPTHS`.
