#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: $0 <ugds|cufile> <path> <binary> <output-dir>" >&2
    exit 2
fi

backend=$1
target_path=$2
binary=$3
output_dir=$4

case "$backend" in
    ugds|cufile) ;;
    *) echo "invalid backend: $backend" >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu=${BREAKDOWN_GPU:-0}
length=${BREAKDOWN_LENGTH:-2G}
warmup_length=${BREAKDOWN_WARMUP_LENGTH:-256M}
repetitions=${BREAKDOWN_REPETITIONS:-5}
cpu_bind=${BREAKDOWN_CPU_BIND:-0-31}
mem_node=${BREAKDOWN_MEM_NODE:-0}
timeout_seconds=${BREAKDOWN_TIMEOUT_SECONDS:-180}
cufile_config=${BREAKDOWN_CUFILE_CONFIG:-$script_dir/cufile_breakdown.json}
cufile_lib_dir=${BREAKDOWN_CUFILE_LIB_DIR:-}
cuda_lib_dir=${BREAKDOWN_CUDA_LIB_DIR:-}
ugds_lib_dir=${BREAKDOWN_UGDS_LIB_DIR:-}

mkdir -p "$output_dir"
export CUDA_VISIBLE_DEVICES="$gpu"
summary="$output_dir/summary.tsv"
printf 'backend\tapi\tthreads\tio_size\trepetition\torder\tlength\tbandwidth_mbps\tavg_us\tp50_us\tp95_us\tp99_us\tthread_cpu_util_pct\tthread_cpu_us_per_io\tprocess_cpu_pct\tuser_seconds\tsystem_seconds\texit_code\n' > "$summary"

library_path=${LD_LIBRARY_PATH:-}
prepend_library_path() {
    local directory=$1
    if [ -n "$directory" ]; then
        library_path="$directory${library_path:+:$library_path}"
    fi
}

prepend_library_path "$cuda_lib_dir"
if [ "$backend" = cufile ]; then
    prepend_library_path "$cufile_lib_dir"
    export CUFILE_ENV_PATH_JSON="$cufile_config"
else
    prepend_library_path "$ugds_lib_dir"
    unset CUFILE_ENV_PATH_JSON || true
    unset UGDS_INTERRUPT_MODE || true
fi
export LD_LIBRARY_PATH="$library_path"

capture_environment() {
    local prefix=$1
    nvidia-smi -i "$gpu" --query-gpu=timestamp,index,name,pci.bus_id,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader > "${prefix}.gpu.txt"
    cat /proc/loadavg > "${prefix}.loadavg.txt"
    cat /sys/devices/system/node/node0/hugepages/hugepages-2048kB/free_hugepages > "${prefix}.hugepages.txt"
}

capture_fast_environment() {
    local prefix=$1
    cat /proc/loadavg > "${prefix}.loadavg.txt"
    cat /sys/devices/system/node/node0/hugepages/hugepages-2048kB/free_hugepages > "${prefix}.hugepages.txt"
}

extract_field() {
    local pattern=$1
    local field=$2
    local file=$3
    awk -v pattern="$pattern" -v field="$field" '$0 ~ pattern {print $field; exit}' "$file"
}

run_sample() {
    local api=$1 threads=$2 io_size=$3 repetition=$4 order=$5 measured_length=$6 tag=$7
    local sample_dir="$output_dir/$tag"
    local mode=read
    [ "$api" = async ] && mode=async-read
    mkdir -p "$sample_dir"
    capture_fast_environment "$sample_dir/before"
    printf '%q ' env CUFILE_LOGFILE_PATH="$sample_dir/cufile.log" numactl --physcpubind="$cpu_bind" --membind="$mem_node" "$binary" -f "$target_path" -l "$measured_length" -s "$io_size" -t "$threads" -i 1 -d "$gpu" -m "$mode" > "$sample_dir/command.txt"
    printf '\n' >> "$sample_dir/command.txt"

    set +e
    CUFILE_LOGFILE_PATH="$sample_dir/cufile.log" /usr/bin/time -v -o "$sample_dir/time.txt" \
        timeout --signal=TERM --kill-after=10s "$timeout_seconds" \
        numactl --physcpubind="$cpu_bind" --membind="$mem_node" \
        "$binary" -f "$target_path" -l "$measured_length" -s "$io_size" \
        -t "$threads" -i 1 -d "$gpu" -m "$mode" \
        > "$sample_dir/stdout.log" 2> "$sample_dir/stderr.log"
    rc=$?
    set -e
    capture_fast_environment "$sample_dir/after"

    if [ "$rc" -ne 0 ]; then
        if [ "$repetition" -ne 0 ]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%s\n' \
                "$backend" "$api" "$threads" "$io_size" "$repetition" "$order" "$measured_length" "$rc" >> "$summary"
        fi
        return "$rc"
    fi

    grep -q 'Total IO operations:' "$sample_dir/stdout.log"
    if rg -q 'IO error|CUDA error|failed:|FAIL:' "$sample_dir/stdout.log" "$sample_dir/stderr.log"; then
        echo "$tag: error marker found" >&2
        return 1
    fi
    bandwidth=$(extract_field '^  Bandwidth:' 2 "$sample_dir/stdout.log")
    avg=$(extract_field '^  Avg latency:' 3 "$sample_dir/stdout.log")
    p50=$(extract_field '^  p50[.]0:' 2 "$sample_dir/stdout.log")
    p95=$(extract_field '^  p95[.]0:' 2 "$sample_dir/stdout.log")
    p99=$(extract_field '^  p99[.]0:' 2 "$sample_dir/stdout.log")
    if [ "$api" = sync ]; then
        cpu_util=$(awk '/^  CPU util:/ {gsub(/%/, "", $3); print $3; exit}' "$sample_dir/stdout.log")
        cpu_per_io=$(awk '/^  CPU util:/ {print $5; exit}' "$sample_dir/stdout.log")
    else
        cpu_util=NA
        cpu_per_io=NA
    fi
    process_cpu=$(awk -F ': ' '/Percent of CPU this job got/ {gsub(/%/, "", $2); print $2}' "$sample_dir/time.txt")
    user_seconds=$(awk -F ': ' '/User time [(]seconds[)]/ {print $2}' "$sample_dir/time.txt")
    system_seconds=$(awk -F ': ' '/System time [(]seconds[)]/ {print $2}' "$sample_dir/time.txt")
    if [ "$repetition" -ne 0 ]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t0\n' \
            "$backend" "$api" "$threads" "$io_size" "$repetition" "$order" "$measured_length" \
            "$bandwidth" "$avg" "$p50" "$p95" "$p99" "$cpu_util" "$cpu_per_io" \
            "$process_cpu" "$user_seconds" "$system_seconds" >> "$summary"
    fi
    return 0
}

cells=(
    sync:1:64K sync:1:1M sync:1:16M
    async:1:64K async:1:1M async:1:16M
    sync:16:64K sync:16:1M sync:16:16M
    async:16:64K async:16:1M async:16:16M
)

capture_environment "$output_dir/campaign_before"
ldd "$binary" > "$output_dir/linked_libraries.txt"
sha256sum "$binary" > "$output_dir/binary.sha256"

# One excluded warmup per cell. Warmups use the same API and concurrency but a
# shorter transfer length and are never appended to summary.tsv.
for cell in "${cells[@]}"; do
    IFS=: read -r api threads io_size <<< "$cell"
    run_sample "$api" "$threads" "$io_size" 0 warmup "$warmup_length" \
        "warmup_${api}_t${threads}_${io_size}" >/dev/null
done

for repetition in $(seq 1 "$repetitions"); do
    capture_environment "$output_dir/round${repetition}_before"
    if [ $((repetition % 2)) -eq 1 ]; then
        indices=($(seq 0 11))
        order=forward
    else
        indices=($(seq 11 -1 0))
        order=reverse
    fi
    for index in "${indices[@]}"; do
        IFS=: read -r api threads io_size <<< "${cells[$index]}"
        tag="r${repetition}_${order}_${api}_t${threads}_${io_size}"
        run_sample "$api" "$threads" "$io_size" "$repetition" "$order" "$length" "$tag"
    done
    capture_environment "$output_dir/round${repetition}_after"
done

capture_environment "$output_dir/campaign_after"
echo "$summary"
