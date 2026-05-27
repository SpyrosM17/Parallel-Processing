/*
 * Scaler_CUDA.cu  —  CUDA Implementation with Double-Buffered I/O
 * -----------------------------------------------------------------
 *
 * Parallelises per-column statistics computation and block scaling using
 * NVIDIA GPUs. It heavily overlaps Disk I/O with Host-to-Device (H2D),
 * Device-to-Host (D2H) memory transfers, and GPU Kernel execution using
 * CUDA streams and double buffering with Pinned Memory.
 *
 * Usage:
 * ./scaler_cuda input.bin output.bin N D mode [block_rows]
 * mode: standard | minmax
 * block_rows: optional, default = 256000
 *
 * Build (Krylov100 - Tesla V100 is sm_70):
 * module load nvhpc
 * nvcc -O3 -std=c++17 -arch=sm_70 -o scaler_cuda Scaler_CUDA.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <limits>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

#ifdef __linux__
#include <fcntl.h>
#endif

/* ------------------------------------------------------------------ */
/* CUDA Error Checker Macro                                          */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[ERROR] CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* ------------------------------------------------------------------ */
/* Wall-clock timer                                                  */
/* ------------------------------------------------------------------ */
static double now_sec() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

/* ------------------------------------------------------------------ */
/* Hint sequential access to the OS (Linux only; no-op elsewhere)    */
/* ------------------------------------------------------------------ */
static void hint_sequential(FILE *f, long long total_bytes) {
#ifdef __linux__
    int fd = fileno(f);
    posix_fadvise(fd, 0, total_bytes, POSIX_FADV_SEQUENTIAL);
#else
    (void)f; (void)total_bytes;
#endif
}

/* ------------------------------------------------------------------ */
/* Per-column accumulators (Host)                                    */
/* ------------------------------------------------------------------ */
struct ColStats {
    double sum, sum_sq, min_val, max_val;
    double mean, var, std_dev;
};

enum class ScalerMode { STANDARD, MINMAX };

/* ------------------------------------------------------------------ */
/* Device Atomics for double precision MIN and MAX                   */
/* ------------------------------------------------------------------ */
__device__ void atomicMinDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

__device__ void atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

/* ================================================================== */
/* CUDA Kernels                                                      */
/* ================================================================== */

/* Phase 1 Kernel: Shared Memory Reduction for Column Stats */
__global__ void phase1_kernel(const double* __restrict__ data,
                              long long elems,
                              long long D,
                              double* __restrict__ g_sum,
                              double* __restrict__ g_sum_sq,
                              double* __restrict__ g_min,
                              double* __restrict__ g_max)
{
    // Dynamically allocated shared memory for D columns
    extern __shared__ double s_data[];
    double* s_sum    = s_data;
    double* s_sum_sq = s_sum + D;
    double* s_min    = s_sum_sq + D;
    double* s_max    = s_min + D;

    // Initialize shared memory
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        s_sum[i]    = 0.0;
        s_sum_sq[i] = 0.0;
        s_min[i]    = INFINITY;
        s_max[i]    = -INFINITY;
    }
    __syncthreads();

    // 1D Grid striding loop (Perfectly coalesced memory access)
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < elems; 
         idx += gridDim.x * blockDim.x) 
    {
        double val = data[idx];
        int col    = idx % D;

        atomicAdd(&s_sum[col], val);
        atomicAdd(&s_sum_sq[col], val * val);
        atomicMinDouble(&s_min[col], val);
        atomicMaxDouble(&s_max[col], val);
    }
    __syncthreads();

    // Write shared memory results back to global memory
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        atomicAdd(&g_sum[i], s_sum[i]);
        atomicAdd(&g_sum_sq[i], s_sum_sq[i]);
        atomicMinDouble(&g_min[i], s_min[i]);
        atomicMaxDouble(&g_max[i], s_max[i]);
    }
}

/* Phase 2 Kernel: Scale Data */
__global__ void phase2_kernel(double* __restrict__ data,
                              long long elems,
                              long long D,
                              const double* __restrict__ shift,
                              const double* __restrict__ scale)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems) {
        int col = idx % D;
        data[idx] = (data[idx] - shift[col]) * scale[col];
    }
}

/* ================================================================== */
/* PHASE 1 — double-buffered read + GPU statistics                   */
/* ================================================================== */
static bool phase1_compute_stats(
    const char   *input_path,
    long long     N,
    long long     D,
    long long     block_rows,
    std::vector<ColStats> &stats,
    double       &wall_time,
    double       &io_time)
{
    FILE *fin = std::fopen(input_path, "rb");
    if (!fin) { std::fprintf(stderr, "[ERROR] Cannot open: %s\n", input_path); return false; }

    const size_t STDIO_BUF = 8ULL * 1024 * 1024; // 8 MB stdio buffer
    std::vector<char> stdio_buf(STDIO_BUF);
    std::setvbuf(fin, stdio_buf.data(), _IOFBF, STDIO_BUF);
    hint_sequential(fin, N * D * (long long)sizeof(double));

    size_t blk_bytes = static_cast<size_t>(block_rows * D * sizeof(double));

    /* Pinned Host Buffers & Device Buffers */
    double *h_buf[2], *d_buf[2];
    CUDA_CHECK(cudaMallocHost(&h_buf[0], blk_bytes));
    CUDA_CHECK(cudaMallocHost(&h_buf[1], blk_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf[0], blk_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf[1], blk_bytes));

    /* Device Statistics Arrays */
    double *d_sum, *d_sum_sq, *d_min, *d_max;
    size_t stats_bytes = D * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_sum, stats_bytes));
    CUDA_CHECK(cudaMalloc(&d_sum_sq, stats_bytes));
    CUDA_CHECK(cudaMalloc(&d_min, stats_bytes));
    CUDA_CHECK(cudaMalloc(&d_max, stats_bytes));

    std::vector<double> h_init_sum(D, 0.0), h_init_min(D, INFINITY), h_init_max(D, -INFINITY);
    CUDA_CHECK(cudaMemcpy(d_sum, h_init_sum.data(), stats_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sum_sq, h_init_sum.data(), stats_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_min, h_init_min.data(), stats_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max, h_init_max.data(), stats_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream[2];
    CUDA_CHECK(cudaStreamCreate(&stream[0]));
    CUDA_CHECK(cudaStreamCreate(&stream[1]));

    int threads = 256;
    size_t shared_mem_size = 4 * D * sizeof(double);

    long long rows_remaining = N;
    long long rows_this = std::min(block_rows, rows_remaining);
    long long elems_this = rows_this * D;

    double t_wall_start = now_sec();
    io_time = 0.0;

    /* Prime the pump: read block 0 synchronously */
    double t0 = now_sec();
    if (std::fread(h_buf[0], sizeof(double), elems_this, fin) != static_cast<size_t>(elems_this)) {
        std::fprintf(stderr, "[ERROR] Phase 1: initial read failed\n");
        return false;
    }
    io_time += now_sec() - t0;

    CUDA_CHECK(cudaMemcpyAsync(d_buf[0], h_buf[0], elems_this * sizeof(double), cudaMemcpyHostToDevice, stream[0]));
    int blocks = std::min(1024LL, (elems_this + threads - 1) / threads);
    phase1_kernel<<<blocks, threads, shared_mem_size, stream[0]>>>(
        d_buf[0], elems_this, D, d_sum, d_sum_sq, d_min, d_max);

    rows_remaining -= rows_this;
    int cur = 0;

    /* Double-buffered loop */
    while (rows_remaining > 0) {
        int next = 1 - cur;
        long long rows_next  = std::min(block_rows, rows_remaining);
        long long elems_next = rows_next * D;

        // Ensure the stream we're about to feed is idle, so we don't overwrite its pinned memory too early
        CUDA_CHECK(cudaStreamSynchronize(stream[next]));

        // Block CPU to read next chunk
        t0 = now_sec();
        if (std::fread(h_buf[next], sizeof(double), elems_next, fin) != static_cast<size_t>(elems_next)) {
            std::fprintf(stderr, "[ERROR] Phase 1 read failed\n");
            return false;
        }
        io_time += now_sec() - t0;

        // Send to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_buf[next], h_buf[next], elems_next * sizeof(double), cudaMemcpyHostToDevice, stream[next]));
        blocks = std::min(1024LL, (elems_next + threads - 1) / threads);
        phase1_kernel<<<blocks, threads, shared_mem_size, stream[next]>>>(
            d_buf[next], elems_next, D, d_sum, d_sum_sq, d_min, d_max);

        rows_remaining -= rows_next;
        cur = next;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    wall_time = now_sec() - t_wall_start;
    std::fclose(fin);

    /* Merge device accumulators into Host stats */
    std::vector<double> h_sum(D), h_sum_sq(D), h_min(D), h_max(D);
    CUDA_CHECK(cudaMemcpy(h_sum.data(), d_sum, stats_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_sq.data(), d_sum_sq, stats_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min.data(), d_min, stats_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max.data(), d_max, stats_bytes, cudaMemcpyDeviceToHost));

    double inv_N = 1.0 / static_cast<double>(N);
    for (long long j = 0; j < D; ++j) {
        stats[j].sum     = h_sum[j];
        stats[j].sum_sq  = h_sum_sq[j];
        stats[j].min_val = h_min[j];
        stats[j].max_val = h_max[j];
        stats[j].mean    = h_sum[j] * inv_N;
        stats[j].var     = std::max(0.0, h_sum_sq[j] * inv_N - stats[j].mean * stats[j].mean);
        stats[j].std_dev = std::sqrt(stats[j].var);
    }

    /* Cleanup */
    CUDA_CHECK(cudaFreeHost(h_buf[0])); CUDA_CHECK(cudaFreeHost(h_buf[1]));
    CUDA_CHECK(cudaFree(d_buf[0]));     CUDA_CHECK(cudaFree(d_buf[1]));
    CUDA_CHECK(cudaFree(d_sum));        CUDA_CHECK(cudaFree(d_sum_sq));
    CUDA_CHECK(cudaFree(d_min));        CUDA_CHECK(cudaFree(d_max));
    CUDA_CHECK(cudaStreamDestroy(stream[0]));
    CUDA_CHECK(cudaStreamDestroy(stream[1]));

    return true;
}

/* ================================================================== */
/* PHASE 2 — double-buffered read + GPU scaling + write              */
/* ================================================================== */
static bool phase2_scale_and_write(
    const char   *input_path,
    const char   *output_path,
    long long     N,
    long long     D,
    long long     block_rows,
    ScalerMode    mode,
    const std::vector<ColStats> &stats,
    double       &wall_time,
    double       &io_time)
{
    FILE *fin  = std::fopen(input_path,  "rb");
    if (!fin)  { std::fprintf(stderr, "[ERROR] Cannot open input: %s\n",  input_path);  return false; }
    FILE *fout = std::fopen(output_path, "wb");
    if (!fout) { std::fprintf(stderr, "[ERROR] Cannot open output: %s\n", output_path); std::fclose(fin); return false; }

    const size_t STDIO_BUF = 8ULL * 1024 * 1024;
    std::vector<char> stdio_in(STDIO_BUF), stdio_out(STDIO_BUF);
    std::setvbuf(fin,  stdio_in.data(),  _IOFBF, STDIO_BUF);
    std::setvbuf(fout, stdio_out.data(), _IOFBF, STDIO_BUF);
    hint_sequential(fin, N * D * (long long)sizeof(double));

    /* Prepare shift/scale arrays */
    std::vector<double> h_shift(D), h_scale(D);
    for (long long j = 0; j < D; ++j) {
        if (mode == ScalerMode::STANDARD) {
            h_shift[j] = stats[j].mean;
            h_scale[j] = (stats[j].std_dev != 0.0) ? 1.0 / stats[j].std_dev : 0.0;
        } else {
            double rng   = stats[j].max_val - stats[j].min_val;
            h_shift[j] = stats[j].min_val;
            h_scale[j] = (rng != 0.0) ? 1.0 / rng : 0.0;
        }
    }

    double *d_shift, *d_scale;
    size_t stats_bytes = D * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_shift, stats_bytes));
    CUDA_CHECK(cudaMalloc(&d_scale, stats_bytes));
    CUDA_CHECK(cudaMemcpy(d_shift, h_shift.data(), stats_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale, h_scale.data(), stats_bytes, cudaMemcpyHostToDevice));

    size_t blk_bytes = static_cast<size_t>(block_rows * D * sizeof(double));
    double *h_buf[2], *d_buf[2];
    CUDA_CHECK(cudaMallocHost(&h_buf[0], blk_bytes));
    CUDA_CHECK(cudaMallocHost(&h_buf[1], blk_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf[0], blk_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf[1], blk_bytes));

    cudaStream_t stream[2];
    CUDA_CHECK(cudaStreamCreate(&stream[0]));
    CUDA_CHECK(cudaStreamCreate(&stream[1]));
    int threads = 256;

    long long rows_remaining = N;
    long long rows_this = std::min(block_rows, rows_remaining);
    long long elems_this = rows_this * D;

    double t_wall_start = now_sec();
    io_time = 0.0;

    /* Prime the pump */
    double t0 = now_sec();
    if (std::fread(h_buf[0], sizeof(double), elems_this, fin) != static_cast<size_t>(elems_this)) {
        std::fprintf(stderr, "[ERROR] Phase 2: initial read failed\n");
        return false;
    }
    io_time += now_sec() - t0;

    CUDA_CHECK(cudaMemcpyAsync(d_buf[0], h_buf[0], elems_this * sizeof(double), cudaMemcpyHostToDevice, stream[0]));
    long long blocks = (elems_this + threads - 1) / threads;
    phase2_kernel<<<blocks, threads, 0, stream[0]>>>(d_buf[0], elems_this, D, d_shift, d_scale);
    CUDA_CHECK(cudaMemcpyAsync(h_buf[0], d_buf[0], elems_this * sizeof(double), cudaMemcpyDeviceToHost, stream[0]));

    int cur = 0;
    rows_remaining -= rows_this;
    long long rows_prev = rows_this;

    /* Double-buffered loop */
    while (rows_remaining > 0) {
        int next = 1 - cur;
        long long rows_next  = std::min(block_rows, rows_remaining);
        long long elems_next = rows_next * D;

        // Sync stream[next] before CPU overwrites h_buf[next]
        CUDA_CHECK(cudaStreamSynchronize(stream[next]));

        // Read next block
        t0 = now_sec();
        if (std::fread(h_buf[next], sizeof(double), elems_next, fin) != static_cast<size_t>(elems_next)) {
            std::fprintf(stderr, "[ERROR] Phase 2 read mismatch\n");
            return false;
        }
        io_time += now_sec() - t0;

        // Sync stream[cur] so we can write h_buf[cur] to disk
        CUDA_CHECK(cudaStreamSynchronize(stream[cur]));

        // Launch GPU processing for NEXT (runs async to overlap with CPU fwrite!)
        CUDA_CHECK(cudaMemcpyAsync(d_buf[next], h_buf[next], elems_next * sizeof(double), cudaMemcpyHostToDevice, stream[next]));
        blocks = (elems_next + threads - 1) / threads;
        phase2_kernel<<<blocks, threads, 0, stream[next]>>>(d_buf[next], elems_next, D, d_shift, d_scale);
        CUDA_CHECK(cudaMemcpyAsync(h_buf[next], d_buf[next], elems_next * sizeof(double), cudaMemcpyDeviceToHost, stream[next]));

        // Write current block to disk
        t0 = now_sec();
        if (std::fwrite(h_buf[cur], sizeof(double), rows_prev * D, fout) != static_cast<size_t>(rows_prev * D)) {
            std::fprintf(stderr, "[ERROR] Phase 2 write failed\n");
            return false;
        }
        io_time += now_sec() - t0;

        cur = next;
        rows_prev = rows_next;
        rows_remaining -= rows_next;
    }

    /* Flush last block */
    CUDA_CHECK(cudaStreamSynchronize(stream[cur]));
    t0 = now_sec();
    std::fwrite(h_buf[cur], sizeof(double), rows_prev * D, fout);
    io_time += now_sec() - t0;

    wall_time = now_sec() - t_wall_start;

    /* Cleanup */
    CUDA_CHECK(cudaFreeHost(h_buf[0])); CUDA_CHECK(cudaFreeHost(h_buf[1]));
    CUDA_CHECK(cudaFree(d_buf[0]));     CUDA_CHECK(cudaFree(d_buf[1]));
    CUDA_CHECK(cudaFree(d_shift));      CUDA_CHECK(cudaFree(d_scale));
    CUDA_CHECK(cudaStreamDestroy(stream[0]));
    CUDA_CHECK(cudaStreamDestroy(stream[1]));
    std::fclose(fin);
    std::fclose(fout);

    return true;
}

/* ------------------------------------------------------------------ */
/* Print stats summary                                               */
/* ------------------------------------------------------------------ */
static void print_stats(const std::vector<ColStats> &stats, long long D, long long max_cols = 5)
{
    std::printf("\n  Per-column statistics (first %lld of %lld columns shown):\n",
                std::min(max_cols, D), D);
    std::printf("  %-6s  %-14s  %-14s  %-14s  %-14s  %-14s\n",
                "col","mean","std_dev","min","max","variance");
    std::printf("  %s\n", std::string(76,'-').c_str());
    for (long long j = 0; j < std::min(max_cols, D); ++j)
        std::printf("  %-6lld  %-14.6f  %-14.6f  %-14.6f  %-14.6f  %-14.6f\n",
                    j, stats[j].mean, stats[j].std_dev,
                    stats[j].min_val, stats[j].max_val, stats[j].var);
    if (D > max_cols)
        std::printf("  ... (%lld more columns)\n", D - max_cols);
    std::printf("\n");
}

/* ------------------------------------------------------------------ */
/* main                                                              */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[])
{
    if (argc < 6 || argc > 7) {
        std::fprintf(stderr,
            "Usage: %s input.bin output.bin N D mode [block_rows]\n"
            "  mode: standard | minmax\n"
            "  block_rows: optional, default = 256000\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_path  = argv[1];
    const char *output_path = argv[2];
    long long   N           = std::atoll(argv[3]);
    long long   D           = std::atoll(argv[4]);
    std::string mode_str    = argv[5];
    long long   block_rows  = (argc == 7) ? std::atoll(argv[6]) : 256000LL;

    if (N <= 0 || D <= 0) {
        std::fprintf(stderr, "[ERROR] N and D must be positive integers.\n");
        return EXIT_FAILURE;
    }
    if (block_rows <= 0) {
        std::fprintf(stderr, "[ERROR] block_rows must be a positive integer.\n");
        return EXIT_FAILURE;
    }
    
    ScalerMode mode;
    if      (mode_str == "standard") mode = ScalerMode::STANDARD;
    else if (mode_str == "minmax")   mode = ScalerMode::MINMAX;
    else {
        std::fprintf(stderr, "[ERROR] Unknown mode '%s'\n", mode_str.c_str());
        return EXIT_FAILURE;
    }

    double file_size_gb = static_cast<double>(N) * D * sizeof(double) / (1 << 30);
    double block_mb     = static_cast<double>(block_rows) * D * sizeof(double) / (1 << 20);

    /* Check CUDA devices */
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::fprintf(stderr, "[ERROR] No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::printf("=== CUDA Scaler — GPU Accelerated + Double-Buffered I/O ===\n");
    std::printf("  GPU:        %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("  Input:      %s\n",   input_path);
    std::printf("  Output:     %s\n",   output_path);
    std::printf("  N (rows):   %lld\n", N);
    std::printf("  D (cols):   %lld\n", D);
    std::printf("  Mode:       %s\n",   mode_str.c_str());
    std::printf("  Block rows: %lld  (≈ %.1f MB per block)\n", block_rows, block_mb);
    std::printf("  File size:  ≈ %.3f GB\n\n", file_size_gb);

    std::vector<ColStats> stats(static_cast<size_t>(D));

    /* ---- Phase 1 ---- */
    std::printf("[Phase 1] Computing per-column statistics (CUDA + async I/O)...\n");
    double wall1 = 0.0, io1 = 0.0;
    if (!phase1_compute_stats(input_path, N, D, block_rows, stats, wall1, io1))
        return EXIT_FAILURE;
    std::printf("[Phase 1] Done in %.3f s  (GPU compute/overlap: %.3f s, pure I/O: %.3f s)\n",
                 wall1, wall1 - io1, io1);
    print_stats(stats, D);

    /* ---- Phase 2 ---- */
    std::printf("[Phase 2] Applying %s and writing output (CUDA + async I/O)...\n",
                mode_str.c_str());
    double wall2 = 0.0, io2 = 0.0;
    if (!phase2_scale_and_write(input_path, output_path, N, D, block_rows, mode, stats, wall2, io2))
        return EXIT_FAILURE;
    std::printf("[Phase 2] Done in %.3f s  (GPU compute/overlap: %.3f s, pure I/O: %.3f s)\n",
                 wall2, wall2 - io2, io2);

    /* ---- Summary ---- */
    double total = wall1 + wall2;
    double total_io = io1 + io2;
    std::printf("\n=== Timing Summary ===\n");
    std::printf("  Phase 1 Total Wall Time : %.3f s  (Compute/Overlap: %.3f s, I/O: %.3f s)\n",
                wall1, wall1 - io1, io1);
    std::printf("  Phase 2 Total Wall Time : %.3f s  (Compute/Overlap: %.3f s, I/O: %.3f s)\n",
                wall2, wall2 - io2, io2);
    std::printf("  -----------------------------------------------\n");
    std::printf("  Total Compute Time Only : %.3f s\n", total - total_io);
    std::printf("  Total Execution Time    : %.3f s\n", total);
    std::printf("  Throughput (3× file)    : %.2f GB/s\n",
                3.0 * file_size_gb / total);

    return EXIT_SUCCESS;
}