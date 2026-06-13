/*
 * Scaler_CUDA.cu
 *
 * CUDA version for the project. Uses double buffering (with pinned memory and streams) 
 * to hide the disk I/O behind the GPU computation.
 *
 * Usage:
 * ./scaler_cuda input.bin output.bin N D mode [block_rows]
 * mode: standard | minmax
 * block_rows: optional, default = 256000
 *
 * Build for krylov100 (V100 GPU is sm_70):
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
/* Macro to easily check for CUDA errors                              */
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
/* Simple wall-clock timer                                            */
/* ------------------------------------------------------------------ */
static double now_sec() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

/* ------------------------------------------------------------------ */
/* Tell Linux we're reading sequentially to speed up disk access      */
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
/* Struct for our per-column stats                                    */
/* ------------------------------------------------------------------ */
struct ColStats {
    double sum, sum_sq, min_val, max_val;
    double mean, var, std_dev;
};

enum class ScalerMode { STANDARD, MINMAX };

/* ------------------------------------------------------------------ */
/* Custom atomics. Older CUDA versions don't have built-in double     */
/* precision min/max, so we have to use atomicCAS                     */
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
/* GPU Kernels                                                        */
/* ================================================================== */

__global__ void phase1_kernel(const double* __restrict__ data,
                              long long elems,
                              long long D,
                              double* __restrict__ g_sum,
                              double* __restrict__ g_sum_sq,
                              double* __restrict__ g_min,
                              double* __restrict__ g_max)
{
    extern __shared__ double s_data[];
    double* s_sum    = s_data;
    double* s_sum_sq = s_sum    + D;
    double* s_min    = s_sum_sq + D;
    double* s_max    = s_min    + D;

    for (int i = threadIdx.x; i < (int)D; i += blockDim.x) {
        s_sum[i]    = 0.0;
        s_sum_sq[i] = 0.0;
        s_min[i]    =  INFINITY;
        s_max[i]    = -INFINITY;
    }
    __syncthreads();

    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < elems;
         idx += (long long)gridDim.x * blockDim.x)
    {
        double val = data[idx];
        int    col = (int)(idx % D);

        atomicAdd      (&s_sum[col],    val);
        atomicAdd      (&s_sum_sq[col], val * val);
        atomicMinDouble(&s_min[col],    val);
        atomicMaxDouble(&s_max[col],    val);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < (int)D; i += blockDim.x) {
        atomicAdd      (&g_sum[i],    s_sum[i]);
        atomicAdd      (&g_sum_sq[i], s_sum_sq[i]);
        atomicMinDouble(&g_min[i],    s_min[i]);
        atomicMaxDouble(&g_max[i],    s_max[i]);
    }
}

__global__ void phase2_kernel(double* __restrict__ data,
                              long long elems,
                              long long D,
                              const double* __restrict__ shift,
                              const double* __restrict__ scale)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems) {
        int col = (int)(idx % D);
        data[idx] = (data[idx] - shift[col]) * scale[col];
    }
}

/* ================================================================== */
/* Phase 1: Read the file in chunks and calculate stats on the GPU    */
/* ================================================================== */
static bool phase1_compute_stats(
    const char            *input_path,
    long long              N,
    long long              D,
    long long              block_rows,
    std::vector<ColStats> &stats,
    double                &wall_time,
    double                &io_time,
    size_t                 shared_mem_size,
    double                &gpu_h2d_time,
    double                &gpu_ker_time)
{
    FILE *fin = std::fopen(input_path, "rb");
    if (!fin) { std::fprintf(stderr, "[ERROR] Cannot open: %s\n", input_path); return false; }

    const size_t STDIO_BUF = 8ULL * 1024 * 1024;
    std::vector<char> stdio_buf(STDIO_BUF);
    std::setvbuf(fin, stdio_buf.data(), _IOFBF, STDIO_BUF);
    hint_sequential(fin, N * D * (long long)sizeof(double));

    size_t blk_bytes = static_cast<size_t>(block_rows * D * sizeof(double));

    double *h_buf[2], *d_buf[2];
    CUDA_CHECK(cudaMallocHost(&h_buf[0], blk_bytes));
    CUDA_CHECK(cudaMallocHost(&h_buf[1], blk_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf[0], blk_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf[1], blk_bytes));

    double *d_sum, *d_sum_sq, *d_min, *d_max;
    size_t stats_bytes = (size_t)D * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_sum,    stats_bytes));
    CUDA_CHECK(cudaMalloc(&d_sum_sq, stats_bytes));
    CUDA_CHECK(cudaMalloc(&d_min,    stats_bytes));
    CUDA_CHECK(cudaMalloc(&d_max,    stats_bytes));

    std::vector<double> h_init_sum(D, 0.0), h_init_min(D, INFINITY), h_init_max(D, -INFINITY);
    CUDA_CHECK(cudaMemcpy(d_sum,    h_init_sum.data(), stats_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sum_sq, h_init_sum.data(), stats_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_min,    h_init_min.data(), stats_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max,    h_init_max.data(), stats_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream[2];
    CUDA_CHECK(cudaStreamCreate(&stream[0]));
    CUDA_CHECK(cudaStreamCreate(&stream[1]));

    /* ---- Need separate events for each chunk because of streams ---- */
    long long total_chunks = (N + block_rows - 1) / block_rows;
    std::vector<cudaEvent_t> ev_h2d_s(total_chunks), ev_h2d_e(total_chunks);
    std::vector<cudaEvent_t> ev_ker_s(total_chunks), ev_ker_e(total_chunks);
    for (int i = 0; i < total_chunks; ++i) {
        CUDA_CHECK(cudaEventCreate(&ev_h2d_s[i])); CUDA_CHECK(cudaEventCreate(&ev_h2d_e[i]));
        CUDA_CHECK(cudaEventCreate(&ev_ker_s[i])); CUDA_CHECK(cudaEventCreate(&ev_ker_e[i]));
    }
    int chunk_idx = 0;

    const int threads = 256;
    long long rows_remaining = N;
    long long rows_this  = std::min(block_rows, rows_remaining);
    long long elems_this = rows_this * D;

    double t_wall_start = now_sec();
    io_time = 0.0;

    /* ---- Read the first chunk so the pipeline can start ---- */
    double t0 = now_sec();
    if (std::fread(h_buf[0], sizeof(double), (size_t)elems_this, fin) != static_cast<size_t>(elems_this)) {
        return false;
    }
    io_time += now_sec() - t0;

    CUDA_CHECK(cudaEventRecord(ev_h2d_s[chunk_idx], stream[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_buf[0], h_buf[0], (size_t)elems_this * sizeof(double), cudaMemcpyHostToDevice, stream[0]));
    CUDA_CHECK(cudaEventRecord(ev_h2d_e[chunk_idx], stream[0]));

    int blocks = (int)std::min(1024LL, (elems_this + threads - 1) / threads);
    CUDA_CHECK(cudaEventRecord(ev_ker_s[chunk_idx], stream[0]));
    phase1_kernel<<<blocks, threads, shared_mem_size, stream[0]>>>(d_buf[0], elems_this, D, d_sum, d_sum_sq, d_min, d_max);
    CUDA_CHECK(cudaEventRecord(ev_ker_e[chunk_idx], stream[0]));

    chunk_idx++;
    rows_remaining -= rows_this;
    int cur = 0;

    /* ---- Loop through the rest of the file using both buffers ---- */
    while (rows_remaining > 0) {
        int       next       = 1 - cur;
        long long rows_next  = std::min(block_rows, rows_remaining);
        long long elems_next = rows_next * D;

        CUDA_CHECK(cudaStreamSynchronize(stream[next]));

        t0 = now_sec();
        if (std::fread(h_buf[next], sizeof(double), (size_t)elems_next, fin) != static_cast<size_t>(elems_next)) {
            return false;
        }
        io_time += now_sec() - t0;

        CUDA_CHECK(cudaEventRecord(ev_h2d_s[chunk_idx], stream[next]));
        CUDA_CHECK(cudaMemcpyAsync(d_buf[next], h_buf[next], (size_t)elems_next * sizeof(double), cudaMemcpyHostToDevice, stream[next]));
        CUDA_CHECK(cudaEventRecord(ev_h2d_e[chunk_idx], stream[next]));

        blocks = (int)std::min(1024LL, (elems_next + threads - 1) / threads);
        CUDA_CHECK(cudaEventRecord(ev_ker_s[chunk_idx], stream[next]));
        phase1_kernel<<<blocks, threads, shared_mem_size, stream[next]>>>(d_buf[next], elems_next, D, d_sum, d_sum_sq, d_min, d_max);
        CUDA_CHECK(cudaEventRecord(ev_ker_e[chunk_idx], stream[next]));

        chunk_idx++;
        rows_remaining -= rows_next;
        cur = next;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    wall_time = now_sec() - t_wall_start;
    std::fclose(fin);

    /* ---- Grab the actual GPU timings from the events ---- */
    gpu_h2d_time = 0.0;
    gpu_ker_time = 0.0;
    for (int i = 0; i < chunk_idx; ++i) {
        float ms1 = 0, ms2 = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms1, ev_h2d_s[i], ev_h2d_e[i]));
        CUDA_CHECK(cudaEventElapsedTime(&ms2, ev_ker_s[i], ev_ker_e[i]));
        gpu_h2d_time += ms1 / 1000.0;
        gpu_ker_time += ms2 / 1000.0;
        
        CUDA_CHECK(cudaEventDestroy(ev_h2d_s[i])); CUDA_CHECK(cudaEventDestroy(ev_h2d_e[i]));
        CUDA_CHECK(cudaEventDestroy(ev_ker_s[i])); CUDA_CHECK(cudaEventDestroy(ev_ker_e[i]));
    }

    /* ---- Pull the final sums/mins/maxes back to the host ---- */
    std::vector<double> h_sum(D), h_sum_sq(D), h_min(D), h_max(D);
    CUDA_CHECK(cudaMemcpy(h_sum.data(),    d_sum,    stats_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_sq.data(), d_sum_sq, stats_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min.data(),    d_min,    stats_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max.data(),    d_max,    stats_bytes, cudaMemcpyDeviceToHost));

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

    /* ---- Free the buffers ---- */
    CUDA_CHECK(cudaFreeHost(h_buf[0]));  CUDA_CHECK(cudaFreeHost(h_buf[1]));
    CUDA_CHECK(cudaFree(d_buf[0]));      CUDA_CHECK(cudaFree(d_buf[1]));
    CUDA_CHECK(cudaFree(d_sum));         CUDA_CHECK(cudaFree(d_sum_sq));
    CUDA_CHECK(cudaFree(d_min));         CUDA_CHECK(cudaFree(d_max));
    CUDA_CHECK(cudaStreamDestroy(stream[0])); CUDA_CHECK(cudaStreamDestroy(stream[1]));

    return true;
}

/* ================================================================== */
/* Phase 2: Read, apply scaling on GPU, then write out to new file    */
/* ================================================================== */
static bool phase2_scale_and_write(
    const char                  *input_path,
    const char                  *output_path,
    long long                    N,
    long long                    D,
    long long                    block_rows,
    ScalerMode                   mode,
    const std::vector<ColStats> &stats,
    double                      &wall_time,
    double                      &io_time,
    double                      &gpu_h2d_time,
    double                      &gpu_ker_time,
    double                      &gpu_d2h_time)
{
    FILE *fin  = std::fopen(input_path,  "rb");
    if (!fin)  { return false; }
    FILE *fout = std::fopen(output_path, "wb");
    if (!fout) { std::fclose(fin); return false; }

    const size_t STDIO_BUF = 8ULL * 1024 * 1024;
    std::vector<char> stdio_in(STDIO_BUF), stdio_out(STDIO_BUF);
    std::setvbuf(fin,  stdio_in.data(),  _IOFBF, STDIO_BUF);
    std::setvbuf(fout, stdio_out.data(), _IOFBF, STDIO_BUF);
    hint_sequential(fin, N * D * (long long)sizeof(double));

    std::vector<double> h_shift(D), h_scale(D);
    for (long long j = 0; j < D; ++j) {
        if (mode == ScalerMode::STANDARD) {
            h_shift[j] = stats[j].mean;
            h_scale[j] = (stats[j].std_dev != 0.0) ? 1.0 / stats[j].std_dev : 0.0;
        } else {
            double rng  = stats[j].max_val - stats[j].min_val;
            h_shift[j]  = stats[j].min_val;
            h_scale[j]  = (rng != 0.0) ? 1.0 / rng : 0.0;
        }
    }

    double *d_shift, *d_scale;
    size_t stats_bytes = (size_t)D * sizeof(double);
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

    /* ---- Event setup for Phase 2 ---- */
    long long total_chunks = (N + block_rows - 1) / block_rows;
    std::vector<cudaEvent_t> ev_h2d_s(total_chunks), ev_h2d_e(total_chunks);
    std::vector<cudaEvent_t> ev_ker_s(total_chunks), ev_ker_e(total_chunks);
    std::vector<cudaEvent_t> ev_d2h_s(total_chunks), ev_d2h_e(total_chunks);
    for (int i = 0; i < total_chunks; ++i) {
        CUDA_CHECK(cudaEventCreate(&ev_h2d_s[i])); CUDA_CHECK(cudaEventCreate(&ev_h2d_e[i]));
        CUDA_CHECK(cudaEventCreate(&ev_ker_s[i])); CUDA_CHECK(cudaEventCreate(&ev_ker_e[i]));
        CUDA_CHECK(cudaEventCreate(&ev_d2h_s[i])); CUDA_CHECK(cudaEventCreate(&ev_d2h_e[i]));
    }
    int chunk_idx = 0;

    const int threads = 256;
    long long rows_remaining = N;
    long long rows_this  = std::min(block_rows, rows_remaining);
    long long elems_this = rows_this * D;

    double t_wall_start = now_sec();
    io_time = 0.0;

    /* ---- Start the pipeline with the first read ---- */
    double t0 = now_sec();
    if (std::fread(h_buf[0], sizeof(double), (size_t)elems_this, fin) != static_cast<size_t>(elems_this)) {
        return false;
    }
    io_time += now_sec() - t0;

    CUDA_CHECK(cudaEventRecord(ev_h2d_s[chunk_idx], stream[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_buf[0], h_buf[0], (size_t)elems_this * sizeof(double), cudaMemcpyHostToDevice, stream[0]));
    CUDA_CHECK(cudaEventRecord(ev_h2d_e[chunk_idx], stream[0]));

    int blocks = (int)((elems_this + threads - 1) / threads);
    CUDA_CHECK(cudaEventRecord(ev_ker_s[chunk_idx], stream[0]));
    phase2_kernel<<<blocks, threads, 0, stream[0]>>>(d_buf[0], elems_this, D, d_shift, d_scale);
    CUDA_CHECK(cudaEventRecord(ev_ker_e[chunk_idx], stream[0]));

    CUDA_CHECK(cudaEventRecord(ev_d2h_s[chunk_idx], stream[0]));
    CUDA_CHECK(cudaMemcpyAsync(h_buf[0], d_buf[0], (size_t)elems_this * sizeof(double), cudaMemcpyDeviceToHost, stream[0]));
    CUDA_CHECK(cudaEventRecord(ev_d2h_e[chunk_idx], stream[0]));

    chunk_idx++;
    int cur = 0;
    rows_remaining -= rows_this;
    long long rows_prev = rows_this;

    /* ---- Double buffering loop for Phase 2 ---- */
    while (rows_remaining > 0) {
        int       next       = 1 - cur;
        long long rows_next  = std::min(block_rows, rows_remaining);
        long long elems_next = rows_next * D;

        CUDA_CHECK(cudaStreamSynchronize(stream[next]));

        t0 = now_sec();
        if (std::fread(h_buf[next], sizeof(double), (size_t)elems_next, fin) != static_cast<size_t>(elems_next)) {
            return false;
        }
        io_time += now_sec() - t0;

        CUDA_CHECK(cudaStreamSynchronize(stream[cur]));

        CUDA_CHECK(cudaEventRecord(ev_h2d_s[chunk_idx], stream[next]));
        CUDA_CHECK(cudaMemcpyAsync(d_buf[next], h_buf[next], (size_t)elems_next * sizeof(double), cudaMemcpyHostToDevice, stream[next]));
        CUDA_CHECK(cudaEventRecord(ev_h2d_e[chunk_idx], stream[next]));

        blocks = (int)((elems_next + threads - 1) / threads);
        CUDA_CHECK(cudaEventRecord(ev_ker_s[chunk_idx], stream[next]));
        phase2_kernel<<<blocks, threads, 0, stream[next]>>>(d_buf[next], elems_next, D, d_shift, d_scale);
        CUDA_CHECK(cudaEventRecord(ev_ker_e[chunk_idx], stream[next]));

        CUDA_CHECK(cudaEventRecord(ev_d2h_s[chunk_idx], stream[next]));
        CUDA_CHECK(cudaMemcpyAsync(h_buf[next], d_buf[next], (size_t)elems_next * sizeof(double), cudaMemcpyDeviceToHost, stream[next]));
        CUDA_CHECK(cudaEventRecord(ev_d2h_e[chunk_idx], stream[next]));

        chunk_idx++;

        t0 = now_sec();
        if (std::fwrite(h_buf[cur], sizeof(double), (size_t)(rows_prev * D), fout) != static_cast<size_t>(rows_prev * D)) {
            return false;
        }
        io_time += now_sec() - t0;

        cur = next;
        rows_prev      = rows_next;
        rows_remaining -= rows_next;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream[cur]));
    t0 = now_sec();
    if (std::fwrite(h_buf[cur], sizeof(double), (size_t)(rows_prev * D), fout) != static_cast<size_t>(rows_prev * D)) {
        return false;
    }
    io_time += now_sec() - t0;

    CUDA_CHECK(cudaDeviceSynchronize());
    wall_time = now_sec() - t_wall_start;

    /* ---- Calculate exact GPU times again ---- */
    gpu_h2d_time = 0.0;
    gpu_ker_time = 0.0;
    gpu_d2h_time = 0.0;
    for (int i = 0; i < chunk_idx; ++i) {
        float ms1 = 0, ms2 = 0, ms3 = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms1, ev_h2d_s[i], ev_h2d_e[i]));
        CUDA_CHECK(cudaEventElapsedTime(&ms2, ev_ker_s[i], ev_ker_e[i]));
        CUDA_CHECK(cudaEventElapsedTime(&ms3, ev_d2h_s[i], ev_d2h_e[i]));
        gpu_h2d_time += ms1 / 1000.0;
        gpu_ker_time += ms2 / 1000.0;
        gpu_d2h_time += ms3 / 1000.0;

        CUDA_CHECK(cudaEventDestroy(ev_h2d_s[i])); CUDA_CHECK(cudaEventDestroy(ev_h2d_e[i]));
        CUDA_CHECK(cudaEventDestroy(ev_ker_s[i])); CUDA_CHECK(cudaEventDestroy(ev_ker_e[i]));
        CUDA_CHECK(cudaEventDestroy(ev_d2h_s[i])); CUDA_CHECK(cudaEventDestroy(ev_d2h_e[i]));
    }

    /* ---- Free the rest of the memory ---- */
    CUDA_CHECK(cudaFreeHost(h_buf[0])); CUDA_CHECK(cudaFreeHost(h_buf[1]));
    CUDA_CHECK(cudaFree(d_buf[0]));     CUDA_CHECK(cudaFree(d_buf[1]));
    CUDA_CHECK(cudaFree(d_shift));      CUDA_CHECK(cudaFree(d_scale));
    CUDA_CHECK(cudaStreamDestroy(stream[0])); CUDA_CHECK(cudaStreamDestroy(stream[1]));
    std::fclose(fin);
    std::fclose(fout);

    return true;
}

/* ------------------------------------------------------------------ */
/* Print out some sample stats to verify things work                  */
/* ------------------------------------------------------------------ */
static void print_stats(const std::vector<ColStats> &stats, long long D, long long max_cols = 5) {
    std::printf("\n  Per-column statistics (first %lld of %lld columns shown):\n", std::min(max_cols, D), D);
    std::printf("  %-6s  %-14s  %-14s  %-14s  %-14s  %-14s\n", "col","mean","std_dev","min","max","variance");
    std::printf("  %s\n", std::string(76,'-').c_str());
    for (long long j = 0; j < std::min(max_cols, D); ++j)
        std::printf("  %-6lld  %-14.6f  %-14.6f  %-14.6f  %-14.6f  %-14.6f\n",
                    j, stats[j].mean, stats[j].std_dev, stats[j].min_val, stats[j].max_val, stats[j].var);
    if (D > max_cols) std::printf("  ... (%lld more columns)\n", D - max_cols);
    std::printf("\n");
}

/* ------------------------------------------------------------------ */
/* Main execution                                                     */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[]) {
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

    double file_size_gb = static_cast<double>(N) * D * sizeof(double) / (1ULL << 30);
    double block_mb     = static_cast<double>(block_rows) * D * sizeof(double) / (1ULL << 20);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::fprintf(stderr, "[ERROR] No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t shared_mem_size = 4ULL * (size_t)D * sizeof(double);
    if (shared_mem_size > prop.sharedMemPerBlock) {
        std::fprintf(stderr,
            "[ERROR] D=%lld requires %zu bytes of shared memory per block,\n"
            "        but the device supports only %zu bytes.\n",
            D, shared_mem_size, prop.sharedMemPerBlock);
        return EXIT_FAILURE;
    }

    std::printf("=== CUDA Scaler — GPU Accelerated + Double-Buffered I/O ===\n");
    std::printf("  GPU:        %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("  Input:      %s\n",   input_path);
    std::printf("  Output:     %s\n",   output_path);
    std::printf("  N (rows):   %lld\n", N);
    std::printf("  D (cols):   %lld\n", D);
    std::printf("  Mode:       %s\n",   mode_str.c_str());
    std::printf("  Block rows: %lld  (≈ %.1f MB per block)\n", block_rows, block_mb);
    std::printf("  File size:  ≈ %.3f GB\n", file_size_gb);
    std::printf("  Shared mem: %zu B / block (limit: %zu B)\n\n", shared_mem_size, prop.sharedMemPerBlock);

    std::vector<ColStats> stats(static_cast<size_t>(D));

    /* ---- Phase 1 ---- */
    std::printf("[Phase 1] Computing per-column statistics (CUDA + async I/O)...\n");
    double wall1 = 0.0, io1 = 0.0, p1_h2d = 0.0, p1_ker = 0.0;
    if (!phase1_compute_stats(input_path, N, D, block_rows, stats, wall1, io1, shared_mem_size, p1_h2d, p1_ker))
        return EXIT_FAILURE;
    std::printf("[Phase 1] Done in %.3f s  (Wall compute/overlap: %.3f s, pure I/O: %.3f s)\n", wall1, wall1 - io1, io1);
    print_stats(stats, D);

    /* ---- Phase 2 ---- */
    std::printf("[Phase 2] Applying %s and writing output (CUDA + async I/O)...\n", mode_str.c_str());
    double wall2 = 0.0, io2 = 0.0, p2_h2d = 0.0, p2_ker = 0.0, p2_d2h = 0.0;
    if (!phase2_scale_and_write(input_path, output_path, N, D, block_rows, mode, stats, wall2, io2, p2_h2d, p2_ker, p2_d2h))
        return EXIT_FAILURE;
    std::printf("[Phase 2] Done in %.3f s  (Wall compute/overlap: %.3f s, pure I/O: %.3f s)\n", wall2, wall2 - io2, io2);

    /* ---- Summary ---- */
    double total    = wall1 + wall2;
    double total_io = io1   + io2;
    std::printf("\n=== Timing Summary ===\n");
    std::printf("  Phase 1 Total Wall Time : %.3f s\n", wall1);
    std::printf("  Phase 2 Total Wall Time : %.3f s\n", wall2);
    std::printf("  -----------------------------------------------\n");
    std::printf("  Total Execution Time    : %.3f s\n", total);
    std::printf("  Throughput (3x file)    : %.2f GB/s\n\n", 3.0 * file_size_gb / total);

    std::printf("=== GPU Detailed Timings ===\n");
    std::printf("  Phase 1 Kernel (Stats)  : %.3f s\n", p1_ker);
    std::printf("  Phase 1 H2D Transfer    : %.3f s\n", p1_h2d);
    std::printf("  Phase 2 Kernel (Scale)  : %.3f s\n", p2_ker);
    std::printf("  Phase 2 H2D Transfer    : %.3f s\n", p2_h2d);
    std::printf("  Phase 2 D2H Transfer    : %.3f s\n", p2_d2h);
    std::printf("  -----------------------------------------------\n");
    std::printf("  Total Kernel Execution  : %.3f s\n", p1_ker + p2_ker);
    std::printf("  Total Memory Transfers  : %.3f s\n", p1_h2d + p2_h2d + p2_d2h);

    return EXIT_SUCCESS;
}