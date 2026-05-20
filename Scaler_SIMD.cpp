/**
 * Scaler_SIMD.cpp  —  SIMD AVX2 Implementation with Double-Buffered I/O
 * -----------------------------------------------------------------------
 *
 * Key optimisation over the naive SIMD version:
 *   DOUBLE BUFFERING — while AVX2 processes block[cur], the OS is already
 *   reading block[1-cur] from disk on a background thread.  This hides
 *   almost all I/O latency behind compute, giving a meaningful wall-clock
 *   win on I/O-bound workloads (the common case for large datasets).
 *
 * Other I/O improvements:
 *   • posix_fadvise(POSIX_FADV_SEQUENTIAL) — tells the kernel to read-ahead
 *   • setvbuf with a large stdio buffer — reduces syscall overhead
 *   • Larger default block_rows (256 000) — amortises read() overhead
 *
 * Compute kernel (unchanged from optimised SIMD):
 *   • Aligned _mm_malloc buffers for shift/scale/accumulators
 *   • _mm256_load_pd / _mm256_store_pd when D is a multiple of 4
 *   • FMA (_mm256_fmadd_pd) for sum-of-squares in phase 1
 *
 * Usage:
 *   ./scaler_simd input.bin output.bin N D mode [block_rows]
 *   mode: standard | minmax
 *
 * Build:
 *   g++ -O3 -march=native -std=c++17 -o scaler_simd Scaler_SIMD.cpp
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
 #include <future>          // std::async, std::future
 #include <immintrin.h>
 
 #ifdef __linux__
 #  include <fcntl.h>       // posix_fadvise
 #endif
 
 /* ------------------------------------------------------------------ */
 /*  Wall-clock timer                                                    */
 /* ------------------------------------------------------------------ */
 static double now_sec() {
     using clk = std::chrono::steady_clock;
     return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
 }
 
 /* ------------------------------------------------------------------ */
 /*  Hint sequential access to the OS (Linux only; no-op elsewhere)     */
 /* ------------------------------------------------------------------ */
 static void hint_sequential(FILE *f, long long total_bytes)
 {
 #ifdef __linux__
     int fd = fileno(f);
     posix_fadvise(fd, 0, total_bytes, POSIX_FADV_SEQUENTIAL);
 #else
     (void)f; (void)total_bytes;
 #endif
 }
 
 /* ------------------------------------------------------------------ */
 /*  Per-column accumulators                                             */
 /* ------------------------------------------------------------------ */
 struct ColStats {
     double sum, sum_sq, min_val, max_val;
     double mean, var, std_dev;
 };
 
 enum class ScalerMode { STANDARD, MINMAX };
 
 /* ------------------------------------------------------------------ */
 /*  Async read helper — reads exactly `elems` doubles from `fin`       */
 /*  into `buf`.  Returns number of elements actually read.             */
 /* ------------------------------------------------------------------ */
 static size_t async_read(FILE *fin, double *buf, size_t elems)
 {
     return std::fread(buf, sizeof(double), elems, fin);
 }
 
 /* ================================================================== */
 /*  PHASE 1 — double-buffered read + AVX2 accumulation                */
 /* ================================================================== */
 static bool phase1_compute_stats(
     const char   *input_path,
     long long     N,
     long long     D,
     long long     block_rows,
     std::vector<ColStats> &stats,
     double       &wall_time,
     double       &compute_time)
 {
     /* ---------- initialise accumulators ---------- */
     for (long long j = 0; j < D; ++j) {
         stats[j].sum     = 0.0;
         stats[j].sum_sq  = 0.0;
         stats[j].min_val =  std::numeric_limits<double>::infinity();
         stats[j].max_val = -std::numeric_limits<double>::infinity();
     }
 
     FILE *fin = std::fopen(input_path, "rb");
     if (!fin) { std::fprintf(stderr, "[ERROR] Cannot open: %s\n", input_path); return false; }
 
     /* Large stdio buffer + sequential hint */
     const size_t STDIO_BUF = 8ULL * 1024 * 1024; // 8 MB
     std::vector<char> stdio_buf(STDIO_BUF);
     std::setvbuf(fin, stdio_buf.data(), _IOFBF, STDIO_BUF);
     hint_sequential(fin, N * D * (long long)sizeof(double));
 
     /* ---------- aligned working buffers ---------- */
     size_t blk_elems = static_cast<size_t>(block_rows * D);
 
     double *buf[2];
     buf[0] = (double*)_mm_malloc(blk_elems * sizeof(double), 32);
     buf[1] = (double*)_mm_malloc(blk_elems * sizeof(double), 32);
 
     double *v_sum    = (double*)_mm_malloc(D * sizeof(double), 32);
     double *v_sum_sq = (double*)_mm_malloc(D * sizeof(double), 32);
     double *v_min    = (double*)_mm_malloc(D * sizeof(double), 32);
     double *v_max    = (double*)_mm_malloc(D * sizeof(double), 32);
 
     if (!buf[0] || !buf[1] || !v_sum || !v_sum_sq || !v_min || !v_max) {
         std::fprintf(stderr, "[ERROR] Allocation failed in phase 1\n");
         std::fclose(fin);
         return false;
     }
 
     for (long long j = 0; j < D; ++j) {
         v_sum[j]    = 0.0;
         v_sum_sq[j] = 0.0;
         v_min[j]    =  std::numeric_limits<double>::infinity();
         v_max[j]    = -std::numeric_limits<double>::infinity();
     }
 
     long long num_vecs  = D / 4;
     bool      aligned_d = (D % 4 == 0);
 
     /* ---------- prime the pump: read block 0 synchronously ---------- */
    long long rows_remaining = N;
    long long rows_this = std::min(block_rows, rows_remaining);
    size_t    elems_this = static_cast<size_t>(rows_this * D);
 
     double t_wall_start = now_sec();
     compute_time = 0.0;
 
     if (std::fread(buf[0], sizeof(double), elems_this, fin) != elems_this) {
         std::fprintf(stderr, "[ERROR] Phase 1: initial read failed\n");
         std::fclose(fin);
         return false;
     }
     rows_remaining -= rows_this;
     int cur = 0;
 
     /* ---------- double-buffered loop ---------- */
     while (true) {
         /* --- kick off async read of NEXT block while we compute --- */
         long long rows_next  = std::min(block_rows, rows_remaining);
         size_t    elems_next = static_cast<size_t>(rows_next * D);
 
         std::future<size_t> io_future;
         if (rows_next > 0) {
             double *next_buf = buf[1 - cur];
             io_future = std::async(std::launch::async,
                                    async_read, fin, next_buf, elems_next);
         }
 
         /* --- AVX2 accumulation on current block --- */
         double t_c0 = now_sec();
         const long long rows_cur = rows_this;
 
         for (long long i = 0; i < rows_cur; ++i) {
             const double *row = buf[cur] + i * D;
 
             for (long long v = 0; v < num_vecs; ++v) {
                 __m256d x  = aligned_d ? _mm256_load_pd (&row[v*4])
                                        : _mm256_loadu_pd(&row[v*4]);
 
                 __m256d s  = _mm256_load_pd(&v_sum[v*4]);
                 s          = _mm256_add_pd(s, x);
                 _mm256_store_pd(&v_sum[v*4], s);
 
                 __m256d sq = _mm256_load_pd(&v_sum_sq[v*4]);
                 sq         = _mm256_fmadd_pd(x, x, sq);
                 _mm256_store_pd(&v_sum_sq[v*4], sq);
 
                 __m256d mn = _mm256_load_pd(&v_min[v*4]);
                 mn         = _mm256_min_pd(mn, x);
                 _mm256_store_pd(&v_min[v*4], mn);
 
                 __m256d mx = _mm256_load_pd(&v_max[v*4]);
                 mx         = _mm256_max_pd(mx, x);
                 _mm256_store_pd(&v_max[v*4], mx);
             }
             /* scalar tail */
             for (long long j = num_vecs * 4; j < D; ++j) {
                 double val = row[j];
                 v_sum[j]    += val;
                 v_sum_sq[j] += val * val;
                 if (val < v_min[j]) v_min[j] = val;
                 if (val > v_max[j]) v_max[j] = val;
             }
         }
         compute_time += (now_sec() - t_c0);
 
         /* --- wait for async read to finish --- */
         if (rows_next > 0) {
             size_t got = io_future.get();
             if (got != elems_next) {
                 std::fprintf(stderr, "[ERROR] Phase 1 async read mismatch\n");
                 std::fclose(fin);
                 return false;
             }
             rows_remaining -= rows_next;
             rows_this       = rows_next;
             cur             = 1 - cur;
         } else {
             break;
         }
     }
 
     wall_time = now_sec() - t_wall_start;
 
     std::fclose(fin);
 
     /* --- merge AVX accumulators into stats --- */
     for (long long j = 0; j < D; ++j) {
         stats[j].sum    += v_sum[j];
         stats[j].sum_sq += v_sum_sq[j];
         if (v_min[j] < stats[j].min_val) stats[j].min_val = v_min[j];
         if (v_max[j] > stats[j].max_val) stats[j].max_val = v_max[j];
     }
 
     _mm_free(buf[0]); _mm_free(buf[1]);
     _mm_free(v_sum); _mm_free(v_sum_sq);
     _mm_free(v_min); _mm_free(v_max);
 
     /* --- derive mean / variance / std_dev --- */
     double inv_N = 1.0 / static_cast<double>(N);
     for (long long j = 0; j < D; ++j) {
         stats[j].mean    = stats[j].sum * inv_N;
         stats[j].var     = std::max(0.0,
                                stats[j].sum_sq * inv_N - stats[j].mean * stats[j].mean);
         stats[j].std_dev = std::sqrt(stats[j].var);
     }
     return true;
 }
 
 /* ================================================================== */
 /*  PHASE 2 — double-buffered read + AVX2 scaling + write             */
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
     double       &compute_time)
 {
     FILE *fin  = std::fopen(input_path,  "rb");
     if (!fin)  { std::fprintf(stderr, "[ERROR] Cannot open input: %s\n",  input_path);  return false; }
     FILE *fout = std::fopen(output_path, "wb");
     if (!fout) { std::fprintf(stderr, "[ERROR] Cannot open output: %s\n", output_path); std::fclose(fin); return false; }
 
     /* I/O hints */
     const size_t STDIO_BUF = 8ULL * 1024 * 1024;
     std::vector<char> stdio_in(STDIO_BUF), stdio_out(STDIO_BUF);
     std::setvbuf(fin,  stdio_in.data(),  _IOFBF, STDIO_BUF);
     std::setvbuf(fout, stdio_out.data(), _IOFBF, STDIO_BUF);
     hint_sequential(fin, N * D * (long long)sizeof(double));
 
     /* --- aligned shift / scale arrays --- */
     double *shift_arr = (double*)_mm_malloc(D * sizeof(double), 32);
     double *scale_arr = (double*)_mm_malloc(D * sizeof(double), 32);
 
     for (long long j = 0; j < D; ++j) {
         if (mode == ScalerMode::STANDARD) {
             shift_arr[j] = stats[j].mean;
             scale_arr[j] = (stats[j].std_dev != 0.0) ? 1.0 / stats[j].std_dev : 0.0;
         } else {
             double rng   = stats[j].max_val - stats[j].min_val;
             shift_arr[j] = stats[j].min_val;
             scale_arr[j] = (rng != 0.0) ? 1.0 / rng : 0.0;
         }
     }
 
     size_t blk_elems = static_cast<size_t>(block_rows * D);
     double *buf[2];
     buf[0] = (double*)_mm_malloc(blk_elems * sizeof(double), 32);
     buf[1] = (double*)_mm_malloc(blk_elems * sizeof(double), 32);
 
     long long num_vecs  = D / 4;
     bool      aligned_d = (D % 4 == 0);
 
     /* --- prime the pump --- */
     long long rows_remaining = N;
     long long rows_this = std::min(block_rows, rows_remaining);
     size_t    elems_this = static_cast<size_t>(rows_this * D);
 
     double t_wall_start = now_sec();
     compute_time = 0.0;
 
     if (std::fread(buf[0], sizeof(double), elems_this, fin) != elems_this) {
         std::fprintf(stderr, "[ERROR] Phase 2: initial read failed\n");
         std::fclose(fin); std::fclose(fout); return false;
     }
     rows_remaining -= rows_this;
     int cur = 0;
 
     /* --- double-buffered loop --- */
     while (true) {
         long long rows_next  = std::min(block_rows, rows_remaining);
         size_t    elems_next = static_cast<size_t>(rows_next * D);
 
         std::future<size_t> io_future;
         if (rows_next > 0) {
             double *next_buf = buf[1 - cur];
             io_future = std::async(std::launch::async,
                                    async_read, fin, next_buf, elems_next);
         }
 
         /* --- AVX2 scaling in-place on current block --- */
         double t_c0 = now_sec();
 
         for (long long i = 0; i < rows_this; ++i) {
             double *row = buf[cur] + i * D;
 
             for (long long v = 0; v < num_vecs; ++v) {
                 __m256d vx     = aligned_d ? _mm256_load_pd (&row[v*4])
                                            : _mm256_loadu_pd(&row[v*4]);
                 __m256d vshift = _mm256_load_pd(&shift_arr[v*4]);
                 __m256d vscale = _mm256_load_pd(&scale_arr[v*4]);
                 __m256d vres   = _mm256_mul_pd(_mm256_sub_pd(vx, vshift), vscale);
 
                 if (aligned_d) _mm256_store_pd (&row[v*4], vres);
                 else           _mm256_storeu_pd(&row[v*4], vres);
             }
             for (long long j = num_vecs * 4; j < D; ++j)
                 row[j] = (row[j] - shift_arr[j]) * scale_arr[j];
         }
         compute_time += (now_sec() - t_c0);
 
        /* write current block: compute actual elements for this block */
        size_t elems_cur = static_cast<size_t>(rows_this * D);
        if (std::fwrite(buf[cur], sizeof(double), elems_cur, fout) != elems_cur) {
            std::fprintf(stderr, "[ERROR] Phase 2: write failed\n");
            std::fclose(fin); std::fclose(fout); return false;
        }
 
         /* --- wait for next read --- */
         if (rows_next > 0) {
             size_t got = io_future.get();
             if (got != elems_next) {
                 std::fprintf(stderr, "[ERROR] Phase 2 async read mismatch\n");
                 std::fclose(fin); std::fclose(fout); return false;
             }
             rows_remaining -= rows_next;
             rows_this       = rows_next;
             cur             = 1 - cur;
         } else {
             break;
         }
     }
 
     wall_time = now_sec() - t_wall_start;
 
     _mm_free(buf[0]); _mm_free(buf[1]);
     _mm_free(shift_arr); _mm_free(scale_arr);
     std::fclose(fin);
     std::fclose(fout);
     return true;
 }
 
 /* ------------------------------------------------------------------ */
 /*  Print stats summary                                                 */
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
 /*  main                                                                */
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
     long long   block_rows  = (argc == 7) ? std::atoll(argv[6]) : 256000LL;  // larger default
 
     ScalerMode mode;
     if      (mode_str == "standard") mode = ScalerMode::STANDARD;
     else if (mode_str == "minmax")   mode = ScalerMode::MINMAX;
     else {
         std::fprintf(stderr, "[ERROR] Unknown mode '%s'\n", mode_str.c_str());
         return EXIT_FAILURE;
     }
 
     double file_size_gb = static_cast<double>(N) * D * sizeof(double) / (1 << 30);
     double block_mb     = static_cast<double>(block_rows) * D * sizeof(double) / (1 << 20);
 
     std::printf("=== SIMD Scaler — AVX2 + Double-Buffered I/O ===\n");
     std::printf("  Input:      %s\n",   input_path);
     std::printf("  Output:     %s\n",   output_path);
     std::printf("  N (rows):   %lld\n", N);
     std::printf("  D (cols):   %lld\n", D);
     std::printf("  Mode:       %s\n",   mode_str.c_str());
     std::printf("  Block rows: %lld  (≈ %.1f MB per block)\n", block_rows, block_mb);
     std::printf("  File size:  ≈ %.3f GB\n\n", file_size_gb);
 
     std::vector<ColStats> stats(static_cast<size_t>(D));
 
     /* ---- Phase 1 ---- */
     std::printf("[Phase 1] Computing per-column statistics (AVX2 + async I/O)...\n");
     double wall1 = 0.0, compute1 = 0.0;
     if (!phase1_compute_stats(input_path, N, D, block_rows, stats, wall1, compute1))
         return EXIT_FAILURE;
     std::printf("[Phase 1] Done in %.3f s  (compute %.3f s, I/O overlap %.3f s)\n",
                 wall1, compute1, wall1 - compute1);
     print_stats(stats, D);
 
     /* ---- Phase 2 ---- */
     std::printf("[Phase 2] Applying %s and writing output (AVX2 + async I/O)...\n",
                 mode_str.c_str());
     double wall2 = 0.0, compute2 = 0.0;
     if (!phase2_scale_and_write(input_path, output_path, N, D, block_rows, mode, stats, wall2, compute2))
         return EXIT_FAILURE;
     std::printf("[Phase 2] Done in %.3f s  (compute %.3f s, I/O overlap %.3f s)\n",
                 wall2, compute2, wall2 - compute2);
 
     /* ---- Summary ---- */
     double total = wall1 + wall2;
     std::printf("\n=== Timing Summary ===\n");
     std::printf("  Phase 1 Total Wall Time : %.3f s  (Compute: %.3f s, I/O: %.3f s)\n",
                 wall1, compute1, wall1 - compute1);
     std::printf("  Phase 2 Total Wall Time : %.3f s  (Compute: %.3f s, I/O: %.3f s)\n",
                 wall2, compute2, wall2 - compute2);
     std::printf("  -----------------------------------------------\n");
     std::printf("  Total Compute Time Only : %.3f s\n", compute1 + compute2);
     std::printf("  Total Execution Time    : %.3f s\n", total);
     std::printf("  Throughput (3× file)    : %.2f GB/s\n",
                 3.0 * file_size_gb / total);
 
     return EXIT_SUCCESS;
 }