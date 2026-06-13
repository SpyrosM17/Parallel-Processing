/**
 * Scaler_SIMD.cpp
 *
 * SIMD version using AVX2. Also added double buffering for I/O with std::async 
 * because disk reads were completely bottlenecking the compute.
 *
 * Run it like this:
 * ./scaler_simd input.bin output.bin N D mode [block_rows]
 * mode: standard | minmax
 *
 * Build:
 * g++ -O3 -march=native -std=c++17 -o scaler_simd Scaler_SIMD.cpp
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

 /* AVX2 requires -mavx2. Just wrapping everything so it compiles everywhere */
 #ifdef __AVX2__
 #  include <immintrin.h>
 #endif

 #ifdef __linux__
 #  include <fcntl.h>       // posix_fadvise
 #endif

 /* ------------------------------------------------------------------ */
 /* Basic timer                                                         */
 /* ------------------------------------------------------------------ */
 static double now_sec() {
     using clk = std::chrono::steady_clock;
     return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
 }

 /* ------------------------------------------------------------------ */
 /* Tell OS to read ahead since we go linearly. Helps Linux a bit.      */
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
 /* Custom aligned malloc for AVX2 (needs 32-byte alignment)            */
 /* ------------------------------------------------------------------ */
 static inline double* simd_malloc(size_t n_doubles)
 {
 #ifdef __AVX2__
     return (double*)_mm_malloc(n_doubles * sizeof(double), 32);
 #else
     void *ptr = nullptr;
     if (posix_memalign(&ptr, 32, n_doubles * sizeof(double)) != 0) return nullptr;
     return (double*)ptr;
 #endif
 }

 static inline void simd_free(void *ptr)
 {
 #ifdef __AVX2__
     _mm_free(ptr);
 #else
     free(ptr);
 #endif
 }

 /* ------------------------------------------------------------------ */
 /* Stats struct                                                        */
 /* ------------------------------------------------------------------ */
 struct ColStats {
     double sum, sum_sq, min_val, max_val;
     double mean, var, std_dev;
 };

 enum class ScalerMode { STANDARD, MINMAX };

 /* ------------------------------------------------------------------ */
 /* Helper to read data in the background                               */
 /* ------------------------------------------------------------------ */
 static size_t async_read(FILE *fin, double *buf, size_t elems)
 {
     return std::fread(buf, sizeof(double), elems, fin);
 }

 /* ================================================================== */
 /* Phase 1: Read with async buffering and do math with AVX2          */
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
     /* ---------- Init to zero/infinity ---------- */
     for (long long j = 0; j < D; ++j) {
         stats[j].sum     = 0.0;
         stats[j].sum_sq  = 0.0;
         stats[j].min_val =  std::numeric_limits<double>::infinity();
         stats[j].max_val = -std::numeric_limits<double>::infinity();
     }

     FILE *fin = std::fopen(input_path, "rb");
     if (!fin) { std::fprintf(stderr, "[ERROR] Cannot open: %s\n", input_path); return false; }

     /* Use a big 8MB buffer for stdio to avoid too many system calls */
     const size_t STDIO_BUF = 8ULL * 1024 * 1024; // 8 MB
     std::vector<char> stdio_buf(STDIO_BUF);
     std::setvbuf(fin, stdio_buf.data(), _IOFBF, STDIO_BUF);
     hint_sequential(fin, N * D * (long long)sizeof(double));

     /* ---------- Prep the aligned memory ---------- */
     size_t blk_elems = static_cast<size_t>(block_rows * D);

     double *buf[2];
     buf[0] = simd_malloc(blk_elems);
     buf[1] = simd_malloc(blk_elems);

     double *v_sum    = simd_malloc(D);
     double *v_sum_sq = simd_malloc(D);
     double *v_min    = simd_malloc(D);
     double *v_max    = simd_malloc(D);

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

 #ifdef __AVX2__
     long long num_vecs  = D / 4;
     bool      aligned_d = (D % 4 == 0);
 #endif

     /* ---------- Read the first block before entering the loop ---------- */
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

     /* ---------- Main loop with double buffering ---------- */
     while (true) {
         /* --- Start reading the NEXT block while we work on this one --- */
         long long rows_next  = std::min(block_rows, rows_remaining);
         size_t    elems_next = static_cast<size_t>(rows_next * D);

         std::future<size_t> io_future;
         if (rows_next > 0) {
             double *next_buf = buf[1 - cur];
             io_future = std::async(std::launch::async,
                                    async_read, fin, next_buf, elems_next);
         }

         /* --- Do the AVX2 math on the current block --- */
         double t_c0 = now_sec();
         const long long rows_cur = rows_this;

         for (long long i = 0; i < rows_cur; ++i) {
             const double *row = buf[cur] + i * D;

 #ifdef __AVX2__
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
             /* Handle the leftover columns if D isn't a multiple of 4 */
             for (long long j = num_vecs * 4; j < D; ++j) {
                 double val = row[j];
                 v_sum[j]    += val;
                 v_sum_sq[j] += val * val;
                 if (val < v_min[j]) v_min[j] = val;
                 if (val > v_max[j]) v_max[j] = val;
             }
 #else
             /* Fallback if AVX2 isn't supported (like on my Mac) */
             for (long long j = 0; j < D; ++j) {
                 double val = row[j];
                 v_sum[j]    += val;
                 v_sum_sq[j] += val * val;
                 if (val < v_min[j]) v_min[j] = val;
                 if (val > v_max[j]) v_max[j] = val;
             }
 #endif
         }
         compute_time += (now_sec() - t_c0);

         /* --- Sync up: wait for the background read to finish --- */
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

     /* --- Copy results from our aligned arrays to the stats struct --- */
     for (long long j = 0; j < D; ++j) {
         stats[j].sum    += v_sum[j];
         stats[j].sum_sq += v_sum_sq[j];
         if (v_min[j] < stats[j].min_val) stats[j].min_val = v_min[j];
         if (v_max[j] > stats[j].max_val) stats[j].max_val = v_max[j];
     }

     simd_free(buf[0]); simd_free(buf[1]);
     simd_free(v_sum);  simd_free(v_sum_sq);
     simd_free(v_min);  simd_free(v_max);

     /* --- Calculate variance and standard dev --- */
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
 /* Phase 2: read, scale with AVX2, and write out to the new file     */
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

     /* Same I/O tweaks as Phase 1 */
     const size_t STDIO_BUF = 8ULL * 1024 * 1024;
     std::vector<char> stdio_in(STDIO_BUF), stdio_out(STDIO_BUF);
     std::setvbuf(fin,  stdio_in.data(),  _IOFBF, STDIO_BUF);
     std::setvbuf(fout, stdio_out.data(), _IOFBF, STDIO_BUF);
     hint_sequential(fin, N * D * (long long)sizeof(double));

     /* --- Aligned arrays for shift and scale so we can load them fast --- */
     double *shift_arr = simd_malloc(D);
     double *scale_arr = simd_malloc(D);

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
     buf[0] = simd_malloc(blk_elems);
     buf[1] = simd_malloc(blk_elems);

 #ifdef __AVX2__
     long long num_vecs  = D / 4;
     bool      aligned_d = (D % 4 == 0);
 #endif

     /* --- Read first block --- */
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

     /* --- Main loop --- */
     while (true) {
         long long rows_next  = std::min(block_rows, rows_remaining);
         size_t    elems_next = static_cast<size_t>(rows_next * D);

         std::future<size_t> io_future;
         if (rows_next > 0) {
             double *next_buf = buf[1 - cur];
             io_future = std::async(std::launch::async,
                                    async_read, fin, next_buf, elems_next);
         }

         /* --- Do the scaling on the current chunk --- */
         double t_c0 = now_sec();

         for (long long i = 0; i < rows_this; ++i) {
             double *row = buf[cur] + i * D;

 #ifdef __AVX2__
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
 #else
             /* Fallback if AVX2 isn't supported */
             for (long long j = 0; j < D; ++j)
                 row[j] = (row[j] - shift_arr[j]) * scale_arr[j];
 #endif
         }
         compute_time += (now_sec() - t_c0);

        /* Write current block to the new file */
        size_t elems_cur = static_cast<size_t>(rows_this * D);
        if (std::fwrite(buf[cur], sizeof(double), elems_cur, fout) != elems_cur) {
            std::fprintf(stderr, "[ERROR] Phase 2: write failed\n");
            std::fclose(fin); std::fclose(fout); return false;
        }

         /* --- Wait for the background read to finish --- */
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

     simd_free(buf[0]); simd_free(buf[1]);
     simd_free(shift_arr); simd_free(scale_arr);
     std::fclose(fin);
     std::fclose(fout);
     return true;
 }

 /* ------------------------------------------------------------------ */
 /* Print a small summary table                                         */
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
 /* Main                                                                */
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
     long long   block_rows  = (argc == 7) ? std::atoll(argv[6]) : 256000LL;  // larger default for less IO overhead

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

 #ifdef __AVX2__
     std::printf("=== SIMD Scaler — AVX2 + Double-Buffered I/O ===\n");
 #else
     std::printf("=== SIMD Scaler — Scalar + Double-Buffered I/O (no AVX2) ===\n");
 #endif
     std::printf("  Input:      %s\n",   input_path);
     std::printf("  Output:     %s\n",   output_path);
     std::printf("  N (rows):   %lld\n", N);
     std::printf("  D (cols):   %lld\n", D);
     std::printf("  Mode:       %s\n",   mode_str.c_str());
     std::printf("  Block rows: %lld  (≈ %.1f MB per block)\n", block_rows, block_mb);
     std::printf("  File size:  ≈ %.3f GB\n\n", file_size_gb);

     std::vector<ColStats> stats(static_cast<size_t>(D));

     /* ---- Phase 1 ---- */
 #ifdef __AVX2__
     std::printf("[Phase 1] Computing per-column statistics (AVX2 + async I/O)...\n");
 #else
     std::printf("[Phase 1] Computing per-column statistics (scalar + async I/O)...\n");
 #endif
     double wall1 = 0.0, compute1 = 0.0;
     if (!phase1_compute_stats(input_path, N, D, block_rows, stats, wall1, compute1))
         return EXIT_FAILURE;
     std::printf("[Phase 1] Done in %.3f s  (compute %.3f s, I/O overlap %.3f s)\n",
                 wall1, compute1, wall1 - compute1);
     print_stats(stats, D);

     /* ---- Phase 2 ---- */
 #ifdef __AVX2__
     std::printf("[Phase 2] Applying %s and writing output (AVX2 + async I/O)...\n",
                 mode_str.c_str());
 #else
     std::printf("[Phase 2] Applying %s and writing output (scalar + async I/O)...\n",
                 mode_str.c_str());
 #endif
     double wall2 = 0.0, compute2 = 0.0;
     if (!phase2_scale_and_write(input_path, output_path, N, D, block_rows, mode, stats, wall2, compute2))
         return EXIT_FAILURE;
     std::printf("[Phase 2] Done in %.3f s  (compute %.3f s, I/O overlap %.3f s)\n",
                 wall2, compute2, wall2 - compute2);

     /* ---- Overall results ---- */
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