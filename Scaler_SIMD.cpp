/**
 * Scaler_SIMD.cpp  —  SIMD AVX2 Implementation (Optimized)
 * ------------------------------------------------
 */

 #include <cstdio>
 #include <cstdlib>
 #include <cstring>
 #include <cmath>
 #include <cassert>
 #include <string>
 #include <vector>
 #include <limits>
 #include <chrono>
 #include <algorithm>
 #include <immintrin.h> // Essential for AVX2 intrinsics
 
 /* ------------------------------------------------------------------ */
 /* Tiny helper: wall-clock timer                                      */
 /* ------------------------------------------------------------------ */
 static double now_sec()
 {
     using clk = std::chrono::steady_clock;
     return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
 }
 
 /* ------------------------------------------------------------------ */
 /* Per-column accumulators                                            */
 /* ------------------------------------------------------------------ */
 struct ColStats {
     double sum;      
     double sum_sq;   
     double min_val;
     double max_val;
     double mean;
     double var;
     double std_dev;
 };
 
 enum class ScalerMode { STANDARD, MINMAX };
 
 /* ------------------------------------------------------------------ */
 /* Phase 1 — SIMD AVX2 implementation                                 */
 /* ------------------------------------------------------------------ */
 static bool phase1_compute_stats(
     const char   *input_path,
     long long     N,
     long long     D,
     long long     block_rows,
     std::vector<ColStats> &stats,
     double       &compute_time)
 {
     for (long long j = 0; j < D; ++j) {
         stats[j].sum     = 0.0;
         stats[j].sum_sq  = 0.0;
         stats[j].min_val =  std::numeric_limits<double>::infinity();
         stats[j].max_val = -std::numeric_limits<double>::infinity();
     }
 
     FILE *fin = std::fopen(input_path, "rb");
     if (!fin) {
         std::fprintf(stderr, "[ERROR] Cannot open input file: %s\n", input_path);
         return false;
     }
 
     // Allocate 32-byte aligned memory for AVX2 processing
     size_t elems_per_block = static_cast<size_t>(block_rows * D);
     double* block_data = (double*)_mm_malloc(elems_per_block * sizeof(double), 32);
     if (!block_data) {
         std::fprintf(stderr, "[ERROR] Memory allocation failed.\n");
         std::fclose(fin);
         return false;
     }
     
     long long num_vecs = D / 4;
     
     std::vector<__m256d> v_sum(num_vecs, _mm256_setzero_pd());
     std::vector<__m256d> v_sum_sq(num_vecs, _mm256_setzero_pd());
     std::vector<__m256d> v_min(num_vecs, _mm256_set1_pd(std::numeric_limits<double>::infinity()));
     std::vector<__m256d> v_max(num_vecs, _mm256_set1_pd(-std::numeric_limits<double>::infinity()));
 
     long long rows_remaining = N;
     compute_time = 0.0;
 
     // If D is a multiple of 4, every row will start perfectly aligned to 32 bytes
     bool can_use_aligned_loads = (D % 4 == 0);
 
     while (rows_remaining > 0) {
         long long rows_this_block = std::min(block_rows, rows_remaining);
         size_t    elems           = static_cast<size_t>(rows_this_block * D);
         
         // I/O: Not counted in compute_time
         size_t got = std::fread(block_data, sizeof(double), elems, fin);
         if (got != elems) {
             _mm_free(block_data);
             std::fclose(fin);
             return false;
         }
 
         // --- START PURE COMPUTE ---
         double t_start = now_sec();
 
         for (long long i = 0; i < rows_this_block; ++i) {
             const double *row = block_data + i * D;
             
             for (long long v = 0; v < num_vecs; ++v) {
                 __m256d x;
                 if (can_use_aligned_loads) {
                     x = _mm256_load_pd(&row[v * 4]); // Faster aligned load
                 } else {
                     x = _mm256_loadu_pd(&row[v * 4]); // Fallback
                 }
                 
                 v_sum[v]    = _mm256_add_pd(v_sum[v], x);
                 v_sum_sq[v] = _mm256_fmadd_pd(x, x, v_sum_sq[v]); // FMA
                 v_min[v]    = _mm256_min_pd(v_min[v], x);
                 v_max[v]    = _mm256_max_pd(v_max[v], x);
             }
             
             // Tail processing if D isn't a multiple of 4
             for (long long j = num_vecs * 4; j < D; ++j) {
                 double val = row[j];
                 stats[j].sum += val;
                 stats[j].sum_sq += val * val;
                 if (val < stats[j].min_val) stats[j].min_val = val;
                 if (val > stats[j].max_val) stats[j].max_val = val;
             }
         }
         
         compute_time += (now_sec() - t_start);
         // --- END PURE COMPUTE ---
 
         rows_remaining -= rows_this_block;
     }
     
     std::fclose(fin);
     _mm_free(block_data); // Free aligned memory
 
     // Extract values from AVX registers back to stats array
     double temp_sum[4], temp_sum_sq[4], temp_min[4], temp_max[4];
     for (long long v = 0; v < num_vecs; ++v) {
         _mm256_storeu_pd(temp_sum, v_sum[v]);
         _mm256_storeu_pd(temp_sum_sq, v_sum_sq[v]);
         _mm256_storeu_pd(temp_min, v_min[v]);
         _mm256_storeu_pd(temp_max, v_max[v]);
         
         for (int k = 0; k < 4; ++k) {
             long long j = v * 4 + k;
             stats[j].sum += temp_sum[k];
             stats[j].sum_sq += temp_sum_sq[k];
             if (temp_min[k] < stats[j].min_val) stats[j].min_val = temp_min[k];
             if (temp_max[k] > stats[j].max_val) stats[j].max_val = temp_max[k];
         }
     }
 
     double inv_N = 1.0 / static_cast<double>(N);
     for (long long j = 0; j < D; ++j) {
         stats[j].mean    = stats[j].sum * inv_N;
         stats[j].var     = std::max(0.0, stats[j].sum_sq * inv_N - stats[j].mean * stats[j].mean);
         stats[j].std_dev = std::sqrt(stats[j].var);
     }
     return true;
 }
 
 /* ------------------------------------------------------------------ */
 /* Phase 2 — SIMD scale and write                                     */
 /* ------------------------------------------------------------------ */
 static bool phase2_scale_and_write(
     const char   *input_path,
     const char   *output_path,
     long long     N,
     long long     D,
     long long     block_rows,
     ScalerMode    mode,
     const std::vector<ColStats> &stats,
     double       &compute_time)
 {
     FILE *fin  = std::fopen(input_path,  "rb");
     FILE *fout = std::fopen(output_path, "wb");
     if (!fin || !fout) return false;
 
     // Arrays mapped for shift/scale pre-calculation
     double* shift_arr = (double*)_mm_malloc(D * sizeof(double), 32);
     double* scale_arr = (double*)_mm_malloc(D * sizeof(double), 32);
 
     for (long long j = 0; j < D; ++j) {
         if (mode == ScalerMode::STANDARD) {
             shift_arr[j] = stats[j].mean;
             scale_arr[j] = (stats[j].std_dev != 0.0) ? 1.0 / stats[j].std_dev : 0.0;
         } else {
             double range = stats[j].max_val - stats[j].min_val;
             shift_arr[j] = stats[j].min_val;
             scale_arr[j] = (range != 0.0) ? 1.0 / range : 0.0;
         }
     }
 
     long long num_vecs = D / 4;
     
     // Pre-load shifts and scales into registers OUTSIDE the processing loop
     std::vector<__m256d> v_shift(num_vecs);
     std::vector<__m256d> v_scale(num_vecs);
     
     bool can_use_aligned_loads = (D % 4 == 0);
 
     for(long long v = 0; v < num_vecs; ++v) {
         if (can_use_aligned_loads) {
             v_shift[v] = _mm256_load_pd(&shift_arr[v * 4]);
             v_scale[v] = _mm256_load_pd(&scale_arr[v * 4]);
         } else {
             v_shift[v] = _mm256_loadu_pd(&shift_arr[v * 4]);
             v_scale[v] = _mm256_loadu_pd(&scale_arr[v * 4]);
         }
     }
 
     size_t elems_per_block = static_cast<size_t>(block_rows * D);
     double* block_data = (double*)_mm_malloc(elems_per_block * sizeof(double), 32);
     
     long long rows_remaining = N;
     compute_time = 0.0;
 
     while (rows_remaining > 0) {
         long long rows_this_block = std::min(block_rows, rows_remaining);
         size_t    elems           = static_cast<size_t>(rows_this_block * D);
         
         size_t got = std::fread(block_data, sizeof(double), elems, fin);
         if (got != elems) {
             _mm_free(block_data); _mm_free(shift_arr); _mm_free(scale_arr);
             std::fclose(fin); std::fclose(fout);
             return false;
         }
 
         // --- START PURE COMPUTE ---
         double t_start = now_sec();
 
         for (long long i = 0; i < rows_this_block; ++i) {
             double *row = block_data + i * D;
             
             for (long long v = 0; v < num_vecs; ++v) {
                 __m256d vx;
                 if (can_use_aligned_loads) {
                     vx = _mm256_load_pd(&row[v * 4]);
                 } else {
                     vx = _mm256_loadu_pd(&row[v * 4]);
                 }
                 
                 // Pure Math using pre-loaded registers: (x - shift) * scale
                 __m256d vres = _mm256_mul_pd(_mm256_sub_pd(vx, v_shift[v]), v_scale[v]);
                 
                 if (can_use_aligned_loads) {
                     _mm256_store_pd(&row[v * 4], vres);
                 } else {
                     _mm256_storeu_pd(&row[v * 4], vres);
                 }
             }
             
             // Tail processing
             for (long long j = num_vecs * 4; j < D; ++j) {
                 row[j] = (row[j] - shift_arr[j]) * scale_arr[j];
             }
         }
 
         compute_time += (now_sec() - t_start);
         // --- END PURE COMPUTE ---
 
         std::fwrite(block_data, sizeof(double), elems, fout);
         rows_remaining -= rows_this_block;
     }
 
     _mm_free(block_data);
     _mm_free(shift_arr);
     _mm_free(scale_arr);
     std::fclose(fin);
     std::fclose(fout);
     return true;
 }
 
 /* ------------------------------------------------------------------ */
 /* Print statistics summary                                           */
 /* ------------------------------------------------------------------ */
 static void print_stats(const std::vector<ColStats> &stats, long long D,
                          long long max_cols = 5)
 {
     std::printf("\n  Per-column statistics (first %lld of %lld columns shown):\n",
                 std::min(max_cols, D), D);
     std::printf("  %-6s  %-14s  %-14s  %-14s  %-14s  %-14s\n",
                 "col", "mean", "std_dev", "min", "max", "variance");
     std::printf("  %s\n", std::string(76, '-').c_str());
     for (long long j = 0; j < std::min(max_cols, D); ++j) {
         std::printf("  %-6lld  %-14.6f  %-14.6f  %-14.6f  %-14.6f  %-14.6f\n",
                     j, stats[j].mean, stats[j].std_dev,
                     stats[j].min_val, stats[j].max_val, stats[j].var);
     }
     if (D > max_cols)
         std::printf("  ... (%lld more columns)\n", D - max_cols);
     std::printf("\n");
 }
 
 /* ------------------------------------------------------------------ */
 /* main                                                               */
 /* ------------------------------------------------------------------ */
 int main(int argc, char *argv[])
 {
     if (argc < 6 || argc > 7) {
         std::fprintf(stderr,
             "Usage: %s input.bin output.bin N D mode [block_rows]\n", argv[0]);
         return EXIT_FAILURE;
     }
 
     const char *input_path  = argv[1];
     const char *output_path = argv[2];
     long long   N           = std::atoll(argv[3]);
     long long   D           = std::atoll(argv[4]);
     std::string mode_str    = argv[5];
     long long   block_rows  = (argc == 7) ? std::atoll(argv[6]) : 100000LL;
 
     ScalerMode mode;
     if (mode_str == "standard") mode = ScalerMode::STANDARD;
     else if (mode_str == "minmax") mode = ScalerMode::MINMAX;
     else return EXIT_FAILURE;
 
     std::printf("=== SIMD Scaler — AVX2 Implementation (Optimized) ===\n");
     std::vector<ColStats> stats(static_cast<size_t>(D));
 
     double compute_time_1 = 0.0;
     double t0 = now_sec();
     if (!phase1_compute_stats(input_path, N, D, block_rows, stats, compute_time_1)) return EXIT_FAILURE;
     double t1 = now_sec();
     
     print_stats(stats, D);
 
     double compute_time_2 = 0.0;
     double t2 = now_sec();
     if (!phase2_scale_and_write(input_path, output_path, N, D, block_rows, mode, stats, compute_time_2)) return EXIT_FAILURE;
     double t3 = now_sec();
 
     std::printf("=== Timing Summary ===\n");
     std::printf("  Phase 1 Total Wall Time:   %.3f s (Compute: %.3f s, I/O: %.3f s)\n", 
                 t1 - t0, compute_time_1, (t1 - t0) - compute_time_1);
     std::printf("  Phase 2 Total Wall Time:   %.3f s (Compute: %.3f s, I/O: %.3f s)\n", 
                 t3 - t2, compute_time_2, (t3 - t2) - compute_time_2);
     std::printf("  --------------------------------------------------\n");
     std::printf("  Total Compute Time Only:   %.3f s\n", compute_time_1 + compute_time_2);
     std::printf("  Total Execution Time:      %.3f s\n", t3 - t0);
 
     return EXIT_SUCCESS;
 }