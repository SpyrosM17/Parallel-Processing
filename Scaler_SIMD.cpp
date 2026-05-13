/**
 * Scaler_SIMD.cpp  —  SIMD AVX2 Implementation (Optimized & Aligned)
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
 #include <immintrin.h>
 
 static double now_sec() {
     using clk = std::chrono::steady_clock;
     return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
 }
 
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
     if (!fin) return false;
 
     size_t elems_per_block = static_cast<size_t>(block_rows * D);
     double* block_data = (double*)_mm_malloc(elems_per_block * sizeof(double), 32);
     
     // Explicitly 32-byte aligned arrays for AVX accumulators
     double* v_sum_arr    = (double*)_mm_malloc(D * sizeof(double), 32);
     double* v_sum_sq_arr = (double*)_mm_malloc(D * sizeof(double), 32);
     double* v_min_arr    = (double*)_mm_malloc(D * sizeof(double), 32);
     double* v_max_arr    = (double*)_mm_malloc(D * sizeof(double), 32);
 
     if (!block_data || !v_sum_arr || !v_sum_sq_arr || !v_min_arr || !v_max_arr) {
         if(block_data) _mm_free(block_data);
         if(v_sum_arr) _mm_free(v_sum_arr);
         if(v_sum_sq_arr) _mm_free(v_sum_sq_arr);
         if(v_min_arr) _mm_free(v_min_arr);
         if(v_max_arr) _mm_free(v_max_arr);
         std::fclose(fin);
         return false;
     }
 
     for (long long j = 0; j < D; ++j) {
         v_sum_arr[j]    = 0.0;
         v_sum_sq_arr[j] = 0.0;
         v_min_arr[j]    = std::numeric_limits<double>::infinity();
         v_max_arr[j]    = -std::numeric_limits<double>::infinity();
     }
 
     long long num_vecs = D / 4;
     long long rows_remaining = N;
     compute_time = 0.0;
 
     bool can_use_aligned_loads = (D % 4 == 0);
 
     while (rows_remaining > 0) {
         long long rows_this_block = std::min(block_rows, rows_remaining);
         size_t    elems           = static_cast<size_t>(rows_this_block * D);
         
         if (std::fread(block_data, sizeof(double), elems, fin) != elems) {
             _mm_free(block_data); _mm_free(v_sum_arr); _mm_free(v_sum_sq_arr);
             _mm_free(v_min_arr); _mm_free(v_max_arr);
             std::fclose(fin);
             return false;
         }
 
         double t_start = now_sec();
 
         for (long long i = 0; i < rows_this_block; ++i) {
             const double *row = block_data + i * D;
             
             for (long long v = 0; v < num_vecs; ++v) {
                 __m256d x = can_use_aligned_loads ? _mm256_load_pd(&row[v * 4]) : _mm256_loadu_pd(&row[v * 4]);
                 
                 // Load from L1 cache, compute, store back
                 __m256d s = _mm256_load_pd(&v_sum_arr[v * 4]);
                 s = _mm256_add_pd(s, x);
                 _mm256_store_pd(&v_sum_arr[v * 4], s);
 
                 __m256d sq = _mm256_load_pd(&v_sum_sq_arr[v * 4]);
                 sq = _mm256_fmadd_pd(x, x, sq);
                 _mm256_store_pd(&v_sum_sq_arr[v * 4], sq);
 
                 __m256d mn = _mm256_load_pd(&v_min_arr[v * 4]);
                 mn = _mm256_min_pd(mn, x);
                 _mm256_store_pd(&v_min_arr[v * 4], mn);
 
                 __m256d mx = _mm256_load_pd(&v_max_arr[v * 4]);
                 mx = _mm256_max_pd(mx, x);
                 _mm256_store_pd(&v_max_arr[v * 4], mx);
             }
             
             for (long long j = num_vecs * 4; j < D; ++j) {
                 double val = row[j];
                 stats[j].sum += val;
                 stats[j].sum_sq += val * val;
                 if (val < stats[j].min_val) stats[j].min_val = val;
                 if (val > stats[j].max_val) stats[j].max_val = val;
             }
         }
         
         compute_time += (now_sec() - t_start);
         rows_remaining -= rows_this_block;
     }
     
     std::fclose(fin);
     _mm_free(block_data);
 
     // Merge vector accumulators into main stats struct
     for (long long j = 0; j < num_vecs * 4; ++j) {
         stats[j].sum += v_sum_arr[j];
         stats[j].sum_sq += v_sum_sq_arr[j];
         if (v_min_arr[j] < stats[j].min_val) stats[j].min_val = v_min_arr[j];
         if (v_max_arr[j] > stats[j].max_val) stats[j].max_val = v_max_arr[j];
     }
 
     _mm_free(v_sum_arr);
     _mm_free(v_sum_sq_arr);
     _mm_free(v_min_arr);
     _mm_free(v_max_arr);
 
     double inv_N = 1.0 / static_cast<double>(N);
     for (long long j = 0; j < D; ++j) {
         stats[j].mean    = stats[j].sum * inv_N;
         stats[j].var     = std::max(0.0, stats[j].sum_sq * inv_N - stats[j].mean * stats[j].mean);
         stats[j].std_dev = std::sqrt(stats[j].var);
     }
     return true;
 }
 
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
     bool can_use_aligned_loads = (D % 4 == 0);
     size_t elems_per_block = static_cast<size_t>(block_rows * D);
     double* block_data = (double*)_mm_malloc(elems_per_block * sizeof(double), 32);
     
     long long rows_remaining = N;
     compute_time = 0.0;
 
     while (rows_remaining > 0) {
         long long rows_this_block = std::min(block_rows, rows_remaining);
         size_t    elems           = static_cast<size_t>(rows_this_block * D);
         
         if (std::fread(block_data, sizeof(double), elems, fin) != elems) {
             _mm_free(block_data); _mm_free(shift_arr); _mm_free(scale_arr);
             std::fclose(fin); std::fclose(fout);
             return false;
         }
 
         double t_start = now_sec();
 
         for (long long i = 0; i < rows_this_block; ++i) {
             double *row = block_data + i * D;
             
             for (long long v = 0; v < num_vecs; ++v) {
                 __m256d vx = can_use_aligned_loads ? _mm256_load_pd(&row[v * 4]) : _mm256_loadu_pd(&row[v * 4]);
                 __m256d vshift = _mm256_load_pd(&shift_arr[v * 4]);
                 __m256d vscale = _mm256_load_pd(&scale_arr[v * 4]);
                 
                 __m256d vres = _mm256_mul_pd(_mm256_sub_pd(vx, vshift), vscale);
                 
                 if (can_use_aligned_loads) {
                     _mm256_store_pd(&row[v * 4], vres);
                 } else {
                     _mm256_storeu_pd(&row[v * 4], vres);
                 }
             }
             
             for (long long j = num_vecs * 4; j < D; ++j) {
                 row[j] = (row[j] - shift_arr[j]) * scale_arr[j];
             }
         }
 
         compute_time += (now_sec() - t_start);
 
         if (std::fwrite(block_data, sizeof(double), elems, fout) != elems) {
             _mm_free(block_data); _mm_free(shift_arr); _mm_free(scale_arr);
             std::fclose(fin); std::fclose(fout);
             return false;
         }
         rows_remaining -= rows_this_block;
     }
 
     _mm_free(block_data);
     _mm_free(shift_arr);
     _mm_free(scale_arr);
     std::fclose(fin);
     std::fclose(fout);
     return true;
 }
 
 static void print_stats(const std::vector<ColStats> &stats, long long D, long long max_cols = 5) {
     std::printf("\n  Per-column statistics (first %lld of %lld columns shown):\n", std::min(max_cols, D), D);
     std::printf("  %-6s  %-14s  %-14s  %-14s  %-14s  %-14s\n", "col", "mean", "std_dev", "min", "max", "variance");
     std::printf("  %s\n", std::string(76, '-').c_str());
     for (long long j = 0; j < std::min(max_cols, D); ++j) {
         std::printf("  %-6lld  %-14.6f  %-14.6f  %-14.6f  %-14.6f  %-14.6f\n",
                     j, stats[j].mean, stats[j].std_dev, stats[j].min_val, stats[j].max_val, stats[j].var);
     }
     if (D > max_cols) std::printf("  ... (%lld more columns)\n", D - max_cols);
     std::printf("\n");
 }
 
 int main(int argc, char *argv[]) {
     if (argc < 6 || argc > 7) {
         std::fprintf(stderr, "Usage: %s input.bin output.bin N D mode [block_rows]\n", argv[0]);
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
     std::printf("  Phase 1 Total Wall Time:   %.3f s (Compute: %.3f s, I/O: %.3f s)\n", t1 - t0, compute_time_1, (t1 - t0) - compute_time_1);
     std::printf("  Phase 2 Total Wall Time:   %.3f s (Compute: %.3f s, I/O: %.3f s)\n", t3 - t2, compute_time_2, (t3 - t2) - compute_time_2);
     std::printf("  --------------------------------------------------\n");
     std::printf("  Total Compute Time Only:   %.3f s\n", compute_time_1 + compute_time_2);
     std::printf("  Total Execution Time:      %.3f s\n", t3 - t0);
 
     return EXIT_SUCCESS;
 }