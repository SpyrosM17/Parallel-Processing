/**
 * scaler.cpp  —  Serial Reference Implementation
 * ------------------------------------------------
 
 * Reads a raw binary matrix X ∈ R^{N×D} (row-major, double precision),
 * computes per-column statistics (mean, min, max, variance, std-dev),
 * then applies either StandardScaler or MinMaxScaler and writes the
 * normalised matrix to a new raw binary file.
 *
 * Processing is done block-by-block so that files larger than available
 * RAM are handled correctly (out-of-core execution).
 *
 * Usage:
 *   ./scaler input.bin output.bin N D mode [block_rows]
 *
 *   input.bin   – raw binary input  (N*D doubles, row-major)
 *   output.bin  – raw binary output (N*D doubles, row-major)
 *   N           – number of rows (samples)
 *   D           – number of columns (features)
 *   mode        – "standard"  →  StandardScaler
 *                 "minmax"    →  MinMaxScaler
 *   block_rows  – (optional) rows per I/O block; default = 100 000
 *
 * Build:
 *   g++ -O2 -std=c++17 -o scaler Scaler.cpp
 *   ./scaler data_10M_128.bin out_standard.bin 1000 12 standard
 *   ./scaler data_10M_128.bin out_minmax.bin 1000 12 minmax 50000
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
 
 /* ------------------------------------------------------------------ */
 /*  Tiny helper: wall-clock timer                                       */
 /* ------------------------------------------------------------------ */
 static double now_sec()
 {
     using clk = std::chrono::steady_clock;
     return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
 }
 
 /* ------------------------------------------------------------------ */
 /*  Per-column accumulators                                             */
 /* ------------------------------------------------------------------ */
 struct ColStats {
     double sum;      /* Σ x_ij                      */
     double sum_sq;   /* Σ x_ij²                     */
     double min_val;
     double max_val;
     /* Derived (filled after full scan) */
     double mean;
     double var;
     double std_dev;
 };
 
 /* ------------------------------------------------------------------ */
 /*  Phase 1 — scan the whole file in blocks, accumulate statistics     */
 /* ------------------------------------------------------------------ */
 static bool phase1_compute_stats(
     const char   *input_path,
     long long     N,          /* total rows                            */
     long long     D,          /* total columns                         */
     long long     block_rows, /* rows to read per iteration            */
     std::vector<ColStats> &stats)
 {
     /* Initialise accumulators */
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
 
     /* Allocate one block buffer */
     std::vector<double> block(static_cast<size_t>(block_rows * D));
 
     long long rows_remaining = N;
     long long total_read     = 0;
 
     while (rows_remaining > 0) {
         long long rows_this_block = std::min(block_rows, rows_remaining);
         size_t    elems           = static_cast<size_t>(rows_this_block * D);
 
         size_t got = std::fread(block.data(), sizeof(double), elems, fin);
         if (got != elems) {
             std::fprintf(stderr,
                 "[ERROR] Phase 1: expected %zu elements, read %zu "
                 "(row offset %lld)\n", elems, got, total_read);
             std::fclose(fin);
             return false;
         }
 
         /* Accumulate column statistics for this block */
         for (long long i = 0; i < rows_this_block; ++i) {
             const double *row = block.data() + i * D;
             for (long long j = 0; j < D; ++j) {
                 double v        = row[j];
                 stats[j].sum   += v;
                 stats[j].sum_sq += v * v;
                 if (v < stats[j].min_val) stats[j].min_val = v;
                 if (v > stats[j].max_val) stats[j].max_val = v;
             }
         }
 
         rows_remaining -= rows_this_block;
         total_read     += rows_this_block;
     }
 
     std::fclose(fin);
 
     /* Derive mean, variance, std-dev from accumulators */
     double inv_N = 1.0 / static_cast<double>(N);
     for (long long j = 0; j < D; ++j) {
         stats[j].mean    = stats[j].sum * inv_N;
         /* Var = E[x²] - E[x]²  (numerically equivalent, single-pass friendly) */
         stats[j].var     = stats[j].sum_sq * inv_N - stats[j].mean * stats[j].mean;
         /* Guard against tiny floating-point negatives due to cancellation */
         if (stats[j].var < 0.0) stats[j].var = 0.0;
         stats[j].std_dev = std::sqrt(stats[j].var);
     }
 
     return true;
 }
 
 /* ------------------------------------------------------------------ */
 /*  Phase 2 — re-read file, apply scaling, write output                */
 /* ------------------------------------------------------------------ */
 enum class ScalerMode { STANDARD, MINMAX };
 
 static bool phase2_scale_and_write(
     const char   *input_path,
     const char   *output_path,
     long long     N,
     long long     D,
     long long     block_rows,
     ScalerMode    mode,
     const std::vector<ColStats> &stats)
 {
     FILE *fin  = std::fopen(input_path,  "rb");
     FILE *fout = std::fopen(output_path, "wb");
 
     if (!fin) {
         std::fprintf(stderr, "[ERROR] Cannot open input file:  %s\n", input_path);
         return false;
     }
     if (!fout) {
         std::fprintf(stderr, "[ERROR] Cannot open output file: %s\n", output_path);
         std::fclose(fin);
         return false;
     }
 
     /*
      * Pre-compute per-column scale factors to avoid repeated division
      * inside the hot loop.
      *
      * StandardScaler:  scale = 1 / σ_j   (set to 0 if σ_j == 0)
      * MinMaxScaler:    scale = 1 / (max_j - min_j)  (set to 0 if range == 0)
      */
     std::vector<double> shift(D), scale(D);
     for (long long j = 0; j < D; ++j) {
         if (mode == ScalerMode::STANDARD) {
             shift[j] = stats[j].mean;
             scale[j] = (stats[j].std_dev != 0.0) ? 1.0 / stats[j].std_dev : 0.0;
         } else {
             double range = stats[j].max_val - stats[j].min_val;
             shift[j] = stats[j].min_val;
             scale[j] = (range != 0.0) ? 1.0 / range : 0.0;
         }
     }
 
     std::vector<double> block(static_cast<size_t>(block_rows * D));
 
     long long rows_remaining = N;
     long long total_written  = 0;
 
     while (rows_remaining > 0) {
         long long rows_this_block = std::min(block_rows, rows_remaining);
         size_t    elems           = static_cast<size_t>(rows_this_block * D);
 
         size_t got = std::fread(block.data(), sizeof(double), elems, fin);
         if (got != elems) {
             std::fprintf(stderr,
                 "[ERROR] Phase 2: expected %zu elements, read %zu "
                 "(row offset %lld)\n", elems, got, total_written);
             std::fclose(fin);
             std::fclose(fout);
             return false;
         }
 
         /* Apply scaling in-place */
         for (long long i = 0; i < rows_this_block; ++i) {
             double *row = block.data() + i * D;
             for (long long j = 0; j < D; ++j) {
                 row[j] = (row[j] - shift[j]) * scale[j];
             }
         }
 
         size_t written = std::fwrite(block.data(), sizeof(double), elems, fout);
         if (written != elems) {
             std::fprintf(stderr,
                 "[ERROR] Phase 2: write failed at row offset %lld\n",
                 total_written);
             std::fclose(fin);
             std::fclose(fout);
             return false;
         }
 
         rows_remaining -= rows_this_block;
         total_written  += rows_this_block;
     }
 
     std::fclose(fin);
     std::fclose(fout);
     return true;
 }
 
 /* ------------------------------------------------------------------ */
 /*  Print statistics summary                                            */
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
                     j,
                     stats[j].mean,
                     stats[j].std_dev,
                     stats[j].min_val,
                     stats[j].max_val,
                     stats[j].var);
     }
     if (D > max_cols)
         std::printf("  ... (%lld more columns)\n", D - max_cols);
     std::printf("\n");
 }
 
 /* ------------------------------------------------------------------ */
 /*  main                                                                */
 /* ------------------------------------------------------------------ */
 int main(int argc, char *argv[])
 {
     /* ---- Argument parsing ---- */
     if (argc < 6 || argc > 7) {
         std::fprintf(stderr,
             "Usage: %s input.bin output.bin N D mode [block_rows]\n"
             "  mode: standard | minmax\n"
             "  block_rows: optional, default = 100000\n",
             argv[0]);
         return EXIT_FAILURE;
     }
 
     const char *input_path  = argv[1];
     const char *output_path = argv[2];
     long long   N           = std::atoll(argv[3]);
     long long   D           = std::atoll(argv[4]);
     std::string mode_str    = argv[5];
     long long   block_rows  = (argc == 7) ? std::atoll(argv[6]) : 100000LL;
 
     if (N <= 0 || D <= 0) {
         std::fprintf(stderr, "[ERROR] N and D must be positive integers.\n");
         return EXIT_FAILURE;
     }
     if (block_rows <= 0) {
         std::fprintf(stderr, "[ERROR] block_rows must be a positive integer.\n");
         return EXIT_FAILURE;
     }
 
     ScalerMode mode;
     if (mode_str == "standard") {
         mode = ScalerMode::STANDARD;
     } else if (mode_str == "minmax") {
         mode = ScalerMode::MINMAX;
     } else {
         std::fprintf(stderr,
             "[ERROR] Unknown mode '%s'. Use 'standard' or 'minmax'.\n",
             mode_str.c_str());
         return EXIT_FAILURE;
     }
 
     /* ---- Configuration summary ---- */
     double file_size_gb = static_cast<double>(N) * D * sizeof(double) / (1 << 30);
     double block_mb     = static_cast<double>(block_rows) * D * sizeof(double) / (1 << 20);
 
     std::printf("=== Serial Scaler — Reference Implementation ===\n");
     std::printf("  Input:      %s\n",   input_path);
     std::printf("  Output:     %s\n",   output_path);
     std::printf("  N (rows):   %lld\n", N);
     std::printf("  D (cols):   %lld\n", D);
     std::printf("  Mode:       %s\n",   mode_str.c_str());
     std::printf("  Block rows: %lld  (≈ %.1f MB per block)\n", block_rows, block_mb);
     std::printf("  File size:  ≈ %.3f GB\n\n", file_size_gb);
 
     /* ---- Allocate statistics array ---- */
     std::vector<ColStats> stats(static_cast<size_t>(D));
 
     /* ================================================================
      * PHASE 1 — compute statistics
      * ================================================================ */
     std::printf("[Phase 1] Computing per-column statistics...\n");
     double t0 = now_sec();
 
     if (!phase1_compute_stats(input_path, N, D, block_rows, stats)) {
         return EXIT_FAILURE;
     }
 
     double t1 = now_sec();
     std::printf("[Phase 1] Done in %.3f seconds.\n", t1 - t0);
     print_stats(stats, D);
 
     /* ================================================================
      * PHASE 2 — scale and write
      * ================================================================ */
     std::printf("[Phase 2] Applying %s and writing output...\n",
                 mode_str.c_str());
     double t2 = now_sec();
 
     if (!phase2_scale_and_write(input_path, output_path, N, D,
                                  block_rows, mode, stats)) {
         return EXIT_FAILURE;
     }
 
     double t3 = now_sec();
     std::printf("[Phase 2] Done in %.3f seconds.\n", t3 - t2);
 
     /* ---- Overall timing ---- */
     std::printf("\n=== Timing Summary ===\n");
     std::printf("  Phase 1 (stats):     %.3f s\n", t1 - t0);
     std::printf("  Phase 2 (scaling):   %.3f s\n", t3 - t2);
     std::printf("  Total:               %.3f s\n", t3 - t0);
     std::printf("  Throughput:          %.2f GB/s  (total data read ≈ 2× input)\n",
                 2.0 * file_size_gb / (t3 - t0));
 
     return EXIT_SUCCESS;
 }
