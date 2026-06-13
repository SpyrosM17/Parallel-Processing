/*
 * Scaler_OpenMP.cpp
 *
 * OpenMP version of the scaler. Uses threads to parallelise the row processing.
 * I/O is still sequential but we process the blocks in parallel to speed it up.
 *
 * Usage:
 * ./scaler_openmp input.bin output.bin N D mode [block_rows]
 * mode: standard | minmax
 *
 * Build:
 * g++ -O3 -std=c++17 -fopenmp -o scaler_openmp Scaler_OpenMP.cpp
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
#include <omp.h>


/* ------------------------------------------------------------------ */
 /* Get current time                                                    */
 /* ------------------------------------------------------------------ */
static double now_sec()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

 /* ------------------------------------------------------------------ */
 /* Stat accumulators                                                   */
 /* ------------------------------------------------------------------ */
struct ColStats {
    double sum;
    double sum_sq;
    double min_val;
    double max_val;
    /* Calculated at the end */
    double mean;
    double var;
    double std_dev;
};

enum class ScalerMode { STANDARD, MINMAX };

/* ------------------------------------------------------------------ */
 /* Phase 1: Read file in blocks and compute stats                     */
 /* ------------------------------------------------------------------ */

static bool phase1_compute_stats(
    const char   *input_path,
    long long     N,
    long long     D,
    long long     block_rows,
    std::vector<ColStats> &stats,
    double       &wall_time,
    double       &compute_time)
{
    /* Init accumulators */
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

    std::vector<double> block(static_cast<size_t>(block_rows * D));
    int max_threads = omp_get_max_threads();
    std::vector<ColStats> local_stats(static_cast<size_t>(max_threads) * D);

    long long rows_remaining = N;
    compute_time = 0.0;
    double t_wall_start = now_sec();

    while (rows_remaining > 0) {
        long long rows_this_block = std::min(block_rows, rows_remaining);
        size_t elems = static_cast<size_t>(rows_this_block * D);

        size_t got = std::fread(block.data(), sizeof(double), elems, fin);
        if (got != elems) {
            std::fprintf(stderr,
                "[ERROR] Phase 1: expected %zu elements, read %zu (row offset %lld)\n",
                elems, got, N - rows_remaining);
            std::fclose(fin);
            return false;
        }

        /* ---- Only time the actual math, not the disk I/O ---- */
        double t_c0 = now_sec();
        int threads_used = 0;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp single
            threads_used = omp_get_num_threads();

            ColStats *my_stats = &local_stats[static_cast<size_t>(tid) * D];

            for (long long j = 0; j < D; ++j) {
                my_stats[j].sum     = 0.0;
                my_stats[j].sum_sq  = 0.0;
                my_stats[j].min_val =  std::numeric_limits<double>::infinity();
                my_stats[j].max_val = -std::numeric_limits<double>::infinity();
            }

            #pragma omp for schedule(static)
            for (long long i = 0; i < rows_this_block; ++i) {
                const double *row = block.data() + i * D;
                for (long long j = 0; j < D; ++j) {
                    double v = row[j];
                    my_stats[j].sum    += v;
                    my_stats[j].sum_sq += v * v;
                    if (v < my_stats[j].min_val) my_stats[j].min_val = v;
                    if (v > my_stats[j].max_val) my_stats[j].max_val = v;
                }
            }
        }

        for (int t = 0; t < threads_used; ++t) {
            ColStats *my_stats = &local_stats[static_cast<size_t>(t) * D];
            for (long long j = 0; j < D; ++j) {
                stats[j].sum    += my_stats[j].sum;
                stats[j].sum_sq += my_stats[j].sum_sq;
                if (my_stats[j].min_val < stats[j].min_val) stats[j].min_val = my_stats[j].min_val;
                if (my_stats[j].max_val > stats[j].max_val) stats[j].max_val = my_stats[j].max_val;
            }
        }

        compute_time += now_sec() - t_c0;
        rows_remaining -= rows_this_block;
    }

    wall_time = now_sec() - t_wall_start;
    std::fclose(fin);

    /* Get the variance and std dev */
    double inv_N = 1.0 / static_cast<double>(N);
    for (long long j = 0; j < D; ++j) {
        stats[j].mean    = stats[j].sum * inv_N;
        stats[j].var     = stats[j].sum_sq * inv_N - stats[j].mean * stats[j].mean;
        if (stats[j].var < 0.0) stats[j].var = 0.0;
        stats[j].std_dev = std::sqrt(stats[j].var);
    }

    return true;
}


 /* ------------------------------------------------------------------ */
 /* Phase 2: scale the values and write them out                       */
 /* ------------------------------------------------------------------ */
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
    FILE *fout = std::fopen(output_path, "wb");
    if (!fin) {
        std::fprintf(stderr, "[ERROR] Cannot open input file: %s\n", input_path);
        return false;
    }
    if (!fout) {
        std::fprintf(stderr, "[ERROR] Cannot open output file: %s\n", output_path);
        std::fclose(fin);
        return false;
    }

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
    compute_time = 0.0;
    double t_wall_start = now_sec();

    while (rows_remaining > 0) {
        long long rows_this_block = std::min(block_rows, rows_remaining);
        size_t elems = static_cast<size_t>(rows_this_block * D);

        size_t got = std::fread(block.data(), sizeof(double), elems, fin);
        if (got != elems) {
            std::fprintf(stderr,
                "[ERROR] Phase 2: expected %zu elements, read %zu (row offset %lld)\n",
                elems, got, N - rows_remaining);
            std::fclose(fin);
            std::fclose(fout);
            return false;
        }

        /* ---- Time the scaling separately ---- */
        double t_c0 = now_sec();

        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < rows_this_block; ++i) {
            double *row = block.data() + i * D;
            for (long long j = 0; j < D; ++j) {
                row[j] = (row[j] - shift[j]) * scale[j];
            }
        }

        compute_time += now_sec() - t_c0;

        size_t written = std::fwrite(block.data(), sizeof(double), elems, fout);
        if (written != elems) {
            std::fprintf(stderr,
                "[ERROR] Phase 2: write failed at row offset %lld\n",
                N - rows_remaining);
            std::fclose(fin);
            std::fclose(fout);
            return false;
        }

        rows_remaining -= rows_this_block;
    }

    wall_time = now_sec() - t_wall_start;
    std::fclose(fin);
    std::fclose(fout);
    return true;
    
}

 /* ------------------------------------------------------------------ */
 /* Print a summary so we know it worked                                */
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
 /* Main                                                                */
 /* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    if (argc < 6 || argc > 7) {
        std::fprintf(stderr,
            "Usage: %s input.bin output.bin N D mode [block_rows]\n"
            "  mode: standard | minmax\n"
            "  block_rows: optional, default = 256000\n",
            argv[0]);
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

    int nthreads = omp_get_max_threads();
    double file_size_gb = static_cast<double>(N) * D * sizeof(double) / (1 << 30);
    double block_mb     = static_cast<double>(block_rows) * D * sizeof(double) / (1 << 20);

    std::printf("=== OpenMP Scaler ===\n");
    std::printf("  Threads:     %d\n", nthreads);
    std::printf("  Input:       %s\n", input_path);
    std::printf("  Output:      %s\n", output_path);
    std::printf("  N (rows):    %lld\n", N);
    std::printf("  D (cols):    %lld\n", D);
    std::printf("  Mode:        %s\n", mode_str.c_str());
    std::printf("  Block rows:  %lld  (≈ %.1f MB per block)\n", block_rows, block_mb);
    std::printf("  File size:   ≈ %.3f GB\n\n", file_size_gb);

    std::vector<ColStats> stats(static_cast<size_t>(D));

     /* ================================================================
      * Phase 1
      * ================================================================ */

    std::printf("[Phase 1] Computing per-column statistics (OpenMP)...\n");
    double wall1 = 0.0, compute1 = 0.0;
    if (!phase1_compute_stats(input_path, N, D, block_rows, stats, wall1, compute1))
        return EXIT_FAILURE;
    std::printf("[Phase 1] Done in %.3f s  (compute %.3f s, I/O %.3f s)\n",
                 wall1, compute1, wall1 - compute1);
     print_stats(stats, D);

     /* ================================================================
      * Phase 2
      * ================================================================ */

    std::printf("[Phase 2] Applying %s and writing output (OpenMP)...\n", mode_str.c_str());
    double wall2 = 0.0, compute2 = 0.0;
    if (!phase2_scale_and_write(input_path, output_path, N, D, block_rows, mode, stats, wall2, compute2))
        return EXIT_FAILURE;
    std::printf("[Phase 2] Done in %.3f s  (compute %.3f s, I/O  %.3f s)\n",
                 wall2, compute2, wall2 - compute2);
    /* ---- End summary ---- */
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