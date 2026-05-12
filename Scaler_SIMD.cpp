/**
 * Scaler_SIMD.cpp  —  SIMD-Optimized Implementation
 * --------------------------------------------------
 *
 * Reads a raw binary matrix X ∈ R^{N×D} (row-major, double precision),
 * computes per-column statistics (mean, min, max, variance, std-dev),
 * then applies either StandardScaler or MinMaxScaler using SIMD intrinsics
 * and writes the normalised matrix to a new raw binary file.
 *
 * SIMD Optimizations:
 * - Uses AVX2 instructions (256-bit vectors) for up to 4 double elements per operation
 * - Compiler flags: -O3 -march=haswell -ffast-math
 * - Vectorizes the scaling transformation loop (Phase 2)
 * - Vectorizes min/max reduction in Phase 1
 *
 * Processing is done block-by-block for out-of-core execution.
 *
 * Usage:
 *   ./scaler_simd input.bin output.bin N D mode [block_rows]
 *
 * Build:
 *   g++ -O3 -march=haswell -std=c++17 -ffast-math -o scaler_simd Scaler_SIMD.cpp
 *   clang++ -O3 -march=haswell -std=c++17 -ffast-math -o scaler_simd Scaler_SIMD.cpp
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
#include <immintrin.h>  /* AVX/AVX2 intrinsics */
#include <algorithm>

/* ------------------------------------------------------------------ */
/*  Tiny helper: wall-clock timer                                     */
/* ------------------------------------------------------------------ */
static double now_sec()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

/* ------------------------------------------------------------------ */
/*  Per-column accumulators                                            */
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

/* ================================================================== */
/*  SIMD Helper: Horizontal min reduction for __m256d              */
/* ================================================================== */
static inline double hmin_pd(__m256d v)
{
    /* v = [a, b, c, d] */
    __m256d tmp1 = _mm256_permute_pd(v, 0b0101);  /* [b, a, d, c] */
    __m256d min1 = _mm256_min_pd(v, tmp1);        /* [min(a,b), min(b,a), min(c,d), min(d,c)] */
    
    __m256d tmp2 = _mm256_shuffle_pd(min1, min1, 0b00); /* Extract [min(a,b), min(a,b), min(c,d), min(c,d)] */
    __m256d min2 = _mm256_min_pd(min1, tmp2);
    
    return _mm256_cvtsd_f64(min2);
}

/* ================================================================== */
/*  SIMD Helper: Horizontal max reduction for __m256d              */
/* ================================================================== */
static inline double hmax_pd(__m256d v)
{
    __m256d tmp1 = _mm256_permute_pd(v, 0b0101);  /* [b, a, d, c] */
    __m256d max1 = _mm256_max_pd(v, tmp1);        /* [max(a,b), max(b,a), max(c,d), max(d,c)] */
    
    __m256d tmp2 = _mm256_shuffle_pd(max1, max1, 0b00); /* Extract [max(a,b), max(a,b), max(c,d), max(c,d)] */
    __m256d max2 = _mm256_max_pd(max1, tmp2);
    
    return _mm256_cvtsd_f64(max2);
}

/* ================================================================== */
/*  SIMD Helper: Horizontal sum reduction for __m256d             */
/* ================================================================== */
static inline double hsum_pd(__m256d v)
{
    __m256d tmp1 = _mm256_permute2f128_pd(v, v, 0x01);  /* [c+d, d+c, a+b, b+a] */
    __m256d sum1 = _mm256_add_pd(v, tmp1);
    __m256d tmp2 = _mm256_permute_pd(sum1, 0b01);
    __m256d sum2 = _mm256_add_pd(sum1, tmp2);
    return _mm256_cvtsd_f64(sum2);
}

/* ------------------------------------------------------------------ */
/*  Phase 1 — scan the whole file in blocks, accumulate statistics    */
/*  SIMD-optimized version                                            */
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
        /* SIMD: Process 4 doubles at a time with AVX2 */
        for (long long j = 0; j < D; ++j) {
            __m256d acc_sum    = _mm256_setzero_pd();
            __m256d acc_sum_sq = _mm256_setzero_pd();
            __m256d acc_min    = _mm256_set1_pd(std::numeric_limits<double>::infinity());
            __m256d acc_max    = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

            /* SIMD loop: process 4 rows at a time */
            long long i;
            for (i = 0; i + 3 < rows_this_block; i += 4) {
                /* Load 4 values from column j of rows i, i+1, i+2, i+3 */
                double *ptr = block.data() + i * D + j;
                __m256d vals = _mm256_setr_pd(ptr[0], ptr[D], ptr[2*D], ptr[3*D]);
                
                /* Accumulate sum and sum_sq */
                acc_sum    = _mm256_add_pd(acc_sum, vals);
                __m256d vals_sq = _mm256_mul_pd(vals, vals);
                acc_sum_sq = _mm256_add_pd(acc_sum_sq, vals_sq);
                
                /* Track min/max */
                acc_min = _mm256_min_pd(acc_min, vals);
                acc_max = _mm256_max_pd(acc_max, vals);
            }

            /* Horizontal reductions from SIMD accumulators */
            stats[j].sum   += hsum_pd(acc_sum);
            stats[j].sum_sq += hsum_pd(acc_sum_sq);
            stats[j].min_val = std::min(stats[j].min_val, hmin_pd(acc_min));
            stats[j].max_val = std::max(stats[j].max_val, hmax_pd(acc_max));

            /* Handle remainder rows (< 4) */
            for (; i < rows_this_block; ++i) {
                double v = *(block.data() + i * D + j);
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
        stats[j].var     = stats[j].sum_sq * inv_N - stats[j].mean * stats[j].mean;
        if (stats[j].var < 0.0) stats[j].var = 0.0;
        stats[j].std_dev = std::sqrt(stats[j].var);
    }

    return true;
}

/* ------------------------------------------------------------------ */
/*  Phase 2 — re-read file, apply scaling, write output               */
/*  SIMD-optimized version                                            */
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

    /* Pre-compute per-column scale factors */
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

        /* Apply scaling in-place with SIMD optimization */
        /* Process each row, vectorizing across columns */
        for (long long i = 0; i < rows_this_block; ++i) {
            double *row = block.data() + i * D;
            
            /* SIMD loop: process 4 columns at a time */
            long long j;
            for (j = 0; j + 3 < D; j += 4) {
                /* Load 4 values */
                __m256d vals = _mm256_loadu_pd(row + j);
                
                /* Load shift and scale vectors */
                __m256d shift_v = _mm256_setr_pd(shift[j], shift[j+1], shift[j+2], shift[j+3]);
                __m256d scale_v = _mm256_setr_pd(scale[j], scale[j+1], scale[j+2], scale[j+3]);
                
                /* Apply: (vals - shift) * scale */
                __m256d result = _mm256_mul_pd(_mm256_sub_pd(vals, shift_v), scale_v);
                
                /* Store result */
                _mm256_storeu_pd(row + j, result);
            }
            
            /* Handle remainder columns (< 4) */
            for (; j < D; ++j) {
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
/*  Print statistics summary                                           */
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
/*  main                                                               */
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
    double file_size_gb = static_cast<double>(N) * D * sizeof(double) / (1LL << 30);
    double block_mb     = static_cast<double>(block_rows) * D * sizeof(double) / (1LL << 20);

    std::printf("=== SIMD-Optimized Scaler (AVX2) ===\n");
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
     * PHASE 1 — compute statistics (SIMD-optimized)
     * ================================================================ */
    std::printf("[Phase 1] Computing per-column statistics (SIMD)...\n");
    double t0 = now_sec();

    if (!phase1_compute_stats(input_path, N, D, block_rows, stats)) {
        return EXIT_FAILURE;
    }

    double t1 = now_sec();
    std::printf("[Phase 1] Done in %.3f seconds.\n", t1 - t0);
    print_stats(stats, D);

    /* ================================================================
     * PHASE 2 — scale and write (SIMD-optimized)
     * ================================================================ */
    std::printf("[Phase 2] Applying %s and writing output (SIMD)...\n",
                mode_str.c_str());
    double t2 = now_sec();

    if (!phase2_scale_and_write(input_path, output_path, N, D,
                                 block_rows, mode, stats)) {
        return EXIT_FAILURE;
    }

    double t3 = now_sec();
    std::printf("[Phase 2] Done in %.3f seconds.\n", t3 - t2);

    /* ---- Overall timing ---- */
    std::printf("\n=== Timing Summary (SIMD) ===\n");
    std::printf("  Phase 1 (stats):     %.3f s\n", t1 - t0);
    std::printf("  Phase 2 (scaling):   %.3f s\n", t3 - t2);
    std::printf("  Total:               %.3f s\n", t3 - t0);
    std::printf("  Throughput:          %.2f GB/s  (total data read ≈ 2× input)\n",
                2.0 * file_size_gb / (t3 - t0));

    return EXIT_SUCCESS;
}
