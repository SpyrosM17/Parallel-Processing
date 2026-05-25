/*
 * Scaler_MPI.cpp  —  MPI Implementation
 * ----------------------------------------
 *
 * Distributes the N×D matrix row-wise across MPI processes.
 * Each process owns a contiguous partition of rows and accesses
 * its portion of the file directly by byte offset using MPI-IO
 * (MPI_File_read_at / MPI_File_write_at — independent I/O).
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │ Phase 1 — Statistics                                        │
 * │  Each process scans its row partition block-by-block,       │
 * │  accumulating local sum, sum_sq, min, max per column.       │
 * │  Four MPI_Allreduce calls (SUM/SUM/MIN/MAX) merge the       │
 * │  partial results into global statistics. All processes then │
 * │  derive mean, variance and std_dev identically.             │
 * ├─────────────────────────────────────────────────────────────┤
 * │ Phase 2 — Scale and Write                                   │
 * │  Each process re-reads its partition block-by-block,        │
 * │  applies the global shift/scale transformation in-place,    │
 * │  and writes the result at the correct byte offset in the    │
 * │  output file using MPI_File_write_at.                       │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Timing:
 *   MPI_Wtime() is used throughout. Wall times are the maximum
 *   across all processes (MPI_Allreduce MAX), giving the true
 *   parallel wall-clock time. The Allreduce communication cost
 *   for Phase 1 is broken out separately.
 *
 * Row Partition:
 *   Process r owns rows [ row_start_r,  row_start_r + my_rows_r )
 *     base      = N / nprocs
 *     remainder = N % nprocs
 *     my_rows   = base + (rank < remainder ? 1 : 0)
 *     row_start = rank * base + min(rank, remainder)
 *   This guarantees at most 1-row imbalance between processes.
 *
 * MPI count overflow guard:
 *   MPI counts are 32-bit ints. We require block_rows * D < INT_MAX.
 *   The default block_rows = 256 000 is safe for D up to ~8 000.
 *
 * Usage:
 *   mpirun -np <P> ./scaler_mpi input.bin output.bin N D mode [block_rows]
 *   mode: standard | minmax
 *   block_rows: optional, default = 256000
 *
 * Build:
 *   mpicxx -O3 -std=c++17 -o scaler_mpi Scaler_MPI.cpp
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>

/* ------------------------------------------------------------------ */
/*  Per-column accumulators                                             */
/* ------------------------------------------------------------------ */
struct ColStats {
    double sum;
    double sum_sq;
    double min_val;
    double max_val;
    /* Derived after global Allreduce */
    double mean;
    double var;
    double std_dev;
};

enum class ScalerMode { STANDARD, MINMAX };

/* ------------------------------------------------------------------ */
/*  Row-partition helper                                                */
/*  Returns the [row_start, row_start + my_rows) range for 'rank'.    */
/* ------------------------------------------------------------------ */
static void compute_partition(long long N, int nprocs, int rank,
                               long long &my_rows, long long &my_row_start)
{
    long long base      = N / (long long)nprocs;
    long long remainder = N % (long long)nprocs;
    my_rows      = base + ((long long)rank < remainder ? 1LL : 0LL);
    my_row_start = (long long)rank * base
                   + std::min((long long)rank, remainder);
}

/* ------------------------------------------------------------------ */
/*  Phase 1 — local accumulation  +  MPI_Allreduce merge               */
/*                                                                      */
/*  wall_time    – max across all processes (MPI_Allreduce MAX)         */
/*  compute_time – local compute-loop time only (no I/O, no comm)      */
/*  comm_time    – time spent in the four MPI_Allreduce calls          */
/* ------------------------------------------------------------------ */
static bool phase1_compute_stats(
    const char            *input_path,
    long long              N,
    long long              D,
    long long              my_rows,
    long long              my_row_start,
    long long              block_rows,
    std::vector<ColStats> &stats,
    double                &wall_time,
    double                &compute_time,
    double                &comm_time,
    MPI_Comm               comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    /* --- Initialise local accumulators --- */
    for (long long j = 0; j < D; ++j) {
        stats[j].sum     = 0.0;
        stats[j].sum_sq  = 0.0;
        stats[j].min_val =  std::numeric_limits<double>::infinity();
        stats[j].max_val = -std::numeric_limits<double>::infinity();
    }

    /* --- Open the input file collectively (MPI-IO) --- */
    MPI_File fh;
    int rc = MPI_File_open(comm, const_cast<char *>(input_path),
                           MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (rc != MPI_SUCCESS) {
        if (rank == 0)
            std::fprintf(stderr, "[ERROR] Phase 1: Cannot open input '%s'\n",
                         input_path);
        return false;
    }

    std::vector<double> block(static_cast<size_t>(block_rows * D));

    /* Starting byte offset for this process */
    MPI_Offset cur_offset    = (MPI_Offset)my_row_start
                               * (MPI_Offset)D
                               * (MPI_Offset)sizeof(double);
    long long rows_remaining = my_rows;
    compute_time             = 0.0;

    double t_wall_start = MPI_Wtime();

    /* --- Block loop: read → accumulate → advance --- */
    while (rows_remaining > 0) {
        long long rows_this = std::min(block_rows, rows_remaining);
        int       elems     = (int)(rows_this * D);   /* safe: checked in main */

        MPI_Status status;
        int err = MPI_File_read_at(fh, cur_offset,
                                    block.data(), elems,
                                    MPI_DOUBLE, &status);
        if (err != MPI_SUCCESS) {
            std::fprintf(stderr,
                "[ERROR] Phase 1 rank %d: MPI_File_read_at failed "
                "(offset %lld, elems %d)\n",
                rank, (long long)cur_offset, elems);
            MPI_File_close(&fh);
            return false;
        }

        /* ---- Compute inner loop: timed separately ---- */
        double t_c0 = MPI_Wtime();

        for (long long i = 0; i < rows_this; ++i) {
            const double *row = block.data() + i * D;
            for (long long j = 0; j < D; ++j) {
                double v      = row[j];
                stats[j].sum    += v;
                stats[j].sum_sq += v * v;
                if (v < stats[j].min_val) stats[j].min_val = v;
                if (v > stats[j].max_val) stats[j].max_val = v;
            }
        }

        compute_time += MPI_Wtime() - t_c0;
        /* ---------------------------------------------- */

        cur_offset    += (MPI_Offset)rows_this
                         * (MPI_Offset)D
                         * (MPI_Offset)sizeof(double);
        rows_remaining -= rows_this;
    }

    MPI_File_close(&fh);

    /* ----------------------------------------------------------------
     * Allreduce: merge partial statistics from all processes
     * Four separate calls keep the logic clear.
     * Total per-process comm traffic: 4 * D * 8 bytes (e.g., 4 KB for D=128)
     * ---------------------------------------------------------------- */
    double t_comm_start = MPI_Wtime();

    std::vector<double> local_sum(D),    global_sum(D);
    std::vector<double> local_sum_sq(D), global_sum_sq(D);
    std::vector<double> local_min(D),    global_min(D);
    std::vector<double> local_max(D),    global_max(D);

    for (long long j = 0; j < D; ++j) {
        local_sum[j]    = stats[j].sum;
        local_sum_sq[j] = stats[j].sum_sq;
        local_min[j]    = stats[j].min_val;
        local_max[j]    = stats[j].max_val;
    }

    MPI_Allreduce(local_sum.data(),    global_sum.data(),    (int)D,
                  MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(local_sum_sq.data(), global_sum_sq.data(), (int)D,
                  MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(local_min.data(),    global_min.data(),    (int)D,
                  MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(local_max.data(),    global_max.data(),    (int)D,
                  MPI_DOUBLE, MPI_MAX, comm);

    comm_time = MPI_Wtime() - t_comm_start;

    /* --- Derive global mean / variance / std_dev (identical on every rank) --- */
    double inv_N = 1.0 / (double)N;
    for (long long j = 0; j < D; ++j) {
        stats[j].sum     = global_sum[j];
        stats[j].sum_sq  = global_sum_sq[j];
        stats[j].min_val = global_min[j];
        stats[j].max_val = global_max[j];
        stats[j].mean    = global_sum[j] * inv_N;
        stats[j].var     = global_sum_sq[j] * inv_N
                           - stats[j].mean * stats[j].mean;
        if (stats[j].var < 0.0) stats[j].var = 0.0;
        stats[j].std_dev = std::sqrt(stats[j].var);
    }

    /* Reduce wall time: report the slowest process */
    double local_wall = MPI_Wtime() - t_wall_start;
    MPI_Allreduce(&local_wall, &wall_time, 1, MPI_DOUBLE, MPI_MAX, comm);

    return true;
}

/* ------------------------------------------------------------------ */
/*  Phase 2 — re-read partition, apply scaling, write via MPI-IO       */
/*                                                                      */
/*  wall_time    – max across all processes (MPI_Allreduce MAX)         */
/*  compute_time – local scaling-loop time only                         */
/* ------------------------------------------------------------------ */
static bool phase2_scale_and_write(
    const char                  *input_path,
    const char                  *output_path,
    long long                    N,
    long long                    D,
    long long                    my_rows,
    long long                    my_row_start,
    long long                    block_rows,
    ScalerMode                   mode,
    const std::vector<ColStats> &stats,
    double                      &wall_time,
    double                      &compute_time,
    MPI_Comm                     comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    /* --- Pre-compute shift / scale vectors (same on all ranks) --- */
    std::vector<double> shift(D), scale(D);
    for (long long j = 0; j < D; ++j) {
        if (mode == ScalerMode::STANDARD) {
            shift[j] = stats[j].mean;
            scale[j] = (stats[j].std_dev != 0.0)
                       ? 1.0 / stats[j].std_dev : 0.0;
        } else {
            double range = stats[j].max_val - stats[j].min_val;
            shift[j]     = stats[j].min_val;
            scale[j]     = (range != 0.0) ? 1.0 / range : 0.0;
        }
    }

    /* --- Open input file for reading (MPI-IO) --- */
    MPI_File fh_in;
    int rc = MPI_File_open(comm, const_cast<char *>(input_path),
                           MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_in);
    if (rc != MPI_SUCCESS) {
        if (rank == 0)
            std::fprintf(stderr,
                "[ERROR] Phase 2: Cannot open input '%s'\n", input_path);
        return false;
    }

    /* --- Open output file for writing (MPI-IO) --- */
    /*
     * All processes open together with CREATE|WRONLY.
     * Rank 0 immediately sets the file to its exact final size so that:
     *   (a) any stale bytes from a longer previous run are discarded, and
     *   (b) the filesystem can pre-allocate space for better write perf.
     * A Barrier ensures the set_size is visible before any writes begin.
     */
    MPI_File fh_out;
    rc = MPI_File_open(comm, const_cast<char *>(output_path),
                       MPI_MODE_CREATE | MPI_MODE_WRONLY,
                       MPI_INFO_NULL, &fh_out);
    if (rc != MPI_SUCCESS) {
        if (rank == 0)
            std::fprintf(stderr,
                "[ERROR] Phase 2: Cannot open output '%s'\n", output_path);
        MPI_File_close(&fh_in);
        return false;
    }

    if (rank == 0) {
        /*MPI_Offset final_size = (MPI_Offset)N
                                * (MPI_Offset)D
                                * (MPI_Offset)sizeof(double);
        MPI_File_set_size(fh_out, final_size);*/
    }
    MPI_Barrier(comm);   /* all ranks wait before the first write */

    std::vector<double> block(static_cast<size_t>(block_rows * D));

    MPI_Offset cur_offset    = (MPI_Offset)my_row_start
                               * (MPI_Offset)D
                               * (MPI_Offset)sizeof(double);
    long long rows_remaining = my_rows;
    compute_time             = 0.0;

    double t_wall_start = MPI_Wtime();

    /* --- Block loop: read → scale → write at same offset --- */
    while (rows_remaining > 0) {
        long long rows_this = std::min(block_rows, rows_remaining);
        int       elems     = (int)(rows_this * D);

        MPI_Status status;

        /* Read */
        int err = MPI_File_read_at(fh_in, cur_offset,
                                    block.data(), elems,
                                    MPI_DOUBLE, &status);
        if (err != MPI_SUCCESS) {
            std::fprintf(stderr,
                "[ERROR] Phase 2 rank %d: MPI_File_read_at failed\n", rank);
            MPI_File_close(&fh_in);
            MPI_File_close(&fh_out);
            return false;
        }

        /* ---- Scale in-place: timed separately ---- */
        double t_c0 = MPI_Wtime();

        for (long long i = 0; i < rows_this; ++i) {
            double *row = block.data() + i * D;
            for (long long j = 0; j < D; ++j) {
                row[j] = (row[j] - shift[j]) * scale[j];
            }
        }

        compute_time += MPI_Wtime() - t_c0;
        /* ------------------------------------------- */

        /* Write at the same byte offset (preserves row order) */
        err = MPI_File_write_at(fh_out, cur_offset,
                                 block.data(), elems,
                                 MPI_DOUBLE, &status);
        if (err != MPI_SUCCESS) {
            std::fprintf(stderr,
                "[ERROR] Phase 2 rank %d: MPI_File_write_at failed\n", rank);
            MPI_File_close(&fh_in);
            MPI_File_close(&fh_out);
            return false;
        }

        cur_offset    += (MPI_Offset)rows_this
                         * (MPI_Offset)D
                         * (MPI_Offset)sizeof(double);
        rows_remaining -= rows_this;
    }

    MPI_File_close(&fh_in);
    MPI_File_close(&fh_out);

    /* Reduce wall time: report the slowest process */
    double local_wall = MPI_Wtime() - t_wall_start;
    MPI_Allreduce(&local_wall, &wall_time, 1, MPI_DOUBLE, MPI_MAX, comm);

    return true;
}

/* ------------------------------------------------------------------ */
/*  Print statistics summary (call from rank 0 only)                   */
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
                    stats[j].mean, stats[j].std_dev,
                    stats[j].min_val, stats[j].max_val, stats[j].var);
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
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ---- Argument parsing ---- */
    if (argc < 6 || argc > 7) {
        if (rank == 0)
            std::fprintf(stderr,
                "Usage: %s input.bin output.bin N D mode [block_rows]\n"
                "  mode: standard | minmax\n"
                "  block_rows: optional, default = 256000\n",
                argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char *input_path  = argv[1];
    const char *output_path = argv[2];
    long long   N           = std::atoll(argv[3]);
    long long   D           = std::atoll(argv[4]);
    std::string mode_str    = argv[5];
    long long   block_rows  = (argc == 7) ? std::atoll(argv[6]) : 256000LL;

    /* Basic validation */
    if (N <= 0 || D <= 0) {
        if (rank == 0)
            std::fprintf(stderr, "[ERROR] N and D must be positive integers.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (block_rows <= 0) {
        if (rank == 0)
            std::fprintf(stderr, "[ERROR] block_rows must be a positive integer.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Guard against MPI count overflow (int limit for MPI element counts) */
    if (block_rows * D > (long long)std::numeric_limits<int>::max()) {
        if (rank == 0)
            std::fprintf(stderr,
                "[ERROR] block_rows * D = %lld exceeds INT_MAX (%d). "
                "Reduce block_rows.\n",
                block_rows * D, std::numeric_limits<int>::max());
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    ScalerMode mode;
    if      (mode_str == "standard") mode = ScalerMode::STANDARD;
    else if (mode_str == "minmax")   mode = ScalerMode::MINMAX;
    else {
        if (rank == 0)
            std::fprintf(stderr,
                "[ERROR] Unknown mode '%s'. Use 'standard' or 'minmax'.\n",
                mode_str.c_str());
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* ---- Row-partition for this process ---- */
    long long my_rows, my_row_start;
    compute_partition(N, nprocs, rank, my_rows, my_row_start);

    double file_size_gb = (double)N * (double)D * sizeof(double) / (1 << 30);
    double block_mb     = (double)block_rows * (double)D * sizeof(double) / (1 << 20);

    /* ---- Print header and partition table (rank 0) ---- */
    if (rank == 0) {
        std::printf("=== MPI Scaler ===\n");
        std::printf("  Processes:   %d\n",         nprocs);
        std::printf("  Input:       %s\n",          input_path);
        std::printf("  Output:      %s\n",          output_path);
        std::printf("  N (rows):    %lld\n",        N);
        std::printf("  D (cols):    %lld\n",        D);
        std::printf("  Mode:        %s\n",          mode_str.c_str());
        std::printf("  Block rows:  %lld  (≈ %.1f MB per block)\n",
                    block_rows, block_mb);
        std::printf("  File size:   ≈ %.3f GB\n\n", file_size_gb);
        std::printf("  Row partition per process:\n");
    }

    /* Sequential print so output is not interleaved */
    for (int r = 0; r < nprocs; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r) {
            long long rows_r, start_r;
            compute_partition(N, nprocs, r, rows_r, start_r);
            std::printf("    Rank %3d: rows [%lld, %lld)  (%lld rows)\n",
                        r, start_r, start_r + rows_r, rows_r);
            std::fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("\n");

    std::vector<ColStats> stats(static_cast<size_t>(D));

    /* ================================================================
     * PHASE 1 — compute per-column statistics
     * ================================================================ */
    if (rank == 0)
        std::printf("[Phase 1] Computing per-column statistics "
                    "(MPI-IO reads + MPI_Allreduce)...\n");

    double wall1 = 0.0, compute1 = 0.0, comm1 = 0.0;
    if (!phase1_compute_stats(input_path, N, D,
                               my_rows, my_row_start, block_rows,
                               stats, wall1, compute1, comm1,
                               MPI_COMM_WORLD)) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        std::printf("[Phase 1] Done in %.3f s  "
                    "(compute %.3f s, Allreduce %.3f s, I/O %.3f s)\n",
                    wall1, compute1, comm1,
                    wall1 - compute1 - comm1);
        print_stats(stats, D);
    }

    /* ================================================================
     * PHASE 2 — apply scaling and write output
     * ================================================================ */
    if (rank == 0)
        std::printf("[Phase 2] Applying %s and writing output "
                    "(MPI-IO writes)...\n", mode_str.c_str());

    double wall2 = 0.0, compute2 = 0.0;
    if (!phase2_scale_and_write(input_path, output_path, N, D,
                                 my_rows, my_row_start, block_rows,
                                 mode, stats, wall2, compute2,
                                 MPI_COMM_WORLD)) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        std::printf("[Phase 2] Done in %.3f s  "
                    "(compute %.3f s, I/O %.3f s)\n",
                    wall2, compute2, wall2 - compute2);
    }

    /* ---- Timing summary (rank 0) ---- */
    if (rank == 0) {
        double total = wall1 + wall2;
        std::printf("\n=== Timing Summary ===\n");
        std::printf("  Processes:              %d\n", nprocs);
        std::printf("  Phase 1 Wall Time:      %.3f s"
                    "  (Compute: %.3f s, Allreduce: %.3f s, I/O: %.3f s)\n",
                    wall1, compute1, comm1,
                    wall1 - compute1 - comm1);
        std::printf("  Phase 2 Wall Time:      %.3f s"
                    "  (Compute: %.3f s, I/O: %.3f s)\n",
                    wall2, compute2, wall2 - compute2);
        std::printf("  -----------------------------------------------\n");
        std::printf("  Total Compute Time:     %.3f s\n", compute1 + compute2);
        std::printf("  Total Execution Time:   %.3f s\n",  total);
        std::printf("  Throughput (3x file):   %.2f GB/s\n",
                    3.0 * file_size_gb / total);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
