#!/usr/bin/env bash
# run_benchmarks.sh
#
# Αυτοματοποιεί τα πειράματα scaling OpenMP και MPI για την αναφορά.
# Εκτελεί και τα δύο datasets (Medium 5M×64, Large 10M×128) και τους
# δύο τρόπους κανονικοποίησης (standard, minmax).
#
# Προϋποθέσεις:
#   - make build έχει εκτελεστεί ήδη
#   - Τα αρχεία δεδομένων υπάρχουν ήδη (τρέξτε make gen-data πρώτα)
#
# Χρήση:
#   chmod +x run_benchmarks.sh
#   ./run_benchmarks.sh
#
# Αποτελέσματα: εκτυπώνονται στο stdout και αποθηκεύονται στο benchmark_results.log

set -euo pipefail

LOG="benchmark_results.log"
BLOCKS=256000

# Datasets
MEDIUM_N=5000000;  MEDIUM_D=64;  MEDIUM_INPUT="data_5000000_64.bin"
LARGE_N=10000000;  LARGE_D=128;  LARGE_INPUT="data_10000000_128.bin"

# OpenMP thread counts
OMP_THREAD_COUNTS=(1 2 4 8 16 24)

# MPI process counts
MPI_PROCESS_COUNTS=(1 2 4 8 16 24)

MODES=("standard" "minmax")

# ────────────────────────────────────────────────────────────────
log() { echo "$*" | tee -a "$LOG"; }
separator() { log ""; log "$(printf '=%.0s' {1..70})"; log ""; }
# ────────────────────────────────────────────────────────────────

# Check that binaries and data exist before starting
check_binary() {
    if [[ ! -f "$1" ]]; then
        echo "[ERROR] Binary '$1' not found. Run 'make build' first." >&2
        exit 1
    fi
}
check_data() {
    if [[ ! -f "$1" ]]; then
        echo "[ERROR] Data file '$1' not found. Run 'make gen-data' with appropriate N/D first." >&2
        exit 1
    fi
}

check_binary ./scaler_serial
check_binary ./scaler_simd
check_binary ./scaler_openmp
check_binary ./scaler_mpi

check_data "$MEDIUM_INPUT"
check_data "$LARGE_INPUT"

# Start fresh log
echo "" > "$LOG"
log "Benchmark run: $(date)"
log "Host: $(hostname)"

# ════════════════════════════════════════════════════════════════
# SERIAL + SIMD baselines
# ════════════════════════════════════════════════════════════════
separator
log ">>> SERIAL & SIMD BASELINES"
separator

for MODE in "${MODES[@]}"; do
    for DATASET in medium large; do
        if [[ "$DATASET" == "medium" ]]; then
            N=$MEDIUM_N; D=$MEDIUM_D; INPUT=$MEDIUM_INPUT
        else
            N=$LARGE_N; D=$LARGE_D; INPUT=$LARGE_INPUT
        fi

        OUT="out_serial_${DATASET}_${MODE}.bin"
        log "--- Serial | ${DATASET^^} (${N}×${D}) | ${MODE} ---"
        ./scaler_serial "$INPUT" "$OUT" "$N" "$D" "$MODE" "$BLOCKS" 2>&1 | tee -a "$LOG"
        log ""

        OUT="out_simd_${DATASET}_${MODE}.bin"
        log "--- SIMD   | ${DATASET^^} (${N}×${D}) | ${MODE} ---"
        ./scaler_simd "$INPUT" "$OUT" "$N" "$D" "$MODE" "$BLOCKS" 2>&1 | tee -a "$LOG"
        log ""
    done
done

# ════════════════════════════════════════════════════════════════
# OPENMP SCALING
# ════════════════════════════════════════════════════════════════
separator
log ">>> OPENMP SCALING"
separator

for MODE in "${MODES[@]}"; do
    for DATASET in medium large; do
        if [[ "$DATASET" == "medium" ]]; then
            N=$MEDIUM_N; D=$MEDIUM_D; INPUT=$MEDIUM_INPUT
        else
            N=$LARGE_N; D=$LARGE_D; INPUT=$LARGE_INPUT
        fi

        for T in "${OMP_THREAD_COUNTS[@]}"; do
            OUT="out_openmp_${DATASET}_${MODE}_t${T}.bin"
            log "--- OpenMP | ${DATASET^^} (${N}×${D}) | ${MODE} | threads=${T} ---"
            OMP_NUM_THREADS=$T ./scaler_openmp \
                "$INPUT" "$OUT" "$N" "$D" "$MODE" "$BLOCKS" 2>&1 | tee -a "$LOG"
            log ""
        done
    done
done

# ════════════════════════════════════════════════════════════════
# MPI SCALING
# ════════════════════════════════════════════════════════════════
separator
log ">>> MPI SCALING"
separator

for MODE in "${MODES[@]}"; do
    for DATASET in medium large; do
        if [[ "$DATASET" == "medium" ]]; then
            N=$MEDIUM_N; D=$MEDIUM_D; INPUT=$MEDIUM_INPUT
        else
            N=$LARGE_N; D=$LARGE_D; INPUT=$LARGE_INPUT
        fi

        for P in "${MPI_PROCESS_COUNTS[@]}"; do
            OUT="out_mpi_${DATASET}_${MODE}_p${P}.bin"
            log "--- MPI    | ${DATASET^^} (${N}×${D}) | ${MODE} | processes=${P} ---"
            mpirun -np "$P" ./scaler_mpi \
                "$INPUT" "$OUT" "$N" "$D" "$MODE" "$BLOCKS" 2>&1 | tee -a "$LOG"
            log ""
        done
    done
done

# ════════════════════════════════════════════════════════════════
# CUDA (αν υπάρχει το εκτελέσιμο)
# ════════════════════════════════════════════════════════════════
if [[ -f "./scaler_cuda" ]]; then
    separator
    log ">>> CUDA"
    separator

    for MODE in "${MODES[@]}"; do
        for DATASET in medium large; do
            if [[ "$DATASET" == "medium" ]]; then
                N=$MEDIUM_N; D=$MEDIUM_D; INPUT=$MEDIUM_INPUT
            else
                N=$LARGE_N; D=$LARGE_D; INPUT=$LARGE_INPUT
            fi

            OUT="out_cuda_${DATASET}_${MODE}.bin"
            log "--- CUDA   | ${DATASET^^} (${N}×${D}) | ${MODE} ---"
            ./scaler_cuda "$INPUT" "$OUT" "$N" "$D" "$MODE" "$BLOCKS" 2>&1 | tee -a "$LOG"
            log ""
        done
    done
else
    log "[INFO] scaler_cuda not found, skipping CUDA benchmarks."
fi

separator
log "All benchmarks complete. Results saved to: $LOG"
separator
