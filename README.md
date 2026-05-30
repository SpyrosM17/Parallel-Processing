# Παράλληλη Προεπεξεργασία Μεγάλου Όγκου Δεδομένων

**Μάθημα:** Παράλληλη Επεξεργασία — Εαρινό Εξάμηνο 2025-2026  
**Τμήμα:** Μηχανικών Η/Υ και Πληροφορικής, Πανεπιστήμιο Πατρών  
**Ομάδα:** Μανταδάκης Σπυρίδων (1100613) · Ιωάννης Κώτσαλος (1100603)

---

## Περιεχόμενα αρχείων

| Αρχείο | Ρόλος |
|---|---|
| `Scaler_Serial.cpp` | Σειριακή έκδοση αναφοράς |
| `Scaler_SIMD.cpp` | SIMD έκδοση με AVX2 intrinsics |
| `Scaler_OpenMP.cpp` | Πολυνηματική έκδοση με OpenMP |
| `Scaler_MPI.cpp` | Κατανεμημένη έκδοση με MPI |
| `Scaler_CUDA.cu` | GPU έκδοση με CUDA |
| `generate_data_chunks.py` | Παραγωγή συνθετικών δεδομένων |
| `Verifier.py` | Έλεγχος ορθότητας εξόδου |
| `Makefile` | Build, εκτέλεση, verification |
| `run_benchmarks.sh` | Αυτοματοποιημένα πειράματα scaling |

---

## Απαιτήσεις

### Για το cluster (krylov100 / scgroup3.ceid.upatras.gr)
- `g++` με C++17 και AVX2 υποστήριξη
- `mpicxx` / `mpirun`
- `/usr/local/cuda-12.2/bin/nvcc` (sm_70 για Tesla V100)
- Python 3 με `numpy` και `scikit-learn`

### Για τοπική ανάπτυξη (Linux / macOS)
- macOS: `brew install libomp open-mpi`
- Η CUDA έκδοση απαιτεί κατάλληλη GPU και CUDA toolkit

---

## Μεταγλώττιση

### Όλες οι εκδόσεις μαζί
```bash
make build
```

### Μεμονωμένες εκδόσεις
```bash
make scaler_serial
make scaler_simd
make scaler_openmp
make scaler_mpi
make scaler_cuda
```

### Άμεση μεταγλώττιση χωρίς Makefile
```bash
g++ -std=c++17 -O2 -Wall -o scaler_serial Scaler_Serial.cpp
g++ -std=c++17 -O3 -Wall -mavx2 -mfma -march=native -pthread -o scaler_simd Scaler_SIMD.cpp
g++ -std=c++17 -O3 -Wall -fopenmp -o scaler_openmp Scaler_OpenMP.cpp
mpicxx -std=c++17 -O3 -Wall -o scaler_mpi Scaler_MPI.cpp
/usr/local/cuda-12.2/bin/nvcc -std=c++17 -O3 -arch=sm_70 -o scaler_cuda Scaler_CUDA.cu
```

---

## Παραγωγή δεδομένων

Το `generate_data_chunks.py` παράγει raw binary αρχεία με συνθετικά δεδομένα σε chunks, ώστε να αποφεύγεται υπερφόρτωση μνήμης.

```bash
# Μέσω Makefile (default: N=5000000, D=64)
make gen-data

# Παραμετρικά
make gen-data N=1000000  D=32  INPUT_DATA=data_1000000_32.bin   DTYPE=float64 SEED=42
make gen-data N=5000000  D=64  INPUT_DATA=data_5000000_64.bin   DTYPE=float64 SEED=42
make gen-data N=10000000 D=128 INPUT_DATA=data_10000000_128.bin DTYPE=float64 SEED=42
```

```bash
# Άμεσα με Python
python3 generate_data_chunks.py \
    --samples 10000000 \
    --features 128 \
    --output data_10000000_128.bin \
    --dtype float64 \
    --seed 42
```

---

## Εκτέλεση εκδόσεων

Κοινές παράμετροι: `input.bin output.bin N D mode [block_rows]`  
`mode`: `standard` (StandardScaler) ή `minmax` (MinMaxScaler)

### Σειριακή
```bash
make run-serial N=5000000 D=64 INPUT_DATA=data_5000000_64.bin \
    OUT_DATA=out_serial.bin MODE=standard BLOCKS=256000

# Ή άμεσα
./scaler_serial data_5000000_64.bin out_serial.bin 5000000 64 standard 256000
```

### SIMD
```bash
make run-simd N=5000000 D=64 INPUT_DATA=data_5000000_64.bin \
    OUT_DATA=out_simd.bin MODE=standard BLOCKS=256000

./scaler_simd data_5000000_64.bin out_simd.bin 5000000 64 standard 256000
```

### OpenMP
```bash
# OMP_THREADS ελέγχει τον αριθμό νημάτων
make run-openmp OMP_THREADS=8 N=5000000 D=64 INPUT_DATA=data_5000000_64.bin \
    OUT_DATA=out_openmp.bin MODE=standard BLOCKS=256000

OMP_NUM_THREADS=8 ./scaler_openmp data_5000000_64.bin out_openmp.bin 5000000 64 standard 256000
```

### MPI
```bash
# NP ελέγχει τον αριθμό processes
make run-mpi NP=8 N=5000000 D=64 INPUT_DATA=data_5000000_64.bin \
    OUT_DATA=out_mpi.bin MODE=standard BLOCKS=256000

mpirun -np 8 ./scaler_mpi data_5000000_64.bin out_mpi.bin 5000000 64 standard 256000
```

### CUDA
```bash
make run-cuda N=5000000 D=64 INPUT_DATA=data_5000000_64.bin \
    OUT_DATA=out_cuda.bin MODE=standard BLOCKS=256000

./scaler_cuda data_5000000_64.bin out_cuda.bin 5000000 64 standard 256000
```

---

## Έλεγχος ορθότητας

Το `Verifier.py` συγκρίνει streaming το output αρχείο με το reference της NumPy και αναφέρει max/mean absolute error.

```bash
# Μέσω Makefile
make verify N=5000000 D=64 INPUT_DATA=data_5000000_64.bin \
    OUT_DATA=out_serial.bin MODE=standard DTYPE=float64 BLOCKS=256000

# Άμεσα
python3 Verifier.py \
    --input  data_5000000_64.bin \
    --cpp-output out_serial.bin \
    --N 5000000 --D 64 \
    --mode standard \
    --dtype float64 \
    --block-rows 256000
```

Το script εκτυπώνει `PASS` αν το max absolute error είναι κάτω από `1e-9`.

---

## Αυτοματοποιημένα πειράματα scaling

Το `run_benchmarks.sh` εκτελεί όλα τα πειράματα OpenMP και MPI scaling που παρουσιάζονται στην αναφορά.

```bash
chmod +x run_benchmarks.sh
./run_benchmarks.sh
```

Αναμένει ότι τα αρχεία δεδομένων υπάρχουν ήδη (δημιουργήστε τα πρώτα με `make gen-data`).  
Τα αποτελέσματα εκτυπώνονται στο stdout και αποθηκεύονται στο `benchmark_results.log`.

---

## Παράμετροι Makefile

| Μεταβλητή | Default | Περιγραφή |
|---|---|---|
| `N` | `5000000` | Αριθμός γραμμών |
| `D` | `64` | Αριθμός στηλών |
| `MODE` | `standard` | `standard` ή `minmax` |
| `BLOCKS` | `256000` | Γραμμές ανά block |
| `INPUT_DATA` | `data_N_D.bin` | Αρχείο εισόδου |
| `OUT_DATA` | `out_N_D_MODE.bin` | Αρχείο εξόδου |
| `NP` | `1` | Αριθμός MPI processes |
| `OMP_THREADS` | `1` | Αριθμός OpenMP νημάτων |
| `DTYPE` | `float64` | Τύπος δεδομένων για verify |
| `SEED` | `42` | Seed για παραγωγή δεδομένων |

---

## Καθαρισμός

```bash
make clean
```

Διαγράφει τα εκτελέσιμα και τα `out_*.bin` αρχεία. Τα αρχεία δεδομένων εισόδου **δεν** διαγράφονται.
