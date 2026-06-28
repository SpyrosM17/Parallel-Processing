# Parallel Preprocessing of Large-Scale Data

**Course:** Parallel Processing — Spring Semester 2025–2026
**Department:** Department of Computer Engineering and Informatics, University of Patras
**Team:** Spyridon Mantadakis (1100613) · Ioannis Kotsalos (1100603)

---

# Project Contents

| File                      | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| `Scaler_Serial.cpp`       | Reference serial implementation               |
| `Scaler_SIMD.cpp`         | SIMD implementation using AVX2 intrinsics     |
| `Scaler_OpenMP.cpp`       | Multithreaded implementation using OpenMP     |
| `Scaler_MPI.cpp`          | Distributed implementation using MPI          |
| `Scaler_CUDA.cu`          | GPU implementation using CUDA                 |
| `generate_data_chunks.py` | Synthetic dataset generator                   |
| `Verifier.py`             | Output correctness verification               |
| `Makefile`                | Build, execution, and verification automation |
| `run_benchmarks.sh`       | Automated OpenMP and MPI scaling experiments  |

---

# Requirements

## Cluster Environment (krylov100 / scgroup3.ceid.upatras.gr)

* `g++` with C++17 and AVX2 support
* `mpicxx` / `mpirun`
* `/usr/local/cuda-12.2/bin/nvcc` (compiled for `sm_70`, Tesla V100)
* Python 3 with:

  * `numpy`
  * `scikit-learn`

## Local Development (Linux / macOS)

* **macOS:** Install dependencies using:

```bash
brew install libomp open-mpi
```

* The CUDA implementation requires a compatible NVIDIA GPU and CUDA Toolkit.

---

# Compilation

## Build All Implementations

```bash
make build
```

## Build Individual Implementations

```bash
make scaler_serial
make scaler_simd
make scaler_openmp
make scaler_mpi
make scaler_cuda
```

## Manual Compilation

```bash
g++ -std=c++17 -O2 -Wall -o scaler_serial Scaler_Serial.cpp

g++ -std=c++17 -O3 -Wall -mavx2 -mfma -march=native -pthread \
    -o scaler_simd Scaler_SIMD.cpp

g++ -std=c++17 -O3 -Wall -fopenmp \
    -o scaler_openmp Scaler_OpenMP.cpp

mpicxx -std=c++17 -O3 -Wall \
    -o scaler_mpi Scaler_MPI.cpp

/usr/local/cuda-12.2/bin/nvcc -std=c++17 -O3 -arch=sm_70 \
    -o scaler_cuda Scaler_CUDA.cu
```

---

# Dataset Generation

The `generate_data_chunks.py` script generates large synthetic datasets in raw binary format using chunked processing, avoiding excessive memory usage during generation.

## Using the Makefile

Default configuration:

* `N = 5,000,000`
* `D = 64`

```bash
make gen-data
```

Custom configurations:

```bash
make gen-data N=1000000  D=32  INPUT_DATA=data_1000000_32.bin   DTYPE=float64 SEED=42

make gen-data N=5000000  D=64  INPUT_DATA=data_5000000_64.bin   DTYPE=float64 SEED=42

make gen-data N=10000000 D=128 INPUT_DATA=data_10000000_128.bin DTYPE=float64 SEED=42
```

## Running the Python Script Directly

```bash
python3 generate_data_chunks.py \
    --samples 10000000 \
    --features 128 \
    --output data_10000000_128.bin \
    --dtype float64 \
    --seed 42
```

---

# Running the Implementations

Common command-line arguments:

```
input.bin output.bin N D mode [block_rows]
```

where

* `mode = standard` → StandardScaler
* `mode = minmax` → MinMaxScaler

---

## Serial Version

```bash
make run-serial \
    N=5000000 D=64 \
    INPUT_DATA=data_5000000_64.bin \
    OUT_DATA=out_serial.bin \
    MODE=standard \
    BLOCKS=256000
```

or

```bash
./scaler_serial \
    data_5000000_64.bin \
    out_serial.bin \
    5000000 \
    64 \
    standard \
    256000
```

---

## SIMD Version

```bash
make run-simd \
    N=5000000 D=64 \
    INPUT_DATA=data_5000000_64.bin \
    OUT_DATA=out_simd.bin \
    MODE=standard \
    BLOCKS=256000
```

or

```bash
./scaler_simd \
    data_5000000_64.bin \
    out_simd.bin \
    5000000 \
    64 \
    standard \
    256000
```

---

## OpenMP Version

The number of threads is controlled through `OMP_THREADS`.

```bash
make run-openmp \
    OMP_THREADS=8 \
    N=5000000 D=64 \
    INPUT_DATA=data_5000000_64.bin \
    OUT_DATA=out_openmp.bin \
    MODE=standard \
    BLOCKS=256000
```

or

```bash
OMP_NUM_THREADS=8 \
./scaler_openmp \
    data_5000000_64.bin \
    out_openmp.bin \
    5000000 \
    64 \
    standard \
    256000
```

---

## MPI Version

The number of MPI processes is controlled through `NP`.

```bash
make run-mpi \
    NP=8 \
    N=5000000 D=64 \
    INPUT_DATA=data_5000000_64.bin \
    OUT_DATA=out_mpi.bin \
    MODE=standard \
    BLOCKS=256000
```

or

```bash
mpirun -np 8 \
./scaler_mpi \
    data_5000000_64.bin \
    out_mpi.bin \
    5000000 \
    64 \
    standard \
    256000
```

---

## CUDA Version

```bash
make run-cuda \
    N=5000000 D=64 \
    INPUT_DATA=data_5000000_64.bin \
    OUT_DATA=out_cuda.bin \
    MODE=standard \
    BLOCKS=256000
```

or

```bash
./scaler_cuda \
    data_5000000_64.bin \
    out_cuda.bin \
    5000000 \
    64 \
    standard \
    256000
```

---

# Correctness Verification

`Verifier.py` compares the generated output against the NumPy reference implementation using streaming I/O and reports:

* Maximum absolute error
* Mean absolute error

## Using the Makefile

```bash
make verify \
    N=5000000 \
    D=64 \
    INPUT_DATA=data_5000000_64.bin \
    OUT_DATA=out_serial.bin \
    MODE=standard \
    DTYPE=float64 \
    BLOCKS=256000
```

## Running Directly

```bash
python3 Verifier.py \
    --input data_5000000_64.bin \
    --cpp-output out_serial.bin \
    --N 5000000 \
    --D 64 \
    --mode standard \
    --dtype float64 \
    --block-rows 256000
```

The script prints **`PASS`** when the maximum absolute error is below:

```text
1e-9
```

---

# Automated Scaling Benchmarks

The `run_benchmarks.sh` script executes all OpenMP and MPI scaling experiments presented in the accompanying report.

```bash
chmod +x run_benchmarks.sh

./run_benchmarks.sh
```

The script assumes that the required input datasets have already been generated (using `make gen-data`).

Benchmark results are:

* printed to the terminal (`stdout`),
* saved to `benchmark_results.log`.

---

# Makefile Parameters

| Variable      | Default            | Description                             |
| ------------- | ------------------ | --------------------------------------- |
| `N`           | `5000000`          | Number of samples (rows)                |
| `D`           | `64`               | Number of features (columns)            |
| `MODE`        | `standard`         | Scaling method (`standard` or `minmax`) |
| `BLOCKS`      | `256000`           | Number of rows processed per block      |
| `INPUT_DATA`  | `data_N_D.bin`     | Input dataset                           |
| `OUT_DATA`    | `out_N_D_MODE.bin` | Output file                             |
| `NP`          | `1`                | Number of MPI processes                 |
| `OMP_THREADS` | `1`                | Number of OpenMP threads                |
| `DTYPE`       | `float64`          | Data type used during verification      |
| `SEED`        | `42`               | Random seed for dataset generation      |

---

# Cleaning the Project

```bash
make clean
```

This command removes:

* all compiled executables,
* all `out_*.bin` output files.

The generated input datasets (`data_*.bin`) are **not** deleted.
