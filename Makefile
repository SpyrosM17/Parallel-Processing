PYTHON     ?= python3
GEN_SCRIPT ?= generate_data_chunks.py
CXX        ?= g++
MPICXX     ?= mpicxx

# Flags for the Serial version (Standard O2 optimization)
CXXFLAGS_SERIAL ?= -std=c++17 -O2 -Wall

# Flags for the SIMD version (AVX2, FMA, O3 optimization, pthread for std::async)
CXXFLAGS_SIMD ?= -std=c++17 -O3 -Wall -mavx2 -mfma -march=native -pthread

# Flags for the OpenMP version
CXXFLAGS_OMP  ?= -std=c++17 -O3 -Wall -fopenmp

# Flags for the OpenMP optimized version (double-buffering)
CXXFLAGS_OMP_OPT ?= -std=c++17 -O3 -Wall -fopenmp -pthread

# Flags for the MPI version
CXXFLAGS_MPI  ?= -std=c++17 -O3 -Wall

# Data generation parameters (override from CLI)
N     ?= 5000000
D     ?= 64
DTYPE ?= float64
SEED  ?= 42
NOISE ?= 0.1
INPUT_DATA ?= data_$(N)_$(D).bin

# Run parameters
MODE   ?= standard
OUT_DATA ?= out_$(N)_$(D)_$(MODE).bin
BLOCKS ?= 500000

# MPI processes
NP ?= 6

#OpenMP Threads
OMP_THREADS ?= 6

.PHONY: help build gen-data clean \
        run-serial run-simd run-openmp run-mpi verify

help:
	@echo "Targets:"
	@echo "  make build           - Builds serial, SIMD, OpenMP and MPI executables"
	@echo "  make scaler_serial   - Builds only the serial version"
	@echo "  make scaler_simd     - Builds only the SIMD version"
	@echo "  make scaler_openmp   - Builds only the OpenMP version"
	@echo "  make scaler_mpi      - Builds only the MPI version"
	@echo "  make gen-data        - Generates the input binary dataset"
	@echo "  make run-serial      - Runs the serial executable"
	@echo "  make run-simd        - Runs the SIMD executable"
	@echo "  make run-openmp      - Runs the OpenMP executable"
	@echo "  make run-mpi         - Runs the MPI executable (NP processes)"
	@echo "  make verify          - Verifies the scaler output against scikit-learn"
	@echo "  make clean           - Removes executables and output .bin files"
	@echo ""
	@echo "Example executions:"
	@echo "  make run-mpi NP=4 N=1000000 D=32 INPUT_DATA=data_1000000_32.bin MODE=standard"
	@echo "  make run-simd N=1000000 D=32 INPUT_DATA=data_1000000_32.bin MODE=standard BLOCKS=256000"

build: scaler_serial scaler_simd scaler_openmp scaler_mpi

scaler_serial: Scaler_Serial.cpp
	$(CXX) $(CXXFLAGS_SERIAL) -o scaler_serial Scaler_Serial.cpp

scaler_simd: Scaler_SIMD.cpp
	$(CXX) $(CXXFLAGS_SIMD) -o scaler_simd Scaler_SIMD.cpp

scaler_openmp: Scaler_OpenMP.cpp
	$(CXX) $(CXXFLAGS_OMP) -o scaler_openmp Scaler_OpenMP.cpp

scaler_mpi: Scaler_MPI.cpp
	$(MPICXX) $(CXXFLAGS_MPI) -o scaler_mpi Scaler_MPI.cpp

gen-data:
	$(PYTHON) $(GEN_SCRIPT) \
		--samples $(N) \
		--features $(D) \
		--output $(INPUT_DATA) \
		--dtype $(DTYPE) \
		--seed $(SEED) \
		--noise $(NOISE)

run-serial: scaler_serial
	./scaler_serial $(INPUT_DATA) $(OUT_DATA) $(N) $(D) $(MODE) $(BLOCKS)

run-simd: scaler_simd
	./scaler_simd $(INPUT_DATA) $(OUT_DATA) $(N) $(D) $(MODE) $(BLOCKS)

run-openmp: scaler_openmp
	OMP_NUM_THREADS=$(OMP_THREADS) ./scaler_openmp  $(INPUT_DATA) $(OUT_DATA) $(N) $(D) $(MODE) $(BLOCKS)

run-mpi: scaler_mpi
	mpirun -np $(NP) ./scaler_mpi $(INPUT_DATA) $(OUT_DATA) $(N) $(D) $(MODE) $(BLOCKS)

verify:
	$(PYTHON) Verifier.py \
		--input $(INPUT_DATA) \
		--cpp-output $(OUT_DATA) \
		--N $(N) \
		--D $(D) \
		--mode $(MODE) \
		--dtype $(DTYPE) \
		--block-rows $(BLOCKS)

clean:
	rm -f scaler_serial scaler_simd scaler_openmp scaler_mpi out_*.bin