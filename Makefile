PYTHON     ?= python3
GEN_SCRIPT ?= generate_data_chunks.py
CXX        ?= g++

# Flags for the SIMD version (AVX2, FMA, O3 optimization, pthread for std::async)
CXXFLAGS_SIMD ?= -std=c++17 -O3 -Wall -mavx2 -mfma -march=native -pthread

# Flags for the Serial version (Standard O2 optimization)
CXXFLAGS_SERIAL ?= -std=c++17 -O2 -Wall

# Data generation parameters (override from CLI)
N     ?= 1000000
D     ?= 32
DTYPE ?= float64
SEED  ?= 42
NOISE ?= 0.1
INPUT_DATA ?= data_10M_128.bin

# Run parameters
MODE ?= standard
OUT_DATA ?= out_$(MODE).bin
BLOCKS ?= 50000

.PHONY: help build gen-data clean run-serial run-simd verify

help:
	@echo "Targets:"
	@echo "  make build       - Builds both serial and SIMD executables"
	@echo "  make scaler      - Builds only the serial version"
	@echo "  make scaler_simd - Builds only the SIMD version"
	@echo "  make gen-data    - Generates the input binary dataset"
	@echo "  make run-serial  - Runs the serial executable"
	@echo "  make run-simd    - Runs the SIMD executable"
	@echo "  make verify      - Verifies the scaler output against scikit-learn"
	@echo "  make clean       - Removes executables and output .bin files"
	@echo ""
	@echo "Example execution:"
	@echo "  make run-simd N=1000 D=12 INPUT_DATA=data_10M_128.bin MODE=standard BLOCKS=256000"

build: scaler scaler_simd

scaler: Scaler.cpp
	$(CXX) $(CXXFLAGS_SERIAL) -o scaler Scaler.cpp

scaler_simd: Scaler_SIMD.cpp
	$(CXX) $(CXXFLAGS_SIMD) -o scaler_simd Scaler_SIMD.cpp


gen-data:
	$(PYTHON) $(GEN_SCRIPT) \
		--samples $(N) \
		--features $(D) \
		--output $(INPUT_DATA) \
		--dtype $(DTYPE) \
		--seed $(SEED) \
		--noise $(NOISE)

run-serial: scaler
	./scaler $(INPUT_DATA) $(OUT_DATA) $(N) $(D) $(MODE) $(BLOCKS)

run-simd: scaler_simd
	./scaler_simd $(INPUT_DATA) $(OUT_DATA) $(N) $(D) $(MODE) $(BLOCKS)

verify:
	$(PYTHON) Verifier.py \
		--input $(INPUT_DATA) \
		--cpp-output $(OUT_DATA) \
		--N $(N) \
		--D $(D) \
		--mode $(MODE) \
		--dtype $(DTYPE)

clean:
	rm -f scaler scaler_simd out_*.bin