PYTHON     ?= python3
GEN_SCRIPT ?= generate_data.py
CXX        ?= g++
CXXFLAGS   ?= -std=c++17 -O2 -Wall
CXXFLAGS_SIMD ?= -std=c++17 -O3 -march=haswell -ffast-math -Wall

# Data generation parameters (override from CLI)
N     ?= 1000
D     ?= 12
DTYPE ?= float64
SEED  ?= 42
NOISE ?= 0.1
OUTPUT ?= data_10M_128.bin

.PHONY: help build build-simd gen-data verify clean

help:
	@echo "=== Parallel Processing Scaler Build System ==="
	@echo ""
	@echo "Targets:"
	@echo "  make build              # Build serial scaler"
	@echo "  make build-simd         # Build SIMD-optimized scaler (AVX2)"
	@echo "  make gen-data           # Generate test data"
	@echo "  make verify             # Verify output against sklearn"
	@echo "  make clean              # Remove binaries and .bin files"
	@echo ""
	@echo "Common overrides:"
	@echo "  N=1000000 D=32 DTYPE=float64 OUTPUT=data_1M_32.bin SEED=42 NOISE=0.1"
	@echo ""
	@echo "Examples:"
	@echo "  1. Generate 1M samples with 32 features:"
	@echo "     make gen-data N=1000000 D=32 OUTPUT=data_1M_32.bin"
	@echo ""
	@echo "  2. Build and run serial scaler:"
	@echo "     make build"
	@echo "     ./scaler data_1M_32.bin output_serial.bin 1000000 32 standard"
	@echo ""
	@echo "  3. Build and run SIMD scaler:"
	@echo "     make build-simd"
	@echo "     ./scaler_simd data_1M_32.bin output_simd.bin 1000000 32 standard"
	@echo ""
	@echo "  4. Verify correctness:"
	@echo "     python3 Verifier.py --input data_1M_32.bin --cpp-output output_serial.bin \\"
	@echo "                         --N 1000000 --D 32 --mode standard"

build: scaler

scaler: Scaler.cpp
	$(CXX) $(CXXFLAGS) -o scaler Scaler.cpp

build-simd: scaler_simd

scaler_simd: Scaler_SIMD.cpp
	$(CXX) $(CXXFLAGS_SIMD) -o scaler_simd Scaler_SIMD.cpp

gen-data:
	$(PYTHON) $(GEN_SCRIPT) \
		--samples $(N) \
		--features $(D) \
		--output $(OUTPUT) \
		--dtype $(DTYPE) \
		--seed $(SEED) \
		--noise $(NOISE)

verify:
	@echo "Verify requires input and output files. Example:"
	@echo "  python3 Verifier.py --input data.bin --cpp-output output.bin \\"
	@echo "                      --N <rows> --D <cols> --mode standard"

clean:
	rm -f *.bin scaler scaler_simd
