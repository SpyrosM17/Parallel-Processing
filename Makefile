PYTHON ?= python3
GEN_SCRIPT ?= generate_data.py

# Data generation parameters (override from CLI)
N ?= 1000
D ?= 12
DTYPE ?= float64
SEED ?= 42
NOISE ?= 0.1
OUTPUT ?= data_10M_128.bin

.PHONY: help gen-data clean

help:
	@echo "Targets:"
	@echo "  make gen-data"
	@echo "  make clean"
	@echo ""
	@echo "Common overrides:"
	@echo "  N=1000000 D=32 DTYPE=float64 OUTPUT=data_1M_32.bin SEED=42 NOISE=0.1"
	@echo ""
	@echo "Example:"
	@echo "  make gen-data N=1000000 D=32 OUTPUT=data_1M_32.bin"

gen-data:
	$(PYTHON) $(GEN_SCRIPT) \
		--samples $(N) \
		--features $(D) \
		--output $(OUTPUT) \
		--dtype $(DTYPE) \
		--seed $(SEED) \
		--noise $(NOISE)

clean:
	rm -f *.bin
