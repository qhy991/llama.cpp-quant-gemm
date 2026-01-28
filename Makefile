# Quantized GEMM from Scratch: A Tutorial
# Compatible with llama.cpp quantization formats

# Compiler settings
NVCC := nvcc
CUDA_ARCH := sm_86  # Change this to match your GPU (sm_75, sm_80, sm_86, sm_89, sm_90, etc.)
NVCC_FLAGS := -O3 -arch=$(CUDA_ARCH) -std=c++17
NVCC_FLAGS += -Xcompiler -Wall -Xcompiler -Wextra
NVCC_FLAGS += -I./include

# Libraries
LIBS := -lcurand

# Optional: Link with llama.cpp for comparison
LLAMA_CPP_DIR := ../llama.cpp
LLAMA_CPP_BUILD := $(LLAMA_CPP_DIR)/build
ifneq ($(wildcard $(LLAMA_CPP_DIR)),)
    NVCC_FLAGS += -DHAS_LLAMA_CPP
    NVCC_FLAGS += -I$(LLAMA_CPP_DIR)/ggml/include
    NVCC_FLAGS += -I$(LLAMA_CPP_DIR)/ggml/src
    LIBS += -L$(LLAMA_CPP_BUILD)/ggml/src -lggml
endif

# Directories
SRC_DIR := tests
BUILD_DIR := build
BIN_DIR := bin

# Source files
SOURCES := $(wildcard $(SRC_DIR)/step*.cu)
TARGETS := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(SOURCES))

# Default target
.PHONY: all
all: $(BIN_DIR) $(TARGETS)

# Create directories
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build rules
$(BIN_DIR)/step1_fp32_gemm: $(SRC_DIR)/step1_fp32_gemm.cu include/*.h include/*.cuh
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

$(BIN_DIR)/step2_quantization: $(SRC_DIR)/step2_quantization.cu include/*.h include/*.cuh
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

$(BIN_DIR)/step3_w4a16_gemm: $(SRC_DIR)/step3_w4a16_gemm.cu include/*.h include/*.cuh
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

$(BIN_DIR)/step4_w4a8_gemm: $(SRC_DIR)/step4_w4a8_gemm.cu include/*.h include/*.cuh
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

$(BIN_DIR)/step5_llama_comparison: $(SRC_DIR)/step5_llama_comparison.cu include/*.h include/*.cuh
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

# Run all tests
.PHONY: test
test: all
	@echo "========================================="
	@echo "Running Step 1: FP32 GEMM"
	@echo "========================================="
	$(BIN_DIR)/step1_fp32_gemm
	@echo ""
	@echo "========================================="
	@echo "Running Step 2: Quantization"
	@echo "========================================="
	$(BIN_DIR)/step2_quantization
	@echo ""
	@echo "========================================="
	@echo "Running Step 3: W4A16 GEMM"
	@echo "========================================="
	$(BIN_DIR)/step3_w4a16_gemm
	@echo ""
	@echo "========================================="
	@echo "Running Step 4: W4A8 GEMM"
	@echo "========================================="
	$(BIN_DIR)/step4_w4a8_gemm
	@echo ""
	@echo "========================================="
	@echo "Running Step 5: llama.cpp Comparison"
	@echo "========================================="
	$(BIN_DIR)/step5_llama_comparison

# Run individual steps
.PHONY: step1 step2 step3 step4 step5
step1: $(BIN_DIR)/step1_fp32_gemm
	$(BIN_DIR)/step1_fp32_gemm

step2: $(BIN_DIR)/step2_quantization
	$(BIN_DIR)/step2_quantization

step3: $(BIN_DIR)/step3_w4a16_gemm
	$(BIN_DIR)/step3_w4a16_gemm

step4: $(BIN_DIR)/step4_w4a8_gemm
	$(BIN_DIR)/step4_w4a8_gemm

step5: $(BIN_DIR)/step5_llama_comparison
	$(BIN_DIR)/step5_llama_comparison

# Clean
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Help
.PHONY: help
help:
	@echo "Quantized GEMM Tutorial - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build all test programs"
	@echo "  test     - Build and run all tests"
	@echo "  step1    - Run Step 1: FP32 GEMM baseline"
	@echo "  step2    - Run Step 2: Quantization introduction"
	@echo "  step3    - Run Step 3: W4A16 quantized GEMM"
	@echo "  step4    - Run Step 4: W4A8 quantized GEMM with compensation"
	@echo "  step5    - Run Step 5: Comparison with llama.cpp"
	@echo "  clean    - Remove build artifacts"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_ARCH=$(CUDA_ARCH) (change to match your GPU)"
	@echo ""
	@echo "Examples:"
	@echo "  make CUDA_ARCH=sm_80    # Build for A100"
	@echo "  make CUDA_ARCH=sm_89    # Build for RTX 4090"
	@echo "  make CUDA_ARCH=sm_90    # Build for H100"
	@echo "  make step4              # Run only step 4"

.PHONY: info
info:
	@echo "Build Configuration:"
	@echo "  NVCC: $(NVCC)"
	@echo "  CUDA_ARCH: $(CUDA_ARCH)"
	@echo "  NVCC_FLAGS: $(NVCC_FLAGS)"
	@echo "  LIBS: $(LIBS)"
	@echo ""
	@echo "Directories:"
	@echo "  SRC_DIR: $(SRC_DIR)"
	@echo "  BIN_DIR: $(BIN_DIR)"
	@echo ""
	@echo "Targets:"
	@$(foreach target,$(TARGETS),echo "  $(target)";)
