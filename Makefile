# ============================================================================
# Quantized GEMM from Scratch: A Tutorial
# Compatible with llama.cpp quantization formats
# ============================================================================

# Compiler settings
NVCC := nvcc
CUDA_ARCH := sm_86  # Change this to match your GPU (sm_75, sm_80, sm_86, sm_89, sm_90, etc.)
NVCC_FLAGS := -O3 -arch=$(CUDA_ARCH) -std=c++17
NVCC_FLAGS += -Xcompiler -Wall -Xcompiler -Wextra
NVCC_FLAGS += -I./include -I./compat -I./kernels -I.

# Libraries
LIBS := -lcurand

# Optional: Link with llama.cpp for comparison
LLAMA_CPP_DIR := ../llama.cpp
LLAMA_CPP_BUILD := $(LLAMA_CPP_DIR)/build
ifneq ($(wildcard $(LLAMA_CPP_DIR)),)
    HAS_LLAMA_CPP := 1
    NVCC_FLAGS += -DHAS_LLAMA_CPP
    NVCC_FLAGS += -I$(LLAMA_CPP_DIR)/ggml/include
    NVCC_FLAGS += -I$(LLAMA_CPP_DIR)/ggml/src
    LIBS += -L$(LLAMA_CPP_BUILD)/ggml/src -lggml
endif

# Directories
SRC_DIR := tests
BUILD_DIR := build
BIN_DIR := bin
UNIT_TEST_DIR := tests/unit
BENCHMARK_DIR := tests/benchmark
INTEGRATION_DIR := tests/integration

# Source files - Legacy step tests
STEP_SOURCES := $(wildcard $(SRC_DIR)/step*.cu)
STEP_TARGETS := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(STEP_SOURCES))

# Unit tests
UNIT_SOURCES := $(wildcard $(UNIT_TEST_DIR)/*.cu)
UNIT_TARGETS := $(patsubst $(UNIT_TEST_DIR)/%.cu,$(BIN_DIR)/unit/%,$(UNIT_SOURCES))

# All targets
TARGETS := $(STEP_TARGETS) $(UNIT_TARGETS)

# Default target
.PHONY: all
all: dirs $(TARGETS)

# Create directories
.PHONY: dirs
dirs:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(BIN_DIR)/unit
	@mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/unit:
	mkdir -p $(BIN_DIR)/unit

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

# ============================================================================
# Unit Tests
# ============================================================================

# Generic rule for unit tests
$(BIN_DIR)/unit/%: $(UNIT_TEST_DIR)/%.cu $(UNIT_TEST_DIR)/../framework/test_framework.cuh compat/ggml_types.h
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

# Specific unit test targets
.PHONY: test-gemm-q4
test-gemm-q4: $(BIN_DIR)/unit/test_gemm_q4
	$(BIN_DIR)/unit/test_gemm_q4

.PHONY: test-gemm-all
test-gemm-all: $(BIN_DIR)/unit/test_gemm_all_quants
	$(BIN_DIR)/unit/test_gemm_all_quants

.PHONY: test-silu
test-silu: $(BIN_DIR)/unit/test_silu
	$(BIN_DIR)/unit/test_silu

.PHONY: test-rms-norm
test-rms-norm: $(BIN_DIR)/unit/test_rms_norm
	$(BIN_DIR)/unit/test_rms_norm

.PHONY: test-softmax
test-softmax: $(BIN_DIR)/unit/test_softmax
	$(BIN_DIR)/unit/test_softmax

.PHONY: test-rope
test-rope: $(BIN_DIR)/unit/test_rope
	$(BIN_DIR)/unit/test_rope

# Run unit tests
.PHONY: unit-test
unit-test: $(UNIT_TARGETS)
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║                  Running Unit Tests                           ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"
	@for test in $(UNIT_TARGETS); do \
		echo ""; \
		echo "Running: $$test"; \
		$$test || exit 1; \
	done
	@echo ""
	@echo "All unit tests passed!"

# ============================================================================
# Run all tests
# ============================================================================

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
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║       Quantized GEMM Tutorial - Build System                  ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Build Targets:"
	@echo "  all           - Build all programs (steps + unit tests)"
	@echo "  dirs          - Create output directories"
	@echo ""
	@echo "Test Targets:"
	@echo "  test          - Run legacy step tests"
	@echo "  unit-test     - Run all unit tests"
	@echo "  test-gemm-q4  - Run GEMM Q4_0 x Q8_1 unit test"
	@echo ""
	@echo "Step Tests (Legacy):"
	@echo "  step1         - FP32 GEMM baseline"
	@echo "  step2         - Quantization introduction"
	@echo "  step3         - W4A16 quantized GEMM"
	@echo "  step4         - W4A8 quantized GEMM with compensation"
	@echo "  step5         - Comparison with llama.cpp"
	@echo ""
	@echo "Other:"
	@echo "  clean         - Remove build artifacts"
	@echo "  info          - Show build configuration"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_ARCH=$(CUDA_ARCH) (change to match your GPU)"
	@echo ""
	@echo "Examples:"
	@echo "  make CUDA_ARCH=sm_80     # Build for A100"
	@echo "  make CUDA_ARCH=sm_89     # Build for RTX 4090"
	@echo "  make unit-test           # Run all unit tests"
	@echo "  make test-gemm-q4        # Run specific test"

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
