#!/bin/bash
# Build and Test Script for Quantized GEMM Tutorial
# Usage: ./scripts/build_and_test.sh [step_number]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CUDA_ARCH=${CUDA_ARCH:-sm_86}  # Default to RTX 3090/4090
BUILD_DIR="build"
BIN_DIR="bin"

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print section header
print_header() {
    echo ""
    print_msg "$BLUE" "=========================================="
    print_msg "$BLUE" "$1"
    print_msg "$BLUE" "=========================================="
}

# Check CUDA installation
check_cuda() {
    print_header "Checking CUDA Installation"

    if ! command -v nvcc &> /dev/null; then
        print_msg "$RED" "Error: nvcc not found. Please install CUDA Toolkit."
        exit 1
    fi

    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_msg "$GREEN" "✓ CUDA found: version $CUDA_VERSION"

    if ! command -v nvidia-smi &> /dev/null; then
        print_msg "$YELLOW" "Warning: nvidia-smi not found. Cannot detect GPU."
    else
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        GPU_COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
        print_msg "$GREEN" "✓ GPU detected: $GPU_NAME (sm_$GPU_COMPUTE)"

        # Update CUDA_ARCH if detected
        if [ ! -z "$GPU_COMPUTE" ]; then
            CUDA_ARCH="sm_$GPU_COMPUTE"
            print_msg "$GREEN" "  Using detected architecture: $CUDA_ARCH"
        fi
    fi
}

# Build all steps
build_all() {
    print_header "Building All Steps"

    mkdir -p "$BIN_DIR"

    print_msg "$YELLOW" "Building with CUDA_ARCH=$CUDA_ARCH"

    make clean
    make CUDA_ARCH=$CUDA_ARCH -j$(nproc)

    if [ $? -eq 0 ]; then
        print_msg "$GREEN" "✓ Build successful!"
    else
        print_msg "$RED" "✗ Build failed!"
        exit 1
    fi
}

# Run a single step
run_step() {
    local step=$1
    local binary="$BIN_DIR/step${step}_*"

    # Find the binary
    binary=$(ls $binary 2>/dev/null | head -n1)

    if [ -z "$binary" ]; then
        print_msg "$RED" "✗ Binary for step $step not found"
        return 1
    fi

    print_header "Running Step $step: $(basename $binary)"

    # Run with timeout (5 minutes)
    timeout 300 $binary

    if [ $? -eq 0 ]; then
        print_msg "$GREEN" "✓ Step $step completed successfully"
        return 0
    else
        print_msg "$RED" "✗ Step $step failed or timed out"
        return 1
    fi
}

# Run all steps
run_all() {
    print_header "Running All Steps"

    local failed=0

    for step in 1 2 3 4 5; do
        if ! run_step $step; then
            failed=$((failed + 1))
        fi
        echo ""
    done

    print_header "Test Summary"
    if [ $failed -eq 0 ]; then
        print_msg "$GREEN" "✓ All tests passed!"
    else
        print_msg "$RED" "✗ $failed test(s) failed"
        exit 1
    fi
}

# Main script
main() {
    print_header "Quantized GEMM Tutorial - Build & Test"

    # Check CUDA
    check_cuda

    # Parse arguments
    if [ $# -eq 0 ]; then
        # No arguments: build and run all
        build_all
        run_all
    elif [ "$1" == "build" ]; then
        # Just build
        build_all
    elif [ "$1" == "clean" ]; then
        # Clean
        print_header "Cleaning Build Artifacts"
        make clean
        print_msg "$GREEN" "✓ Clean complete"
    elif [[ "$1" =~ ^[1-5]$ ]]; then
        # Run specific step
        if [ ! -f "$BIN_DIR/step${1}_"* ]; then
            print_msg "$YELLOW" "Binary not found, building first..."
            build_all
        fi
        run_step $1
    else
        print_msg "$RED" "Usage: $0 [build|clean|1|2|3|4|5]"
        print_msg "$YELLOW" "  build  - Build all steps"
        print_msg "$YELLOW" "  clean  - Clean build artifacts"
        print_msg "$YELLOW" "  1-5    - Run specific step"
        print_msg "$YELLOW" "  (none) - Build and run all steps"
        exit 1
    fi
}

main "$@"
