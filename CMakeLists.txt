cmake_minimum_required(VERSION 3.18)
project(quant_gemm_from_scratch VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA settings
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA is enabled via enable_language(CUDA) above
# No need for find_package(CUDA) in modern CMake

# Set CUDA architecture (auto-detect or specify)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Try to detect GPU architecture
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(GPU_COMPUTE_CAP)
        # Extract first compute capability (e.g., "8.6" -> "86")
        string(REGEX REPLACE "([0-9]+)\\.([0-9]+)" "\\1\\2" GPU_ARCH "${GPU_COMPUTE_CAP}")
        set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})
        message(STATUS "Detected GPU compute capability: ${GPU_COMPUTE_CAP} (sm_${GPU_ARCH})")
    else()
        # Default to sm_75 (Turing) if detection fails
        set(CMAKE_CUDA_ARCHITECTURES 75)
        message(STATUS "Could not detect GPU, defaulting to sm_75")
    endif()
endif()

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# Find curand library (part of CUDA)
find_library(CURAND_LIBRARY 
    NAMES curand
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib64 lib
    REQUIRED
)

# Test executables
set(TEST_SOURCES
    tests/step1_fp32_gemm.cu
    tests/step2_quantization.cu
    tests/step3_w4a16_gemm.cu
    tests/step4_w4a8_gemm.cu
)

foreach(test_file ${TEST_SOURCES})
    get_filename_component(test_name ${test_file} NAME_WE)
    
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} 
        PRIVATE 
        ${CURAND_LIBRARY}
    )
    target_compile_features(${test_name} PRIVATE cxx_std_17)
    
    # Set CUDA architecture for each target
    set_target_properties(${test_name} PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    message(STATUS "Added test: ${test_name}")
endforeach()

# Create a custom target to run all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_COMMAND} -E echo "Running all tests..."
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/step1_fp32_gemm
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/step2_quantization
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/step3_w4a16_gemm
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/step4_w4a8_gemm
    DEPENDS step1_fp32_gemm step2_quantization step3_w4a16_gemm step4_w4a8_gemm
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all test programs"
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "CUDA Toolkit: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "=============================")
message(STATUS "")
message(STATUS "To build: cmake --build .")
message(STATUS "To run all tests: cmake --build . --target run_all_tests")
message(STATUS "")