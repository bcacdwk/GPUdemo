cmake_minimum_required(VERSION 3.12)
project(imma_perf_test LANGUAGES CXX CUDA)

# Find CUDA and add the required libraries
find_package(CUDA REQUIRED)

# Add the Common directory from CUDAsampleSHY for helper files
include_directories("../../../CUDAsampleSHY/Common")

# Add the executable
add_executable(imma_perf_test imma_perf_test.cu)

# Target specific CUDA properties
set_target_properties(imma_perf_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA libraries
target_link_libraries(imma_perf_test PRIVATE ${CUDA_LIBRARIES})