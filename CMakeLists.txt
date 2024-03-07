# This CMake config hopefully makes it easier to compile.
# Ensure the CUDA Toolkit is available on your path. Then run:
#   For  GCC: `cmake -B build . && cmake --build build`
#   For MSVC: `cmake -B build . && cmake --build build --config Release`
# You can also use the following options and variables
#  - COMPUTE_BACKEND: Set to `cpu`, `cuda`, or `mps` to select the backend
#  - CUDA_VERSION: The expected CUDA version, for sanity checking. The actual version
#                  is whatever CMake finds on your path.
#  - COMPUTE_CAPABILITY: Which GPU Arch/Compute codes to provide to NVCC.
#                        Separate by semicolons, i.e. `-DCOMPUTE_CAPABILITY=89;90`
#                        Check your compute capability here: https://developer.nvidia.com/cuda-gpus
#  - PTXAS_VERBOSE: Pass the `-v` option to the PTX Assembler
cmake_minimum_required(VERSION 3.22.1)

project(bitsandbytes LANGUAGES CXX)

# Define included source files
set(CPP_FILES csrc/common.cpp csrc/cpu_ops.cpp csrc/pythonInterface.cpp)
set(CUDA_FILES csrc/ops.cu csrc/kernels.cu)
set(MPS_FILES csrc/mps_ops.mm)
set(METAL_FILES csrc/mps_kernels.metal)
# C++ sources are always included
list(APPEND SRC_FILES ${CPP_FILES})

set(COMPUTE_BACKEND "cpu" CACHE STRING "The compute backend to use (cpu, cuda, mps)")
set_property(CACHE COMPUTE_BACKEND PROPERTY STRINGS cpu cuda mps)
option(PTXAS_VERBOSE "Pass through -v flag to PTX Assembler" OFF)

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 13.1)
endif()

set(BNB_OUTPUT_NAME "bitsandbytes")

message(STATUS "Configuring ${PROJECT_NAME} (Backend: ${COMPUTE_BACKEND})")

if(${COMPUTE_BACKEND} STREQUAL "cuda")
    if(APPLE)
        message(FATAL_ERROR "CUDA is not supported on macOS" )
    endif()
    set(BUILD_CUDA ON)
    set(BUILD_MPS OFF)
elseif(${COMPUTE_BACKEND} STREQUAL "mps")
    if(NOT APPLE)
        message(FATAL_ERROR "MPS is only supported on macOS" )
    endif()
    set(BUILD_CUDA OFF)
    set(BUILD_MPS ON)
else()
    set(BUILD_CUDA OFF)
    set(BUILD_MPS OFF)
endif()


if(BUILD_CUDA)
    enable_language(CUDA) # This will fail if CUDA is not found
    find_package(CUDAToolkit REQUIRED)

    # Convert the CUDA version from X.Y.z to XY. There's probably a shorter way of doing this
    string(REGEX MATCH "^[0-9]+.[0-9]+" _CUDA_VERSION_FIRST_TWO "${CMAKE_CUDA_COMPILER_VERSION}")
    string(REPLACE "." "" CUDA_VERSION_SHORT "${_CUDA_VERSION_FIRST_TWO}")

    # Expose a cache variable that the user can set to ensure the correct version of CUDA is found
    set(CUDA_VERSION "${CUDA_VERSION_SHORT}" CACHE STRING "Expected CUDA Version Shortcode")

    message(STATUS "CUDA Version: ${CUDA_VERSION_SHORT} (${CMAKE_CUDA_COMPILER_VERSION})")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

    # It should match the discovered version
    if(NOT CUDA_VERSION STREQUAL "${CUDA_VERSION_SHORT}")
        message(FATAL_ERROR "You've specified CUDA version ${CUDA_VERSION} however the CUDA compiler found is ${CUDA_VERSION_SHORT}."
            " Ensure the desired CUDA compiler is the first one available on your PATH."
        )
    endif()

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
        message(FATAL_ERROR "CUDA Version < 11 is not supported")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
        message(FATAL_ERROR "CUDA Version > 12 is not supported")
    endif()

    # CMake < 3.23.0 does not define CMAKE_CUDA_ARCHITECTURES_ALL.
    if(CMAKE_VERSION VERSION_LESS "3.23.0")
        message(STATUS "CMake < 3.23.0; determining CUDA architectures supported...")

        # 11.x and 12.x both support these at a minimum.
        set(CMAKE_CUDA_ARCHITECTURES_ALL 50 52 53 60 61 62 70 72 75 80)
        set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR 50 60 70 80)

        # CUDA 11.1 adds Ampere support for GA102-GA107.
        if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.1")
            list(APPEND CMAKE_CUDA_ARCHITECTURES_ALL 86)
        endif()

        # CUDA 11.4 adds Ampere support for GA10B.
        if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.4")
            list(APPEND CMAKE_CUDA_ARCHITECTURES_ALL 87)
        endif()

        # CUDA 11.8 adds support for Ada and Hopper.
        if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
            list(APPEND CMAKE_CUDA_ARCHITECTURES_ALL 89 90)
            list(APPEND CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR 90)
        endif()
    endif()

    string(APPEND CMAKE_CUDA_FLAGS " --use_fast_math")
    if(PTXAS_VERBOSE)
        # Verbose? Outputs register usage information, and other things...
        string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v")
    endif()

    foreach(capability ${CMAKE_CUDA_ARCHITECTURES_ALL})
        # Most of the items here are like: `xx-real`, so we just extract the `xx` portion
        string(REGEX MATCH "[0-9]+" capability_id "${capability}")
        if(capability_id GREATER 0)
            list(APPEND POSSIBLE_CAPABILITIES ${capability_id})
        endif()
    endforeach()

    # This can be changed via -D argument to CMake
    # By default all possible capabilities are compiled
    set(COMPUTE_CAPABILITY "${POSSIBLE_CAPABILITIES}" CACHE STRING "Compute Capabilities Targeted")

    message(STATUS "CUDA Capabilities Available: ${POSSIBLE_CAPABILITIES}")
    message(STATUS "CUDA Capabilities  Selected: ${COMPUTE_CAPABILITY}")

    # Use the "real" option to build native cubin for all selections.
    # Ensure we build the PTX for the latest version.
    # This behavior of adding a PTX (virtual) target for the highest architecture
    # is similar to how the "all" and "all-major" options would behave in CMake >= 3.23.
    # TODO: Consider bumping CMake requirement and using CMAKE_CUDA_ARCHITECTURES=[all | native] by default
    list(REMOVE_DUPLICATES COMPUTE_CAPABILITY)
    list(SORT COMPUTE_CAPABILITY COMPARE NATURAL)
    list(POP_BACK COMPUTE_CAPABILITY _LATEST_CAPABILITY)
    list(TRANSFORM COMPUTE_CAPABILITY APPEND "-real" OUTPUT_VARIABLE CMAKE_CUDA_ARCHITECTURES)
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${_LATEST_CAPABILITY})

    message(STATUS "CUDA Targets: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA NVCC Flags: ${CMAKE_CUDA_FLAGS}")

    list(APPEND SRC_FILES ${CUDA_FILES})

    string(APPEND BNB_OUTPUT_NAME "_cuda${CUDA_VERSION_SHORT}")

    add_compile_definitions(BUILD_CUDA)
elseif(BUILD_MPS)
    if(NOT APPLE)
        message(FATAL_ERROR "MPS is only supported on macOS" )
    endif()

    enable_language(OBJCXX)

    list(APPEND SRC_FILES ${MPS_FILES})

    string(APPEND BNB_OUTPUT_NAME "_mps")
    add_compile_definitions(BUILD_MPS)
    file(MAKE_DIRECTORY "build")
    add_custom_command(OUTPUT "bitsandbytes/bitsandbytes.metallib"
                COMMAND xcrun metal -c -o "build/bitsandbytes.air" ${METAL_FILES}
                COMMAND xcrun metallib "build/bitsandbytes.air" -o "bitsandbytes/bitsandbytes.metallib"
                DEPENDS "${METAL_FILES}"
                COMMENT "Compiling Metal kernels"
                VERBATIM)
    add_custom_target(metallib DEPENDS "bitsandbytes/bitsandbytes.metallib")
else()
    string(APPEND BNB_OUTPUT_NAME "_cpu")
    set(GPU_SOURCES)
endif()

if(WIN32)
    # Export all symbols
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /fp:fast")
endif()

set_source_files_properties(${CPP_FILES} PROPERTIES LANGUAGE CXX)
add_library(bitsandbytes SHARED ${SRC_FILES})
target_compile_features(bitsandbytes PUBLIC cxx_std_14)
target_include_directories(bitsandbytes PUBLIC csrc include)


if(BUILD_CUDA)
    target_include_directories(bitsandbytes PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # Note: As of CUDA 11.0, cublas depends on cublasLt.
    # See: https://gitlab.kitware.com/cmake/cmake/-/merge_requests/6857/diffs
    # It is listed here for assurance. In CMake > 3.23.0, it's implicit when linking CUDA::cublas.
    target_link_libraries(bitsandbytes PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusparse)

    set_target_properties(bitsandbytes
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
endif()
if(BUILD_MPS)
    add_dependencies(bitsandbytes metallib)
    target_link_libraries(bitsandbytes objc "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders" "-framework MetalPerformanceShadersGraph")
endif()

if(WIN32)
    set_target_properties(bitsandbytes PROPERTIES PREFIX "lib")
endif()
set_target_properties(bitsandbytes PROPERTIES OUTPUT_NAME ${BNB_OUTPUT_NAME})
if(MSVC)
    set_target_properties(bitsandbytes PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bitsandbytes")
    set_target_properties(bitsandbytes PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bitsandbytes")
    set_target_properties(bitsandbytes PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bitsandbytes")
    set_target_properties(bitsandbytes PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bitsandbytes")
endif()

set_target_properties(bitsandbytes PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bitsandbytes")
