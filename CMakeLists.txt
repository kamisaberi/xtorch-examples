cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(xtorch_examples LANGUAGES CXX)

# =============================================================================
# ONNX Runtime Dependency Management
#
# This block automatically downloads and prepares ONNX Runtime for all
# sub-projects. It creates a modern CMake target "ONNX::Runtime"
# that subdirectories can easily link against.
# =============================================================================

# Include the module that allows us to download content
include(FetchContent)

# --- 1. Add an option for the user to choose between CPU and GPU ---
option(XTORCH_USE_GPU "Build with GPU support for ONNX Runtime (CUDA)" OFF)

# --- 2. Define ONNX Runtime versions and download URLs ---
# It's good practice to define these at the top for easy updates.
# Using ONNX Runtime v1.22 as a stable example.

if(XTORCH_USE_GPU)
    message(STATUS "Configuring ONNX Runtime for GPU (CUDA)")
    # --- GPU URLs ---
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz")
        set(ONNXRUNTIME_SHA256 "2a19dbfa403672ec27378c3d40a68f793ac7a6327712cd0e8240a86be2b10c55")
    else()
        message(FATAL_ERROR "GPU builds for ONNX Runtime are not defined for ${CMAKE_SYSTEM_NAME} in this script.")
    endif()
else()
    message(STATUS "Configuring ONNX Runtime for CPU")
    # --- CPU URLs ---
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz")
        set(ONNXRUNTIME_SHA256 "8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-arm64-1.22.0.zip")
        set(ONNXRUNTIME_SHA256 "7008f7ff82f8e7de563a22f2b590e08e706a1289eba606b93de2b56edfb1e04b")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin") # macOS
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-universal2-1.22.0.tgz")
        set(ONNXRUNTIME_SHA256 "cfa6f6584d87555ed9f6e7e8a000d3947554d589efe3723b8bfa358cd263d03c")
    endif()
endif()

# --- 3. Declare the dependency ---
FetchContent_Declare(
        onnxruntime
        URL           ${ONNXRUNTIME_URL}
        URL_HASH      SHA256=${ONNXRUNTIME_SHA256}
)

# --- 4. Make the dependency available ---
# This downloads and extracts the content into the build directory.
# The variable 'onnxruntime_SOURCE_DIR' will point to the extracted folder.
FetchContent_MakeAvailable(onnxruntime)

# --- 5. Create a modern IMPORTED target for easy linking ---
# This is the best practice. We create a target named "ONNX::Runtime"
# that bundles the library and include paths together.
find_library(
        ONNXRUNTIME_LIBRARY_PATH
        NAMES onnxruntime # CMake will find libonnxruntime.so, onnxruntime.lib, etc.
        PATHS "${onnxruntime_SOURCE_DIR}/lib"
        REQUIRED
)

add_library(ONNX::Runtime SHARED IMPORTED)
set_target_properties(ONNX::Runtime PROPERTIES
        IMPORTED_LOCATION             "${ONNXRUNTIME_LIBRARY_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
)

message(STATUS "ONNX Runtime ready. Use target 'ONNX::Runtime' to link.")
# =============================================================================
# End of ONNX Runtime Dependency Management
# =============================================================================


# --- Add your sub-projects ---
# These subdirectories can now use the ONNX::Runtime target.
message(STATUS "Adding subdirectories...")

#add_subdirectory(temp)
add_subdirectory(computer_vision)