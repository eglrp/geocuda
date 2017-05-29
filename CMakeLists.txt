cmake_minimum_required(VERSION 2.8)

project(geocuda)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(GEOCUDA_BUILD_TESTS "build test" OFF)

if(MSVC)
    list(APPEND CMAKE_CXX_FLAGS "/wd4819")
endif(MSVC)

find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_30,code=sm_30")
list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_35,code=sm_35")
if(NOT CUDA_VERSION VERSION_LESS "6.5") # >= 6.5
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_37,code=sm_37")
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_50,code=sm_50")
endif()
if(NOT CUDA_VERSION VERSION_LESS "7.0") # >= 7.0
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_52,code=sm_52")
endif()
if(NOT CUDA_VERSION VERSION_LESS "8.0") # >= 8.0
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_60,code=sm_60")
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_61,code=sm_61")
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR})

if(GEOCUDA_BUILD_TESTS)
    set(GeographicLib_USE_STATIC_LIBS ON)
    add_definitions(-DGEOGRAPHICLIB_SHARED_LIB=0)
    find_package(GeographicLib REQUIRED)
    add_subdirectory(tests)
endif(GEOCUDA_BUILD_TESTS)


