cmake_minimum_required(VERSION 2.8)

project(geocuda)

if(MSVC)
    list(APPEND CMAKE_CXX_FLAGS "/wd4819")
endif(MSVC)


find_package(CUDA REQUIRED)

if(BUILD_SHARED_LIBS AND MSVC)
    add_definitions(-DGEOCUDA_SHARED)
endif()

add_subdirectory(src)