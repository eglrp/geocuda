cmake_minimum_required(VERSION 2.8)

project(geocuda)

option(GEOCUDA_BUILD_TESTS "build test" OFF)

if(MSVC)
    list(APPEND CMAKE_CXX_FLAGS "/wd4819")
endif(MSVC)

find_package(CUDA REQUIRED)
if(GEOCUDA_BUILD_TESTS)
    set(GeographicLib_USE_STATIC_LIBS ON)
    add_definitions(-DGEOGRAPHICLIB_SHARED_LIB=0)
    find_package(GeographicLib REQUIRED)    
endif(GEOCUDA_BUILD_TESTS)


if(BUILD_SHARED_LIBS AND MSVC)
    add_definitions(-DGEOCUDA_SHARED)
endif()


add_subdirectory(src)