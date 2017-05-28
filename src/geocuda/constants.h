/*
 * \file constants.h
 * \author Han
 * \data 2017-05-28
 * 
 * 包含一些常亮信息和常用的宏定义
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _MSC_VER 
#ifdef GEOCUDA_SHARED
#ifdef geocuda_EXPORTS
#define GEOCUDA_API __declspec(dllexport)
#else
#define GEOCUDA_API __declspec(dllimport)
#endif
#else
#define GEOCUDA_API
#endif
#endif

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE_INLINE
#define CUDA_HOST_DEVICE
#endif // __CUDACC__

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif // !M_PI


namespace geocuda{

const double WGS84_a = 6378137.0;
const double WGS84_f = 1/298.257223563;
const double MOON2000_a = 1737400.0;
const double MOON2000_f = 0.0;
const double PI = M_PI;
const double RAD2DEG = 180.0 / PI;
const double DEG2RAD = PI / 180.0;

CUDA_HOST_DEVICE_INLINE double deg2rad() {
    return M_PI / 180.0;
}

CUDA_HOST_DEVICE_INLINE double rad2deg() {
    return 180.0 / M_PI;
}

}// end geocuda
