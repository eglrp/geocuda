/*!
 * \file wrapper.cu
 *
 * \author Han
 * \date 2017/05/29
 *
 * 测试函数封装
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "wrapper.h"

#include <geocuda/geocentric.h>
#include <geocuda/lambert_conformal_conic.h>

#include <GeographicLib/Geocentric.hpp>
Point3d *to_device(const Point3d &point) {
    Point3d *d_point;
    cudaMalloc(&d_point, sizeof(Point3d));
    cudaMemcpy(d_point, &point, sizeof(Point3d), cudaMemcpyHostToDevice);
    return d_point;
}

Point3d to_host(const Point3d *d_point) {
    Point3d point;
    cudaMemcpy(&point, d_point, sizeof(Point3d), cudaMemcpyDeviceToHost);
    return point;
}

__global__ void latlon_to_geocentric_kernel(size_t num_points, const Point3d *latlon, Point3d *geoc,
                                            geocuda::Geocentric converter) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points)
        return;

    const Point3d &ll = latlon[idx];
    Point3d &gc = geoc[idx];
    converter.Forward(ll.x, ll.y, ll.z, gc.x, gc.y, gc.z);
}

Point3d latlon_to_geocentric(double a, double f, const Point3d &latlon) {
    Point3d *d_point = to_device(latlon);
    Point3d geoc;
    Point3d *d_geoc = to_device(geoc);

    geocuda::Geocentric converter(a, f);

    latlon_to_geocentric_kernel<<<1, 1>>>(1, d_point, d_geoc, converter);

    geoc = to_host(d_geoc);

    cudaFree(d_point);
    cudaFree(d_geoc);
    return geoc;
}

__global__ void latlon_to_lambert_kernel(size_t num_points, double lon0, const Point3d *latlon, Point3d *lambert,
                                         geocuda::LambertConformalConic converter) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points)
        return;

    const Point3d &ll = latlon[idx];
    Point3d &lb = lambert[idx];

    converter.Forward(lon0, ll.x, ll.y, lb.x, lb.y);
    lb.z = ll.z;
}
Point3d latlon_to_lambert(double a, double f, double lon0, const Point3d &latlon) {
    Point3d *d_latlon = to_device(latlon);
    Point3d lambert;
    Point3d *d_lambert = to_device(lambert);
    geocuda::LambertConformalConic converter(a, f, 0.0, 1.0);

    latlon_to_lambert_kernel<<<1, 1>>>(1, lon0, d_latlon, d_lambert, converter);
    lambert = to_host(d_lambert);

    cudaFree(d_lambert);
    cudaFree(d_latlon);

    return lambert;
}

__global__ void geocentric_to_latlon_kernel(size_t num_points, const Point3d *geoc, Point3d *latlon,
                                            geocuda::Geocentric converter) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points)
        return;

    const Point3d &gc = geoc[idx];
    Point3d &ll = latlon[idx];
    converter.Reverse(gc.x, gc.y, gc.z, ll.x, ll.y, ll.z);
}
Point3d geocentric_to_latlon(double a, double f, const Point3d &geoc) {
    Point3d *d_geoc = to_device(geoc);
    Point3d latlon;
    Point3d *d_latlon = to_device(latlon);

    geocuda::Geocentric converter(a, f);
    geocentric_to_latlon_kernel<<<1, 1>>>(1, d_geoc, d_latlon, converter);
    latlon = to_host(d_latlon);

    cudaFree(d_geoc);
    cudaFree(d_latlon);
    return latlon;
}

__global__ void lambert_to_latlon_kernel(size_t num_points, double lon0, const Point3d* lambert, Point3d* latlon, geocuda::LambertConformalConic converter) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points)
        return;

    const Point3d& lb = lambert[idx];
    Point3d& ll = latlon[idx];

    converter.Reverse(lon0, lb.x, lb.y, ll.x, ll.y);
    ll.z = lb.z;
}
Point3d lambert_to_latlon(double a, double f, double lon0, const Point3d & lambert)
{
    Point3d* d_lambert = to_device(lambert);
    Point3d latlon;
    Point3d* d_latlon = to_device(latlon);

    geocuda::LambertConformalConic convert(a, f, 0.0, 1.0);

    lambert_to_latlon_kernel<<<1, 1>>>(1, lon0, d_lambert, d_latlon, convert);
    latlon = to_host(d_latlon);

    cudaFree(d_latlon);
    cudaFree(d_lambert);

    return latlon;
}
