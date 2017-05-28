#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "wrapper.h"

#include <geocuda/geocentric.h>
#include <GeographicLib/Geocentric.hpp>

Point3d * to_device(const Point3d & point)
{
    Point3d* d_point;
    cudaMalloc(&d_point, sizeof(Point3d));
    cudaMemcpy(d_point, &point, sizeof(Point3d), cudaMemcpyHostToDevice);
    return d_point;
}

Point3d to_host(const Point3d * d_point)
{
    Point3d point;
    cudaMemcpy(&point, d_point, sizeof(Point3d), cudaMemcpyDeviceToHost);
    return point;
}

__global__ void latlon_to_geocentric_kernel(size_t num_points, Point3d* latlon, Point3d* geoc, geocuda::Geocentric converter) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points)
        return;

    const Point3d& ll = latlon[idx];
    Point3d& gc = geoc[idx];
    converter.Forward(ll.x, ll.y, ll.z, gc.x, gc.y, gc.z);
}

Point3d latlon_to_geocentric(double a, double f, const Point3d & latlon)
{
    Point3d* d_point = to_device(latlon);
    Point3d geoc;
    Point3d* d_geoc = to_device(geoc);

    geocuda::Geocentric converter(a, f);

    latlon_to_geocentric_kernel << <1, 1 >> > (1, d_point, d_geoc, converter);

    geoc = to_host(d_geoc);

    GeographicLib::Geocentric geographic_converter(a, f);
    Point3d geographic_geoc;
    geographic_converter.Forward(latlon.x, latlon.y, latlon.z, geographic_geoc.x, geographic_geoc.y, geographic_geoc.z);


    cudaFree(d_point);
    cudaFree(d_geoc);
    return geoc;
}
