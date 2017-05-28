#pragma once

struct Point3d
{
    double x, y, z;
};

Point3d* to_device(const Point3d& point);
Point3d to_host(const Point3d* d_point);

Point3d latlon_to_geocentric(double a, double f, const Point3d& latlon);
Point3d latlon_to_lambert(double a, double f, double lon0, const Point3d& latlon);

Point3d geocentric_to_latlon(double a, double f, const Point3d& latlon);
Point3d lambert_to_latlon(double a, double f, double lon0, const Point3d& latlon);
