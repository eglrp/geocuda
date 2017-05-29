/*!
 * \file test_main.cpp
 *
 * \author Han
 * \date 2017/05/29
 *
 * 主函数
 */
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <geocuda/constants.h>
#include "wrapper.h"

#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LambertConformalConic.hpp>

// CPU 和 GPU 之间可能会存在一点差异
const double EPSILON = 1e-6;

double norm(const Point3d& p1, const Point3d& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;

    return sqrt(dx * dx + dy * dy + dz * dz);
}


TEST_CASE("test_geocentric", "test geocentric transform") {
    Point3d latlon;
    Point3d geoc;

    latlon.x = 10.0;
    latlon.y = 20.0;
    latlon.z = 100.0;

    geoc = latlon_to_geocentric(geocuda::MOON2000_a, geocuda::MOON2000_f, latlon);
    GeographicLib::Geocentric converter(geocuda::MOON2000_a, geocuda::MOON2000_f);
    Point3d geoc2;
    converter.Forward(latlon.x, latlon.y, latlon.z, geoc2.x, geoc2.y, geoc2.z);

    Point3d latlon2 = geocentric_to_latlon(geocuda::MOON2000_a, geocuda::MOON2000_f, geoc);

    REQUIRE(norm(geoc, geoc2) < EPSILON);
    REQUIRE(norm(latlon, latlon2) < EPSILON);
}

TEST_CASE("test_lambert", "test lambert conformal conic transformation") {
    Point3d latlon, latlon2;
    Point3d lambert, lambert2;

    latlon.x = 10.0;
    latlon.y = 20.0;
    latlon.z = 100.0;

    double lon0 = 21.0;

    lambert = latlon_to_lambert(geocuda::MOON2000_a, geocuda::MOON2000_f, lon0, latlon);

    GeographicLib::LambertConformalConic converter(geocuda::MOON2000_a, geocuda::MOON2000_f, 0.0, 1.0);
    converter.Forward(lon0, latlon.x, latlon.y, lambert2.x, lambert2.y);
    lambert2.z = latlon.z;

    latlon2 = lambert_to_latlon(geocuda::MOON2000_a, geocuda::MOON2000_f, lon0, lambert);

    REQUIRE(norm(lambert, lambert2) < EPSILON);
    REQUIRE(norm(latlon, latlon2) < EPSILON);
}
