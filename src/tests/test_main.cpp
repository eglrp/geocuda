#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <geocuda/constants.h>
#include "wrapper.h"

#include <GeographicLib/Geocentric.hpp>

TEST_CASE("test_geocentric", "test geocentric transform") {
    Point3d latlon;
    Point3d geoc;

    latlon.x = 10.0;
    latlon.y = 20.0;
    latlon.z = 100.0;

    geoc = latlon_to_geocentric(geocuda::MOON2000_a, geocuda::MOON2000_f, latlon);
}