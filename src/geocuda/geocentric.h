/*!
 * \file geocentric.h
 *
 * \author Han
 * \date 2017/05/28
 *
 * 在 kernel 内核中使用的地心坐标系变换
 */
#pragma once
#include <geocuda/constants.h>
#include <geocuda/math.h>

namespace geocuda {
class GEOCUDA_API Geocentric {
private:
    static const size_t dim_ = 3;
    static const size_t dim2_ = dim_ * dim_;
    double _a, _f, _e2, _e2m, _e2a, _e4a, _maxrad;
    CUDA_HOST_DEVICE_INLINE static void Rotation(double sphi, double cphi, double slam, double clam,
        double M[dim2_]);
    CUDA_HOST_DEVICE_INLINE static void Rotate(double M[dim2_], double x, double y, double z,
        double& X, double& Y, double& Z) {
        // Perform [X,Y,Z]^t = M.[x,y,z]^t
        // (typically local cartesian to geocentric)
        X = M[0] * x + M[1] * y + M[2] * z;
        Y = M[3] * x + M[4] * y + M[5] * z;
        Z = M[6] * x + M[7] * y + M[8] * z;
    }
    CUDA_HOST_DEVICE_INLINE static void Unrotate(double M[dim2_], double X, double Y, double Z,
        double& x, double& y, double& z) {
        // Perform [x,y,z]^t = M^t.[X,Y,Z]^t
        // (typically geocentric to local cartesian)
        x = M[0] * X + M[3] * Y + M[6] * Z;
        y = M[1] * X + M[4] * Y + M[7] * Z;
        z = M[2] * X + M[5] * Y + M[8] * Z;
    }
    CUDA_HOST_DEVICE void IntForward(double lat, double lon, double h, double& X, double& Y, double& Z,
        double M[dim2_]) const;
    CUDA_HOST_DEVICE void IntReverse(double X, double Y, double Z, double& lat, double& lon, double& h,
        double M[dim2_]) const;

public:

    /**
    * Constructor for a ellipsoid with
    *
    * @param[in] a equatorial radius (meters).
    * @param[in] f flattening of ellipsoid.  Setting \e f = 0 gives a sphere.
    *   Negative \e f gives a prolate ellipsoid.
    * @exception GeographicErr if \e a or (1 &minus; \e f) \e a is not
    *   positive.
    **********************************************************************/
    CUDA_HOST_DEVICE Geocentric(double a, double f);

    /**
    * A default constructor (for use by NormalGravity).
    **********************************************************************/
    CUDA_HOST_DEVICE Geocentric() : _a(-1) {}

    /**
    * Convert from geodetic to geocentric coordinates.
    *
    * @param[in] lat latitude of point (degrees).
    * @param[in] lon longitude of point (degrees).
    * @param[in] h height of point above the ellipsoid (meters).
    * @param[out] X geocentric coordinate (meters).
    * @param[out] Y geocentric coordinate (meters).
    * @param[out] Z geocentric coordinate (meters).
    *
    * \e lat should be in the range [&minus;90&deg;, 90&deg;].
    **********************************************************************/
    CUDA_HOST_DEVICE void Forward(double lat, double lon, double h, double& X, double& Y, double& Z)
        const {
        if (Init())
            IntForward(lat, lon, h, X, Y, Z, NULL);
    }


    /**
    * Convert from geocentric to geodetic to coordinates.
    *
    * @param[in] X geocentric coordinate (meters).
    * @param[in] Y geocentric coordinate (meters).
    * @param[in] Z geocentric coordinate (meters).
    * @param[out] lat latitude of point (degrees).
    * @param[out] lon longitude of point (degrees).
    * @param[out] h height of point above the ellipsoid (meters).
    *
    * In general there are multiple solutions and the result which maximizes
    * \e h is returned.  If there are still multiple solutions with different
    * latitudes (applies only if \e Z = 0), then the solution with \e lat > 0
    * is returned.  If there are still multiple solutions with different
    * longitudes (applies only if \e X = \e Y = 0) then \e lon = 0 is
    * returned.  The value of \e h returned satisfies \e h &ge; &minus; \e a
    * (1 &minus; <i>e</i><sup>2</sup>) / sqrt(1 &minus; <i>e</i><sup>2</sup>
    * sin<sup>2</sup>\e lat).  The value of \e lon returned is in the range
    * [&minus;180&deg;, 180&deg;].
    **********************************************************************/
    CUDA_HOST_DEVICE void Reverse(double X, double Y, double Z, double& lat, double& lon, double& h)
        const {
        if (Init())
            IntReverse(X, Y, Z, lat, lon, h, NULL);
    }

   

    /** \name Inspector functions
    **********************************************************************/
    ///@{
    /**
    * @return true if the object has been initialized.
    **********************************************************************/
    CUDA_HOST_DEVICE bool Init() const { return _a > 0; }
    /**
    * @return \e a the equatorial radius of the ellipsoid (meters).  This is
    *   the value used in the constructor.
    **********************************************************************/
    CUDA_HOST_DEVICE double MajorRadius() const
    {
        return Init() ? _a : nan("");
    }

    /**
    * @return \e f the  flattening of the ellipsoid.  This is the
    *   value used in the constructor.
    **********************************************************************/
    CUDA_HOST_DEVICE double Flattening() const
    {
        return Init() ? _f : nan("");
    }
    ///@}
};
}// end geocuda
