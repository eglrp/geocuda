/*!
 * \file geocentric.cu
 *
 * \author Han
 * \date 2017/05/28
 *
 * 在 kernel 内核中使用的地心坐标系变换
 */
#include <float.h>
#include <cmath>

#include <geocuda/geocentric.h>
namespace geocuda {

using namespace std;

Geocentric::Geocentric(double a, double f)
    : _a(a)
    , _f(f)
    , _e2(_f * (2 - _f))
    , _e2m(math::sq(1 - _f))    // 1 - _e2
    , _e2a(fabs(_e2))
    , _e4a(math::sq(_e2))
    , _maxrad(2 * _a / DBL_MIN)
{}

void Geocentric::IntForward(double lat, double lon, double h,
    double& X, double& Y, double& Z,
    double M[dim2_]) const {
    double sphi, cphi, slam, clam;
    math::sincosd(math::latfix(lat), sphi, cphi);
    math::sincosd(lon, slam, clam);
    double n = _a / sqrt(1 - _e2 * math::sq(sphi));
    Z = (_e2m * n + h) * sphi;
    X = (n + h) * cphi;
    Y = X * slam;
    X *= clam;
    if (M)
        Rotation(sphi, cphi, slam, clam, M);
}

void Geocentric::IntReverse(double X, double Y, double Z,
    double& lat, double& lon, double& h,
    double M[dim2_]) const {
    double R = hypot(X, Y),
        slam = R ? Y / R : 0,
        clam = R ? X / R : 1;
    h = hypot(R, Z);      // Distance to center of earth
    double sphi, cphi;
    if (h > _maxrad) {
        // We doublely far away (> 12 million light years); treat the earth as a
        // point and h, above, is an acceptable approximation to the height.
        // This avoids overflow, e.g., in the computation of disc below.  It's
        // possible that h has overflowed to inf; but that's OK.
        //
        // Treat the case X, Y finite, but R overflows to +inf by scaling by 2.
        R = hypot(X / 2, Y / 2);
        slam = R ? (Y / 2) / R : 0;
        clam = R ? (X / 2) / R : 1;
        double H = hypot(Z / 2, R);
        sphi = (Z / 2) / H;
        cphi = R / H;
    }
    else if (_e4a == 0) {
        // Treat the spherical case.  Dealing with underflow in the general case
        // with _e2 = 0 is difficult.  Origin maps to N pole same as with
        // ellipsoid.
        double H = hypot(h == 0 ? 1 : Z, R);
        sphi = (h == 0 ? 1 : Z) / H;
        cphi = R / H;
        h -= _a;
    }
    else {
        // Treat prolate spheroids by swapping R and Z here and by switching
        // the arguments to phi = atan2(...) at the end.
        double
            p = math::sq(R / _a),
            q = _e2m * math::sq(Z / _a),
            r = (p + q - _e4a) / 6;
        if (_f < 0) math::swap(p, q);
        if (!(_e4a * q == 0 && r <= 0)) {
            double
                // Avoid possible division by zero when r = 0 by multiplying
                // equations for s and t by r^3 and r, resp.
                S = _e4a * p * q / 4, // S = r^3 * s
                r2 = math::sq(r),
                r3 = r * r2,
                disc = S * (2 * r3 + S);
            double u = r;
            if (disc >= 0) {
                double T3 = S + r3;
                // Pick the sign on the sqrt to maximize abs(T3).  This minimizes
                // loss of precision due to cancellation.  The result is unchanged
                // because of the way the T is used in definition of u.
                T3 += T3 < 0 ? -sqrt(disc) : sqrt(disc); // T3 = (r * t)^3
                                                         // N.B. cbrt always returns the double root.  cbrt(-8) = -2.
                double T = cbrt(T3); // T = r * t
                                         // T can be zero; but then r2 / T -> 0.
                u += T + (T ? r2 / T : 0);
            }
            else {
                // T is complex, but the way u is defined the result is double.
                double ang = atan2(sqrt(-disc), -(S + r3));
                // There are three possible cube roots.  We choose the root which
                // avoids cancellation.  Note that disc < 0 implies that r < 0.
                u += 2 * r * cos(ang / 3);
            }
            double
                v = sqrt(math::sq(u) + _e4a * q), // guaranteed positive
                                                  // Avoid loss of accuracy when u < 0.  Underflow doesn't occur in
                                                  // e4 * q / (v - u) because u ~ e^4 when q is small and u < 0.
                uv = u < 0 ? _e4a * q / (v - u) : u + v, // u+v, guaranteed positive
                                                         // Need to guard against w going negative due to roundoff in uv - q.
                w = max(double(0), _e2a * (uv - q) / (2 * v)),
                // Rearrange expression for k to avoid loss of accuracy due to
                // subtraction.  Division by 0 not possible because uv > 0, w >= 0.
                k = uv / (sqrt(uv + math::sq(w)) + w),
                k1 = _f >= 0 ? k : k - _e2,
                k2 = _f >= 0 ? k + _e2 : k,
                d = k1 * R / k2,
                H = hypot(Z / k1, R / k2);
            sphi = (Z / k1) / H;
            cphi = (R / k2) / H;
            h = (1 - _e2m / k1) * hypot(d, Z);
        }
        else {                  // e4 * q == 0 && r <= 0
                                // This leads to k = 0 (oblate, equatorial plane) and k + e^2 = 0
                                // (prolate, rotation axis) and the generation of 0/0 in the general
                                // formulas for phi and h.  using the general formula and division by 0
                                // in formula for h.  So handle this case by taking the limits:
                                // f > 0: z -> 0, k      ->   e2 * sqrt(q)/sqrt(e4 - p)
                                // f < 0: R -> 0, k + e2 -> - e2 * sqrt(q)/sqrt(e4 - p)
            double
                zz = sqrt((_f >= 0 ? _e4a - p : p) / _e2m),
                xx = sqrt(_f < 0 ? _e4a - p : p),
                H = hypot(zz, xx);
            sphi = zz / H;
            cphi = xx / H;
            if (Z < 0) sphi = -sphi; // for tiny negative Z (not for prolate)
            h = -_a * (_f >= 0 ? _e2m : 1) * H / _e2a;
        }
    }
    lat = math::atan2d(sphi, cphi);
    lon = math::atan2d(slam, clam);
    if (M)
        Rotation(sphi, cphi, slam, clam, M);
}

void Geocentric::Rotation(double sphi, double cphi, double slam, double clam,
    double M[dim2_]) {
    // This rotation matrix is given by the following quaternion operations
    // qrot(lam, [0,0,1]) * qrot(phi, [0,-1,0]) * [1,1,1,1]/2
    // or
    // qrot(pi/2 + lam, [0,0,1]) * qrot(-pi/2 + phi , [-1,0,0])
    // where
    // qrot(t,v) = [cos(t/2), sin(t/2)*v[1], sin(t/2)*v[2], sin(t/2)*v[3]]

    // Local X axis (east) in geocentric coords
    M[0] = -slam;        M[3] = clam;        M[6] = 0;
    // Local Y axis (north) in geocentric coords
    M[1] = -clam * sphi; M[4] = -slam * sphi; M[7] = cphi;
    // Local Z axis (up) in geocentric coords
    M[2] = clam * cphi; M[5] = slam * cphi; M[8] = sphi;
}
}//end geocuda
