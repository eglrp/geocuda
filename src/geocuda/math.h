/*!
 * \file math.h
 *
 * \author Han
 * \date 2017/05/28
 *
 * 常用的数学函数
 */
#pragma once
#include <math_functions.h>
#include <geocuda/constants.h>

namespace geocuda {
namespace math {
template <typename T>
CUDA_HOST_DEVICE_INLINE T sq(T x) { return x * x; }


template<typename T> CUDA_HOST_DEVICE_INLINE static void swap(T& x, T&y) {
#ifdef __CUDACC__
    T z(y);
    y = x;
    x = z;
#else
    std::swap(x, y);
#endif // __CUDACC__
}

template<typename T> CUDA_HOST_DEVICE_INLINE static inline T latfix(T x)
{
    return abs(x) > 90.0 ? nan("") : x;
}

template<typename T> CUDA_HOST_DEVICE_INLINE static void sincosd(T x, T& sinx, T& cosx) {
    // In order to minimize round-off errors, this function exactly reduces
    // the argument to the range [-45, 45] before converting it to radians.
    T r; int q;
    r = remquo(x, T(90), &q);

    // now abs(r) <= 45
    r *= deg2rad();
    // Possibly could call the gnu extension sincos
    T s = sin(r), c = cos(r);

    switch (unsigned(q) & 3U) {
    case 0U: sinx = s; cosx = c; break;
    case 1U: sinx = c; cosx = -s; break;
    case 2U: sinx = -s; cosx = -c; break;
    default: sinx = -c; cosx = s; break; // case 3U
    }
    // Set sign of 0 results.  -0 only produced for sin(-0)
    if (x) { sinx += T(0); cosx += T(0); }
}

/**
* Evaluate the atan2 function with the result in degrees
*
* @tparam T the type of the arguments and the returned value.
* @param[in] y
* @param[in] x
* @return atan2(<i>y</i>, <i>x</i>) in degrees.
*
* The result is in the range (&minus;180&deg; 180&deg;].  N.B.,
* atan2d(&plusmn;0, &minus;1) = +180&deg;; atan2d(&minus;&epsilon;,
* &minus;1) = &minus;180&deg;, for &epsilon; positive and tiny;
* atan2d(&plusmn;0, +1) = &plusmn;0&deg;.
**********************************************************************/
template<typename T> CUDA_HOST_DEVICE_INLINE static inline T atan2d(T y, T x) {
    // In order to minimize round-off errors, this function rearranges the
    // arguments so that result of atan2 is in the range [-pi/4, pi/4] before
    // converting it to degrees and mapping the result to the correct
    // quadrant.
    int q = 0;
    if (abs(y) > abs(x)) { swap(x, y); q = 2; }
    if (x < 0) { x = -x; ++q; }
    // here x >= 0 and x >= abs(y), so angle is in [-pi/4, pi/4]
    T ang = atan2(y, x) * rad2deg();
    switch (q) {
        // Note that atan2d(-0.0, 1.0) will return -0.  However, we expect that
        // atan2d will not be called with y = -0.  If need be, include
        //
        //   case 0: ang = 0 + ang; break;
        //
        // and handle mpfr as in AngRound.
    case 1: ang = (y >= 0 ? 180 : -180) - ang; break;
    case 2: ang = 90 - ang; break;
    case 3: ang = -90 + ang; break;
    }
    return ang;
}

/**
* Evaluate the atan function with the result in degrees
*
* @tparam T the type of the argument and the returned value.
* @param[in] x
* @return atan(<i>x</i>) in degrees.
**********************************************************************/
template<typename T> CUDA_HOST_DEVICE_INLINE static inline T atand(T x)
{
    return atan2d(x, T(1));
}

/**
* Evaluate <i>e</i> atanh(<i>e x</i>)
*
* @tparam T the type of the argument and the returned value.
* @param[in] x
* @param[in] es the signed eccentricity =  sign(<i>e</i><sup>2</sup>)
*    sqrt(|<i>e</i><sup>2</sup>|)
* @return <i>e</i> atanh(<i>e x</i>)
*
* If <i>e</i><sup>2</sup> is negative (<i>e</i> is imaginary), the
* expression is evaluated in terms of atan.
**********************************************************************/
template<typename T> CUDA_HOST_DEVICE_INLINE static inline T eatanhe(T x, T es) {
    return es > T(0) ? es * atanh(es * x) : -es * atan(es * x);
}

/**
* @tparam T the type of the returned value.
* @return the number of radians in a degree.
**********************************************************************/
CUDA_HOST_DEVICE_INLINE static inline double degree() {
    return M_PI / 180;
}

/**
* The error-free sum of two numbers.
*
* @tparam T the type of the argument and the returned value.
* @param[in] u
* @param[in] v
* @param[out] t the exact error given by (\e u + \e v) - \e s.
* @return \e s = round(\e u + \e v).
*
* See D. E. Knuth, TAOCP, Vol 2, 4.2.2, Theorem B.  (Note that \e t can be
* the same as one of the first two arguments.)
**********************************************************************/
template<typename T> CUDA_HOST_DEVICE_INLINE static inline T sum(T u, T v, T& t) {
    T s = u + v;
    T up = s - v;
    T vpp = s - up;
    up -= u;
    vpp -= v;
    t = -(up + vpp);
    // u + v =       s      + t
    //       = round(u + v) + t
    return s;
}

/**
* Normalize an angle.
*
* @tparam T the type of the argument and returned value.
* @param[in] x the angle in degrees.
* @return the angle reduced to the range([&minus;180&deg;, 180&deg;].
*
* The range of \e x is unrestricted.
**********************************************************************/
template<typename T> CUDA_HOST_DEVICE_INLINE static inline T AngNormalize(T x) {
    x = remainder(x, T(360));
    return x != -180 ? x : 180;
}
/**
* The exact difference of two angles reduced to
* (&minus;180&deg;, 180&deg;].
*
* @tparam T the type of the arguments and returned value.
* @param[in] x the first angle in degrees.
* @param[in] y the second angle in degrees.
* @param[out] e the error term in degrees.
* @return \e d, the truncated value of \e y &minus; \e x.
*
* This computes \e z = \e y &minus; \e x exactly, reduced to
* (&minus;180&deg;, 180&deg;]; and then sets \e z = \e d + \e e where \e d
* is the nearest representable number to \e z and \e e is the truncation
* error.  If \e d = &minus;180, then \e e &gt; 0; If \e d = 180, then \e e
* &le; 0.
**********************************************************************/
template<typename T> CUDA_HOST_DEVICE_INLINE static inline T AngDiff(T x, T y, T& e) {
    T t, d = AngNormalize(sum(remainder(-x, T(360)),
        remainder(y, T(360)), t));
    return sum(d == 180 && t > 0 ? -180 : d, t, e);
}

/**
* Difference of two angles reduced to [&minus;180&deg;, 180&deg;]
*
* @tparam T the type of the arguments and returned value.
* @param[in] x the first angle in degrees.
* @param[in] y the second angle in degrees.
* @return \e y &minus; \e x, reduced to the range [&minus;180&deg;,
*   180&deg;].
*
* The result is equivalent to computing the difference exactly, reducing
* it to (&minus;180&deg;, 180&deg;] and rounding the result.  Note that
* this prescription allows &minus;180&deg; to be returned (e.g., if \e x
* is tiny and negative and \e y = 180&deg;).
**********************************************************************/
template<typename T> CUDA_HOST_DEVICE_INLINE static inline T AngDiff(T x, T y)
{
    T e; 
    return AngDiff(x, y, e);
}

template<typename T> CUDA_HOST_DEVICE_INLINE static inline T taupf(T tau, T es) {
    T tau1 = hypot(T(1), tau),
        sig = sinh(eatanhe(tau / tau1, es));
    return hypot(T(1), sig) * tau - sig * tau1;
}

/**
* tan&phi; in terms of tan&chi;
*
* @tparam T the type of the argument and the returned value.
* @param[in] taup &tau;&prime; = tan&chi;
* @param[in] es the signed eccentricity = sign(<i>e</i><sup>2</sup>)
*   sqrt(|<i>e</i><sup>2</sup>|)
* @return &tau; = tan&phi;
*
* See Eqs. (19--21) of
* C. F. F. Karney,
* <a href="https://doi.org/10.1007/s00190-011-0445-3">
* Transverse Mercator with an accuracy of a few nanometers,</a>
* J. Geodesy 85(8), 475--485 (Aug. 2011)
* (preprint <a href="https://arxiv.org/abs/1002.1417">arXiv:1002.1417</a>).
**********************************************************************/
template<typename T> CUDA_HOST_DEVICE static T tauf(T taup, T es) {
    const int numit = 5;
    const T tol = sqrt(DBL_MIN) / T(10);
    T e2m = T(1) - sq(es),
        // To lowest order in e^2, taup = (1 - e^2) * tau = _e2m * tau; so use
        // tau = taup/_e2m as a starting guess.  (This starting guess is the
        // geocentric latitude which, to first order in the flattening, is equal
        // to the conformal latitude.)  Only 1 iteration is needed for |lat| <
        // 3.35 deg, otherwise 2 iterations are needed.  If, instead, tau = taup
        // is used the mean number of iterations increases to 1.99 (2 iterations
        // are needed except near tau = 0).
        tau = taup / e2m,
        stol = tol * max(T(1), abs(taup));
    // min iterations = 1, max iterations = 2; mean = 1.94
    for (int i = 0; i < numit; ++i) {
        T taupa = taupf(tau, es),
            dtau = (taup - taupa) * (1 + e2m * sq(tau)) /
            (e2m * hypot(T(1), tau) * hypot(T(1), taupa));
        tau += dtau;
        if (!(abs(dtau) >= stol))
            break;
    }
    return tau;
}

}//end math
}//end geocuda
