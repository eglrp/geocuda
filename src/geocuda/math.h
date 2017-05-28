/*!
 * \file math.h
 *
 * \author Han
 * \date 2017/05/28
 *
 * 常用的数学函数
 */

#include <math_functions.h>
#include <geocuda/constants.h>

namespace geocuda {
namespace math {
template <typename T>
CUDA_HOST_DEVICE_INLINE T sq(T x) { return x * x; }


template<typename T> CUDA_HOST_DEVICE_INLINE void swap(T& x, T&y) {
#ifdef __CUDACC__
    T z(y);
    y = x;
    x = z;
#else
    std::swap(x, y);
#endif // __CUDACC__
}

template<typename T> CUDA_HOST_DEVICE_INLINE T latfix(T x)
{
    return abs(x) > 90.0 ? nan("") : x;
}

template<typename T> CUDA_HOST_DEVICE_INLINE void sincosd(T x, T& sinx, T& cosx) {
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
template<typename T> CUDA_HOST_DEVICE_INLINE T atan2d(T y, T x) {
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
}//end math
}//end geocuda
