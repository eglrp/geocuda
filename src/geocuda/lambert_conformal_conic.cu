/*!
 * \file lambert_conformal_conic.cu
 *
 * \author Han
 * \date 2017/05/28
 *
 * 墨卡托投影是该投影的一个子集
 */

#include <cuda_runtime.h>
#include <math_functions.h>

#include <cmath>
#include <float.h>

#include <geocuda/math.h>
#include <geocuda/constants.h>
#include <geocuda/lambert_conformal_conic.h>

namespace geocuda{

using namespace std;

LambertConformalConic::LambertConformalConic(double a, double f,
    double stdlat, double k0)
    : eps_(DBL_MIN)
    , epsx_(math::sq(eps_))
    , ahypover_(DBL_DIG * log(double(DBL_RADIX)) + 2)
    , _a(a)
    , _f(f)
    , _fm(1 - _f)
    , _e2(_f * (2 - _f))
    , _es((_f < 0 ? -1 : 1) * sqrt(abs(_e2)))
{
    double sphi, cphi;
    math::sincosd(stdlat, sphi, cphi);
    Init(sphi, cphi, sphi, cphi, k0);
}

LambertConformalConic::LambertConformalConic(double a, double f,
    double stdlat1, double stdlat2,
    double k1)
    : eps_(DBL_MIN)
    , epsx_(math::sq(eps_))
    , ahypover_(DBL_DIG * log(double(DBL_RADIX)) + 2)
    , _a(a)
    , _f(f)
    , _fm(1 - _f)
    , _e2(_f * (2 - _f))
    , _es((_f < 0 ? -1 : 1) * sqrt(abs(_e2)))
{
    double sphi1, cphi1, sphi2, cphi2;
    math::sincosd(stdlat1, sphi1, cphi1);
    math::sincosd(stdlat2, sphi2, cphi2);
    Init(sphi1, cphi1, sphi2, cphi2, k1);
}

LambertConformalConic::LambertConformalConic(double a, double f,
    double sinlat1, double coslat1,
    double sinlat2, double coslat2,
    double k1)
    : eps_(DBL_MIN)
    , epsx_(math::sq(eps_))
    , ahypover_(DBL_DIG * log(double(DBL_RADIX)) + 2)
    , _a(a)
    , _f(f)
    , _fm(1 - _f)
    , _e2(_f * (2 - _f))
    , _es((_f < 0 ? -1 : 1) * sqrt(abs(_e2)))
{
    Init(sinlat1, coslat1, sinlat2, coslat2, k1);
}

void LambertConformalConic::Init(double sphi1, double cphi1,
    double sphi2, double cphi2, double k1) {
        {
            double r;
            r = hypot(sphi1, cphi1);
            sphi1 /= r; cphi1 /= r;
            r = hypot(sphi2, cphi2);
            sphi2 /= r; cphi2 /= r;
        }
        bool polar = (cphi1 == 0);
        cphi1 = max(epsx_, cphi1);   // Avoid singularities at poles
        cphi2 = max(epsx_, cphi2);
        // Determine hemisphere of tangent latitude
        _sign = sphi1 + sphi2 >= 0 ? 1 : -1;
        // Internally work with tangent latitude positive
        sphi1 *= _sign; sphi2 *= _sign;
        if (sphi1 > sphi2) {
            math::swap(sphi1, sphi2);
            math::swap(cphi1, cphi2); // Make phi1 < phi2
        }
        double
            tphi1 = sphi1 / cphi1, tphi2 = sphi2 / cphi2, tphi0;
        //
        // Snyder: 15-8: n = (log(m1) - log(m2))/(log(t1)-log(t2))
        //
        // m = cos(bet) = 1/sec(bet) = 1/sqrt(1+tan(bet)^2)
        // bet = parametric lat, tan(bet) = (1-f)*tan(phi)
        //
        // t = tan(pi/4-chi/2) = 1/(sec(chi) + tan(chi)) = sec(chi) - tan(chi)
        // log(t) = -asinh(tan(chi)) = -psi
        // chi = conformal lat
        // tan(chi) = tan(phi)*cosh(xi) - sinh(xi)*sec(phi)
        // xi = eatanhe(sin(phi)), eatanhe(x) = e * atanh(e*x)
        //
        // n = (log(sec(bet2))-log(sec(bet1)))/(asinh(tan(chi2))-asinh(tan(chi1)))
        //
        // Let log(sec(bet)) = b(tphi), asinh(tan(chi)) = c(tphi)
        // Then n = Db(tphi2, tphi1)/Dc(tphi2, tphi1)
        // In limit tphi2 -> tphi1, n -> sphi1
        //
        double
            tbet1 = _fm * tphi1, scbet1 = hyp(tbet1),
            tbet2 = _fm * tphi2, scbet2 = hyp(tbet2);
        double
            scphi1 = 1 / cphi1,
            xi1 = math::eatanhe(sphi1, _es), shxi1 = sinh(xi1), chxi1 = hyp(shxi1),
            tchi1 = chxi1 * tphi1 - shxi1 * scphi1, scchi1 = hyp(tchi1),
            scphi2 = 1 / cphi2,
            xi2 = math::eatanhe(sphi2, _es), shxi2 = sinh(xi2), chxi2 = hyp(shxi2),
            tchi2 = chxi2 * tphi2 - shxi2 * scphi2, scchi2 = hyp(tchi2),
            psi1 = asinh(tchi1);
        if (tphi2 - tphi1 != 0) {
            // Db(tphi2, tphi1)
            double num = Dlog1p(math::sq(tbet2) / (1 + scbet2),
                math::sq(tbet1) / (1 + scbet1))
                * Dhyp(tbet2, tbet1, scbet2, scbet1) * _fm;
            // Dc(tphi2, tphi1)
            double den = Dasinh(tphi2, tphi1, scphi2, scphi1)
                - Deatanhe(sphi2, sphi1) * Dsn(tphi2, tphi1, sphi2, sphi1);
            _n = num / den;

            if (_n < 0.25)
                _nc = sqrt((1 - _n) * (1 + _n));
            else {
                // Compute nc = cos(phi0) = sqrt((1 - n) * (1 + n)), evaluating 1 - n
                // carefully.  First write
                //
                // Dc(tphi2, tphi1) * (tphi2 - tphi1)
                //   = log(tchi2 + scchi2) - log(tchi1 + scchi1)
                //
                // then den * (1 - n) =
                // (log((tchi2 + scchi2)/(2*scbet2)) - log((tchi1 + scchi1)/(2*scbet1)))
                // / (tphi2 - tphi1)
                // = Dlog1p(a2, a1) * (tchi2+scchi2 + tchi1+scchi1)/(4*scbet1*scbet2)
                //   * fm * Q
                //
                // where
                // a1 = ( (tchi1 - scbet1) + (scchi1 - scbet1) ) / (2 * scbet1)
                // Q = ((scbet2 + scbet1)/fm)/((scchi2 + scchi1)/D(tchi2, tchi1))
                //     - (tbet2 + tbet1)/(scbet2 + scbet1)
                double t;
                {
                    double
                        // s1 = (scbet1 - scchi1) * (scbet1 + scchi1)
                        s1 = (tphi1 * (2 * shxi1 * chxi1 * scphi1 - _e2 * tphi1) -
                            math::sq(shxi1) * (1 + 2 * math::sq(tphi1))),
                        s2 = (tphi2 * (2 * shxi2 * chxi2 * scphi2 - _e2 * tphi2) -
                            math::sq(shxi2) * (1 + 2 * math::sq(tphi2))),
                        // t1 = scbet1 - tchi1
                        t1 = tchi1 < 0 ? scbet1 - tchi1 : (s1 + 1) / (scbet1 + tchi1),
                        t2 = tchi2 < 0 ? scbet2 - tchi2 : (s2 + 1) / (scbet2 + tchi2),
                        a2 = -(s2 / (scbet2 + scchi2) + t2) / (2 * scbet2),
                        a1 = -(s1 / (scbet1 + scchi1) + t1) / (2 * scbet1);
                    t = Dlog1p(a2, a1) / den;
                }
                // multiply by (tchi2 + scchi2 + tchi1 + scchi1)/(4*scbet1*scbet2) * fm
                t *= (((tchi2 >= 0 ? scchi2 + tchi2 : 1 / (scchi2 - tchi2)) +
                    (tchi1 >= 0 ? scchi1 + tchi1 : 1 / (scchi1 - tchi1))) /
                    (4 * scbet1 * scbet2)) * _fm;

                // Rewrite
                // Q = (1 - (tbet2 + tbet1)/(scbet2 + scbet1)) -
                //     (1 - ((scbet2 + scbet1)/fm)/((scchi2 + scchi1)/D(tchi2, tchi1)))
                //   = tbm - tam
                // where
                double tbm = (((tbet1 > 0 ? 1 / (scbet1 + tbet1) : scbet1 - tbet1) +
                    (tbet2 > 0 ? 1 / (scbet2 + tbet2) : scbet2 - tbet2)) /
                    (scbet1 + scbet2));

                // tam = (1 - ((scbet2+scbet1)/fm)/((scchi2+scchi1)/D(tchi2, tchi1)))
                //
                // Let
                //   (scbet2 + scbet1)/fm = scphi2 + scphi1 + dbet
                //   (scchi2 + scchi1)/D(tchi2, tchi1) = scphi2 + scphi1 + dchi
                // then
                //   tam = D(tchi2, tchi1) * (dchi - dbet) / (scchi1 + scchi2)
                double
                    // D(tchi2, tchi1)
                    dtchi = den / Dasinh(tchi2, tchi1, scchi2, scchi1),
                    // (scbet2 + scbet1)/fm - (scphi2 + scphi1)
                    dbet = (_e2 / _fm) * (1 / (scbet2 + _fm * scphi2) +
                        1 / (scbet1 + _fm * scphi1));

                // dchi = (scchi2 + scchi1)/D(tchi2, tchi1) - (scphi2 + scphi1)
                // Let
                //    tzet = chxiZ * tphi - shxiZ * scphi
                //    tchi = tzet + nu
                //    scchi = sczet + mu
                // where
                //    xiZ = eatanhe(1), shxiZ = sinh(xiZ), chxiZ = cosh(xiZ)
                //    nu =   scphi * (shxiZ - shxi) - tphi * (chxiZ - chxi)
                //    mu = - scphi * (chxiZ - chxi) + tphi * (shxiZ - shxi)
                // then
                // dchi = ((mu2 + mu1) - D(nu2, nu1) * (scphi2 + scphi1)) /
                //         D(tchi2, tchi1)
                double
                    xiZ = math::eatanhe(double(1), _es),
                    shxiZ = sinh(xiZ), chxiZ = hyp(shxiZ),
                    // These are differences not divided differences
                    // dxiZ1 = xiZ - xi1; dshxiZ1 = shxiZ - shxi; dchxiZ1 = chxiZ - chxi
                    dxiZ1 = Deatanhe(double(1), sphi1) / (scphi1*(tphi1 + scphi1)),
                    dxiZ2 = Deatanhe(double(1), sphi2) / (scphi2*(tphi2 + scphi2)),
                    dshxiZ1 = Dsinh(xiZ, xi1, shxiZ, shxi1, chxiZ, chxi1) * dxiZ1,
                    dshxiZ2 = Dsinh(xiZ, xi2, shxiZ, shxi2, chxiZ, chxi2) * dxiZ2,
                    dchxiZ1 = Dhyp(shxiZ, shxi1, chxiZ, chxi1) * dshxiZ1,
                    dchxiZ2 = Dhyp(shxiZ, shxi2, chxiZ, chxi2) * dshxiZ2,
                    // mu1 + mu2
                    amu12 = (-scphi1 * dchxiZ1 + tphi1 * dshxiZ1
                        - scphi2 * dchxiZ2 + tphi2 * dshxiZ2),
                    // D(xi2, xi1)
                    dxi = Deatanhe(sphi1, sphi2) * Dsn(tphi2, tphi1, sphi2, sphi1),
                    // D(nu2, nu1)
                    dnu12 =
                    ((_f * 4 * scphi2 * dshxiZ2 > _f * scphi1 * dshxiZ1 ?
                        // Use divided differences
                    (dshxiZ1 + dshxiZ2) / 2 * Dhyp(tphi1, tphi2, scphi1, scphi2)
                        - ((scphi1 + scphi2) / 2
                            * Dsinh(xi1, xi2, shxi1, shxi2, chxi1, chxi2) * dxi) :
                        // Use ratio of differences
                            (scphi2 * dshxiZ2 - scphi1 * dshxiZ1) / (tphi2 - tphi1))
                        + ((tphi1 + tphi2) / 2 * Dhyp(shxi1, shxi2, chxi1, chxi2)
                            * Dsinh(xi1, xi2, shxi1, shxi2, chxi1, chxi2) * dxi)
                        - (dchxiZ1 + dchxiZ2) / 2),
                    // dtchi * dchi
                    dchia = (amu12 - dnu12 * (scphi2 + scphi1)),
                    tam = (dchia - dtchi * dbet) / (scchi1 + scchi2);
                t *= tbm - tam;
                _nc = sqrt(max(double(0), t) * (1 + _n));
            }
            {
                double r = hypot(_n, _nc);
                _n /= r;
                _nc /= r;
            }
            tphi0 = _n / _nc;
        }
        else {
            tphi0 = tphi1;
            _nc = 1 / hyp(tphi0);
            _n = tphi0 * _nc;
            if (polar)
                _nc = 0;
        }

        _scbet0 = hyp(_fm * tphi0);
        double shxi0 = sinh(math::eatanhe(_n, _es));
        _tchi0 = tphi0 * hyp(shxi0) - shxi0 * hyp(tphi0); _scchi0 = hyp(_tchi0);
        _psi0 = asinh(_tchi0);

        _lat0 = atan(_sign * tphi0) * rad2deg();
        _t0nm1 = expm1(-_n * _psi0); // Snyder's t0^n - 1
                                           // a * k1 * m1/t1^n = a * k1 * m2/t2^n = a * k1 * n * (Snyder's F)
                                           // = a * k1 / (scbet1 * exp(-n * psi1))
        _scale = _a * k1 / scbet1 *
            // exp(n * psi1) = exp(- (1 - n) * psi1) * exp(psi1)
            // with (1-n) = nc^2/(1+n) and exp(-psi1) = scchi1 + tchi1
            exp(-(math::sq(_nc) / (1 + _n)) * psi1)
            * (tchi1 >= 0 ? scchi1 + tchi1 : 1 / (scchi1 - tchi1));
        // Scale at phi0 = k0 = k1 * (scbet0*exp(-n*psi0))/(scbet1*exp(-n*psi1))
        //                    = k1 * scbet0/scbet1 * exp(n * (psi1 - psi0))
        // psi1 - psi0 = Dasinh(tchi1, tchi0) * (tchi1 - tchi0)
        _k0 = k1 * (_scbet0 / scbet1) *
            exp(-(math::sq(_nc) / (1 + _n)) *
                Dasinh(tchi1, _tchi0, scchi1, _scchi0) * (tchi1 - _tchi0))
            * (tchi1 >= 0 ? scchi1 + tchi1 : 1 / (scchi1 - tchi1)) /
            (_scchi0 + _tchi0);
        _nrho0 = polar ? 0 : _a * _k0 / _scbet0;
        {
            // Figure _drhomax using code at beginning of Forward with lat = -90
            double
                sphi = -1, cphi = epsx_,
                tphi = sphi / cphi,
                scphi = 1 / cphi, shxi = sinh(math::eatanhe(sphi, _es)),
                tchi = hyp(shxi) * tphi - shxi * scphi, scchi = hyp(tchi),
                psi = asinh(tchi),
                dpsi = Dasinh(tchi, _tchi0, scchi, _scchi0) * (tchi - _tchi0);
            _drhomax = -_scale * (2 * _nc < 1 && dpsi != 0 ?
                (exp(math::sq(_nc) / (1 + _n) * psi) *
                (tchi > 0 ? 1 / (scchi + tchi) : (scchi - tchi))
                    - (_t0nm1 + 1)) / (-_n) :
                Dexp(-_n * psi, -_n * _psi0) * dpsi);
        }
}

void LambertConformalConic::Forward(double lon0, double lat, double lon,
    double& x, double& y, double& gamma, double& k)
    const {
    lon = math::AngDiff(lon0, lon);
    // From Snyder, we have
    //
    // theta = n * lambda
    // x = rho * sin(theta)
    //   = (nrho0 + n * drho) * sin(theta)/n
    // y = rho0 - rho * cos(theta)
    //   = nrho0 * (1-cos(theta))/n - drho * cos(theta)
    //
    // where nrho0 = n * rho0, drho = rho - rho0
    // and drho is evaluated with divided differences
    double sphi, cphi;
    math::sincosd(math::latfix(lat) * _sign, sphi, cphi);
    cphi = max(epsx_, cphi);
    double
        lam = lon * math::degree(),
        tphi = sphi / cphi, scbet = hyp(_fm * tphi),
        scphi = 1 / cphi, shxi = sinh(math::eatanhe(sphi, _es)),
        tchi = hyp(shxi) * tphi - shxi * scphi, scchi = hyp(tchi),
        psi = asinh(tchi),
        theta = _n * lam, stheta = sin(theta), ctheta = cos(theta),
        dpsi = Dasinh(tchi, _tchi0, scchi, _scchi0) * (tchi - _tchi0),
        drho = -_scale * (2 * _nc < 1 && dpsi != 0 ?
        (exp(math::sq(_nc) / (1 + _n) * psi) *
            (tchi > 0 ? 1 / (scchi + tchi) : (scchi - tchi))
            - (_t0nm1 + 1)) / (-_n) :
            Dexp(-_n * psi, -_n * _psi0) * dpsi);
    x = (_nrho0 + _n * drho) * (_n ? stheta / _n : lam);
    y = _nrho0 *
        (_n ?
        (ctheta < 0 ? 1 - ctheta : math::sq(stheta) / (1 + ctheta)) / _n : 0)
        - drho * ctheta;
    k = _k0 * (scbet / _scbet0) /
        (exp(-(math::sq(_nc) / (1 + _n)) * dpsi)
            * (tchi >= 0 ? scchi + tchi : 1 / (scchi - tchi)) / (_scchi0 + _tchi0));
    y *= _sign;
    gamma = _sign * theta * rad2deg();
}

void LambertConformalConic::Reverse(double lon0, double x, double y,
    double& lat, double& lon,
    double& gamma, double& k)
    const {
    // From Snyder, we have
    //
    //        x = rho * sin(theta)
    // rho0 - y = rho * cos(theta)
    //
    // rho = hypot(x, rho0 - y)
    // drho = (n*x^2 - 2*y*nrho0 + n*y^2)/(hypot(n*x, nrho0-n*y) + nrho0)
    // theta = atan2(n*x, nrho0-n*y)
    //
    // From drho, obtain t^n-1
    // psi = -log(t), so
    // dpsi = - Dlog1p(t^n-1, t0^n-1) * drho / scale
    y *= _sign;
    double
        // Guard against 0 * inf in computation of ny
        nx = _n * x, ny = _n ? _n * y : 0, y1 = _nrho0 - ny,
        den = hypot(nx, y1) + _nrho0, // 0 implies origin with polar aspect
                                            // isfinite test is to avoid inf/inf
        drho = ((den != 0 && isfinite(den))
            ? (x*nx + y * (ny - 2 * _nrho0)) / den
            : den);
    drho = min(drho, _drhomax);
    if (_n == 0)
        drho = max(drho, -_drhomax);
    double
        tnm1 = _t0nm1 + _n * drho / _scale,
        dpsi = (den == 0 ? 0 :
        (tnm1 + 1 != 0 ? -Dlog1p(tnm1, _t0nm1) * drho / _scale :
            ahypover_));
    double tchi;
    if (2 * _n <= 1) {
        // tchi = sinh(psi)
        double
            psi = _psi0 + dpsi, tchia = sinh(psi), scchi = hyp(tchia),
            dtchi = Dsinh(psi, _psi0, tchia, _tchi0, scchi, _scchi0) * dpsi;
        tchi = _tchi0 + dtchi;    // Update tchi using divided difference
    }
    else {
        // tchi = sinh(-1/n * log(tn))
        //      = sinh((1-1/n) * log(tn) - log(tn))
        //      = + sinh((1-1/n) * log(tn)) * cosh(log(tn))
        //        - cosh((1-1/n) * log(tn)) * sinh(log(tn))
        // (1-1/n) = - nc^2/(n*(1+n))
        // cosh(log(tn)) = (tn + 1/tn)/2; sinh(log(tn)) = (tn - 1/tn)/2
        double
            tn = tnm1 + 1 == 0 ? epsx_ : tnm1 + 1,
            sh = sinh(-math::sq(_nc) / (_n * (1 + _n)) *
            (2 * tn > 1 ? log1p(tnm1) : log(tn)));
        tchi = sh * (tn + 1 / tn) / 2 - hyp(sh) * (tnm1 * (tn + 1) / tn) / 2;
    }

    // log(t) = -asinh(tan(chi)) = -psi
    gamma = atan2(nx, y1);
    double
        tphi = math::tauf(tchi, _es),
        scbet = hyp(_fm * tphi), scchi = hyp(tchi),
        lam = _n ? gamma / _n : x / y1;
    lat = math::atand(_sign * tphi);
    lon = lam * rad2deg();
    lon = math::AngNormalize(lon + math::AngNormalize(lon0));
    k = _k0 * (scbet / _scbet0) /
        (exp(_nc ? -(math::sq(_nc) / (1 + _n)) * dpsi : 0)
            * (tchi >= 0 ? scchi + tchi : 1 / (scchi - tchi)) / (_scchi0 + _tchi0));
    gamma /= _sign * math::degree();
}

void LambertConformalConic::SetScale(double lat, double k) {
    double x, y, gamma, kold;
    Forward(0, lat, 0, x, y, gamma, kold);
    k /= kold;
    _scale *= k;
    _k0 *= k;
}
}