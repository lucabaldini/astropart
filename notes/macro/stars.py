# Copyright (C) 2023, Luca Baldini (luca.baldini@pi.infn.it)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Utilities related to the properties of light.

This includes all the functions related to magnitude, color index, and spectra.
"""

from collections.abc import Callable

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from plotting import plt, setup_ca, DATA_FOLDER_PATH


# Shorthands for some useful physical constants (in cgs).
c = const.c.cgs.value
h = const.h.cgs.value
kB = const.k_B.cgs.value
hbar = h / 2. / np.pi

# A few useful numbers for the filters. All values are expressed in nm.
# This is mainly from https://en.wikipedia.org/wiki/Photometric_system
VISIBLE_RANGE = (380., 750.)
IR_RANGE = (700., 1.e6)
UV_RANGE = (100, 380.)
U_FILTER_CENTER = 365.
U_FILTER_FWHM = 66.
B_FILTER_CENTER = 445.
B_FILTER_FWHM = 94.
V_FILTER_CENTER = 551.
V_FILTER_FWHM = 88.
R_FILTER_CENTER = 658.
R_FILTER_FWHM = 138.
I_FILTER_CENTER = 806.
I_FILTER_FWHM = 149.

def _mag(value : float, zero_point : float) -> float:
    """Basic magnitude definition.
    """
    return -2.5 * np.log10(value / zero_point)

def _isl(value : float, distance : float) -> float:
    """Inverse square law.
    """
    return value / 4. / np.pi / distance**2.


# Basic Sun characteristics---these are all numerical values in the cgs system.
# A few notes from https://www.iau.org/static/resolutions/IAU2015_English.pdf
# The zero point of the absolute bolometric magnitude scale is L0 = 3.0128e28 W
# This implies that the nominal solar luminosity (3.828e26 W) correspond to a
# bolometric magnitude of 4.74.
# The zero point for the apparent bolometric magnitude is 2.518021002e-8 W m^{-2}
# (this can be calculated from L0 at 10 pc)
# This implies that the nominal solar irradiance (1361 W m^{-2}) corresponds to
# an apparent bolometric magnitude of -26.832
# Also, the effective temperature of the Sun is quoted as 5772 degrees K.
ABSOLUTE_BOLOMETRIC_ZERO_POINT = 3.0128e35
STANDARD_DISTANCE = (10. * u.pc).cgs.value
AU = (1. * u.au).cgs.value
APPARENT_BOLOMETRIC_ZERO_POINT = _isl(ABSOLUTE_BOLOMETRIC_ZERO_POINT, STANDARD_DISTANCE)
SUN_MASS = const.M_sun.cgs.value
SUN_RADIUS = const.R_sun.cgs.value
SUN_EFFECTIVE_TEMPERATURE = 5772.
SUN_LUMINOSITY = const.L_sun.cgs.value
SUN_IRRADIANCE = _isl(SUN_LUMINOSITY, AU)
SUN_ABSOLUTE_BOLOMETRIC_MAGNITUDE = _mag(SUN_LUMINOSITY, ABSOLUTE_BOLOMETRIC_ZERO_POINT)
SUN_APPARENT_BOLOMETRIC_MAGNITUDE = _mag(SUN_IRRADIANCE, APPARENT_BOLOMETRIC_ZERO_POINT)
SUN_APPARENT_VISUAL_MAGNITUDE = -26.76
SUN_BOLOMETRIC_CORRECTION = SUN_APPARENT_BOLOMETRIC_MAGNITUDE - SUN_APPARENT_VISUAL_MAGNITUDE
SUN_ABSOLUTE_VISUAL_MAGNITUDE = SUN_ABSOLUTE_BOLOMETRIC_MAGNITUDE - SUN_BOLOMETRIC_CORRECTION
SUN_BU_COLOR_INDEX = 0.656

# Deafault grids for interpolation.
DEFAULT_BV_GRID = np.linspace(2.5, -0.65, 250)
DEFAULT_T_GRID = np.geomspace(2000., 60000., 250)


def luminosity_to_magnitude(L : float) -> float:
    """Convert a luminosity (expressed in L_sun) to absolute bolometric magnitude.

    Arguments
    ---------
    L : float
        The luminosity expressed in units of L_sun.
    """
    return _mag(L, SUN_ABSOLUTE_BOLOMETRIC_MAGNITUDE)

def magnitude_to_luminosity(M : float) -> float:
    """Convert an absolute bolometric magnitude to luminosity (expressed in L_sun).

    Arguments
    ---------
    M : float
        The absolute bolometric magnitude.
    """
    return 10.**((-M + SUN_ABSOLUTE_BOLOMETRIC_MAGNITUDE) / 2.5)

def planck_nu(nu : float, T : float) -> float:
    """Return the Planck radiance at a given temperature as a function of the frequency.

    Arguments
    ---------
    nu : array_like
        The frequency in THz.

    T : float
        The temperature in degrees K.
    """
    nu = u.THz.to(u.Hz, nu)
    x = h * nu / kB / T
    return 2. * h * nu**3. / c**2. / (np.exp(x) - 1.)

def planck_lambda(lambda_ : float, T : float) -> float:
    """Return the Planck radiance at a given temperature as a function of the wavelength.

    Arguments
    ---------
    lambda_ : array_like
        The wavelength in nm.

    T : float
        The temperature in degrees K.
    """
    lambda_ = u.nm.to(u.cm, lambda_)
    x = h * c / lambda_ / kB / T
    return 2. * h * c**2. / lambda_**5. / (np.exp(x) - 1.)

def wien_nu(T : float) -> float:
    """Return the peak frequency (in THz) at a temperature T according to the Wien law
    (in the wavelength formulation of the Planck law).

    Arguments
    ---------
    T : float
        The temperature in degrees K.
    """
    return 5.879e-2 * T

def wien_lambda(T : float) -> float:
    """Return the peak wavelength (in nm) at a temperature T according to the Wien law
    (in the wavelength formulation of the Planck law).

    Arguments
    ---------
    T : float
        The temperature in degrees K.
    """
    return u.cm.to(u.nm, 0.2897771955185172661 / T)

def stefan_boltzman(surface : float, T : float) -> float:
    """Stefan-Boltzmann law.

    Arguments
    ---------
    surface : float
        The emitting surface in cm^2.

    T : float
        The temperature in degrees K.
    """
    return const.sigma_sb.cgs.value * surface * T**4.

def _polynomial(x, *coefficients):
    """Simple polynomial factory.

    Given a set of coefficients c0, c1, c2, ... , cn, this is returning
    c0 + c1 * x + c2 * x**2. + ...  + cn * x**(n - 1)

    Arguments
    ---------
    x : array_like
        The values of the independent variable.

    coefficients : float
        The polynomial coefficients.
    """
    return sum(c * x**i for i, c in enumerate(coefficients))

def _invspline(f : Callable[[np.array], np.array], x : np.array, k : int = 3):
    """Invert an arbitrary function via a spline interpolation.

    Arguments
    ---------
    f : callable
        The function to invert.

    x : array_like
        The array of x-values to which f is applied for the inversion.

    k : int
        The spline order.
    """
    _spline = InterpolatedUnivariateSpline(f(x), x, k=k)
    return lambda val : _spline(val)

def _flower_bc_lt(logT : float) -> float:
    """Bolometric correction according to Flower (1996), table 6, as amended in
    Torres (2010), table 1, for the low-temperature part of the parametrization.

    Arguments
    ---------
    logT : array_like
        The logarithm of the temperature in degrees K.
    """
    coefficients = (
        -0.190537291496456e+05,
        0.155144866764412e+05,
        -0.421278819301717e+04,
        0.381476328422343e+03
    )
    return _polynomial(logT, *coefficients)

def _flower_bc_mt(logT: float) -> float:
    """Bolometric correction according to Flower (1996), table 6, as amended in
    Torres (2010), table 1, for the mid-temperature part of the parametrization.


    Arguments
    ---------
    logT : array_like
        The logarithm of the temperature in degrees K.
    """
    coefficients = (
        -0.370510203809015e+05,
        0.385672629965804e+05,
        -0.150651486316025e+05,
        0.261724637119416e+04,
        -0.170623810323864e+03
    )
    return _polynomial(logT, *coefficients)

def _flower_bc_ht(logT : float) -> float:
    """Bolometric correction according to Flower (1996), table 6, as amended in
    Torres (2010), table 1, for the high-temperature part of the parametrization.


    Arguments
    ---------
    logT : array_like
        The logarithm of the temperature in degrees K.
    """
    coefficients = (
        -0.118115450538963e+06,
        0.137145973583929e+06,
        -0.636233812100225e+05,
        0.147412923562646e+05,
        -0.170587278406872e+04,
        0.788731721804990e+02
    )
    return _polynomial(logT, *coefficients)

def bolometric_correction_flower(T : float) -> float:
    """Implementation of the bolometric correction from Flower (1996), table 6,
    as amended in Torres (2010), table 1.

    Reference: https://iopscience.iop.org/article/10.1088/0004-6256/140/5/1158/pdf

    Arguments
    ---------
    T : array_like
        The Effective temperature in degrees K.
    """
    logT = np.log10(T)
    return _flower_bc_lt(logT) * (logT < 3.70) +\
        _flower_bc_mt(logT) * (logT >= 3.70) * (logT < 3.90) +\
        _flower_bc_ht(logT) * (logT > 3.90)

def bolometric_correction_reed(T : float) -> float:
    """Implementation of the bolometric correction from Reed (1998), eq. (5).

    Note that the original paper is wrong, and we need the transformation
    log(T) -> log(T) - 4 for this to work.

    Arguments
    ---------
    T : array_like
        The Effective temperature in degrees K.
    """
    logT = np.log10(T) - 4.
    coefficients = (
        - 0.438,
        - 3.901,
        - 8.131,
        13.421,
        -8.499
    )
    return _polynomial(logT, *coefficients)

def bolometric_correction(T : float) -> float:
    """Default bolometric_correction.

    This is essentially taken from Reed (1998), and slightly adjusted to match
    the target bolometric correction for the sun.
    """
    delta = bolometric_correction_reed(SUN_EFFECTIVE_TEMPERATURE) - SUN_BOLOMETRIC_CORRECTION
    return bolometric_correction_reed(T) - delta

def _reed_bv_lt(logT : float) -> float:
    """Implementation of the conversion from temperature to (B - V) color index
    from Reed (1998), eq. (4) for log T < 3.961.

    Arguments
    ---------
    logT : array_like
        The logarithm of the temperature in degrees K.
    """
    coefficients = (
        14.551,
        -3.684
    )
    return _polynomial(logT, *coefficients)

def _reed_bv_ht(logT : float) -> float:
    """Implementation of the conversion from temperature to (B - V) color index
    from Reed (1998), eq. (4) for log T >= 3.961

    Arguments
    ---------
    logT : array_like
        The logarithm of the temperature in degrees K.
    """
    coefficients = (
        8.037,
        -3.402,
        0.344
    )
    return _polynomial(logT, *coefficients)

def temperature_to_bv_reed(T : float) -> float:
    """Implementation of the conversion from effective temperature to (B - V)
    color index from Reed (1998), eq. (4).

    Arguments
    ---------
    logT : array_like
        The temperature in degrees K.
    """
    logT = np.log10(T)
    logT0 = 3.961
    return _reed_bv_lt(logT) * (logT < logT0) + _reed_bv_ht(logT) * (logT >= logT0)

def bv_to_temperature_flower(bv : float) -> float:
    """Implementation of the conversion from (B - V) color index to effective
    temperature from Flower (1996), table 5, as amended in Torres (2010), table 2.

    Arguments
    ---------
    bv : array_like
        The (B - V) color index.
    """
    coefficients = (
        3.979145106714099,
        -0.654992268598245,
        1.740690042385095,
        -4.608815154057166,
        6.792599779944473,
        -5.396909891322525,
        2.192970376522490,
        -0.359495739295671
    )
    return 10.**_polynomial(bv, *coefficients)

temperature_to_bv_flower = _invspline(bv_to_temperature_flower, DEFAULT_BV_GRID)

def bv_to_temperature_ballesteros(bv : float) -> float:
    """Implementation of the conversion from (B - V) color index to effective
    temperature from Ballesteros (2012).

    Reference: See https://ui.adsabs.harvard.edu/abs/2012EL.....9734008B/abstract
    """
    return 4600. * (1. / (0.92 * bv + 1.7) + 1. / (0.92 * bv + 0.62))

temperature_to_bv_ballesteros = _invspline(bv_to_temperature_ballesteros, DEFAULT_BV_GRID)

def _temperature_to_bv(T : float, tmin : float = 5000., tmax : float = 7000.) -> float:
    """Combination of the variuos compilations.

    This should be smooth across all temperatures of interestm and is the
    default parametrization that we shall use.
    """
    # Define an array of weights that transitions linearly from 0 to 1 between
    # a given tmin and tmax.
    w = np.clip((T - tmin) / (tmax - tmin), 0., 1.)
    # We use the compilation by Reed at low temperature and that by Flower at
    # high temperature.
    return temperature_to_bv_reed(T) * (1. - w) + temperature_to_bv_flower(T) * w

def temperature_to_bv(T : float) -> float:
    """Combination of the variuos compilations.

    This should be smooth across all temperatures of interestm and is the
    default parametrization that we shall use.
    """
    delta = _temperature_to_bv(SUN_EFFECTIVE_TEMPERATURE) - SUN_BU_COLOR_INDEX
    return _temperature_to_bv(T) - delta



def test_sun():
    """Test the basic solar figures.
    """
    print(f'Absolute bolometric zero point: {ABSOLUTE_BOLOMETRIC_ZERO_POINT:.4e} erg s^{-1}')
    print(f'Apparent bolometric zero point: {APPARENT_BOLOMETRIC_ZERO_POINT:.4e} erg s^{-1} cm^{-2}')
    print(f'Sun luminosity: {SUN_LUMINOSITY:.4e} erg s^{-1}')
    print(f'Sun irradiance: {SUN_IRRADIANCE:.4e} erg s^{-1} cm^{-2}')
    print(f'Sun absolute bolometric magnitude: {SUN_ABSOLUTE_BOLOMETRIC_MAGNITUDE:.3f}')
    assert np.allclose(SUN_ABSOLUTE_BOLOMETRIC_MAGNITUDE, 4.74)
    print(f'Sun apparent bolometric magnitude: {SUN_APPARENT_BOLOMETRIC_MAGNITUDE:.3f}')
    assert np.allclose(SUN_APPARENT_BOLOMETRIC_MAGNITUDE, -26.832)
    print(f'Sun absolute visual magnitude: {SUN_ABSOLUTE_VISUAL_MAGNITUDE:.3f}')
    print(f'Sun apparent visual magnitude: {SUN_APPARENT_VISUAL_MAGNITUDE:.3f}')
    print(f'Sun (B - V) color index: {SUN_BU_COLOR_INDEX:.3f}')

def test_wien():
    """Unit test for the Wien law.
    """
    plt.figure('Wien law')
    T = 10000.
    lambda_ = np.linspace(100., 1500., 100)
    plt.plot(lambda_, planck_lambda(lambda_, T))
    plt.axvline(wien_lambda(T), ls='dashed')
    setup_ca(xlabel='wavelength [nm]', ylabel='Irradiance', ymin=0.)

def test_bolometric_correction():
    """Small unit test for the bolometric correction.
    """
    plt.figure('Bolometric correction')
    T = DEFAULT_T_GRID
    plt.plot(T, bolometric_correction_reed(T), label='Reed (1998)')
    x, y = np.loadtxt(DATA_FOLDER_PATH / 'reed_1998_bc.txt', delimiter=',', unpack=True)
    plt.scatter(10.**x, y, s=10.)
    plt.plot(T, bolometric_correction_flower(T), label='Flower (1996)')
    x, y = np.loadtxt(DATA_FOLDER_PATH / 'flower_1996_bc.txt', delimiter=',', unpack=True)
    plt.scatter(10.**x, y, s=10.)
    plt.plot(T, bolometric_correction(T), label='Combination')
    setup_ca(logx=True, xlabel='Effective temperature [K]', ylabel='Bolometric correction',
        xmin=1000., xmax=100000., legend=True)

def test_color_index():
    """Small unit test for the color index conversion.
    """
    plt.figure('(B - V) color index')
    T = DEFAULT_T_GRID
    plt.plot(T, temperature_to_bv_reed(T), label='Reed (1998)')
    x, y = np.loadtxt(DATA_FOLDER_PATH / 'reed_1998_bv.txt', delimiter=',', unpack=True)
    plt.scatter(10.**x, y, s=10.)
    plt.plot(T, temperature_to_bv_flower(T), label='Flower (1996)')
    y, x = np.loadtxt(DATA_FOLDER_PATH / 'flower_1996_bv.txt', delimiter=',', unpack=True)
    plt.scatter(10.**x, y, s=10.)
    plt.plot(T, temperature_to_bv_ballesteros(T), label='Ballesteros (2012)')
    plt.plot(T, temperature_to_bv(T), label='Combination')
    setup_ca(logx=True, xlabel='Effective temperature [K]', ylabel='(B - V) color index',
        xmin=1000., xmax=100000., legend=True)




if __name__ == '__main__':
    test_sun()
    test_wien()
    test_bolometric_correction()
    test_color_index()
    plt.show()
