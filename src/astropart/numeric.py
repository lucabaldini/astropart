# Copyright (C) 2024, Luca Baldini (luca.baldini@pi.infn.it)
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

"""Numerical facilities.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
from scipy.optimize import fmin

from astropart import logger
from astropart.plot import plt


class Spline(InterpolatedUnivariateSpline):

    """Small wrapper around the scipy.interpolate.InterpolatedUnivariateSpline class.
    """

    def __init__(self, x: np.array, y: np.array, k: int = 3, ext: int = 0) -> None:
        """Constructor.
        """
        self._x = x
        self._y = y
        self._xmin = x.min()
        self._xmax = x.max()
        super().__init__(x, y, k=k, ext=ext)

    @classmethod
    def from_file(cls, file_path: str, k: int = 3, ext: int = 0, cols: tuple = (0, 1),
        xscale: float = 1., yscale: float = 1.):
        """Create a spline from a text file.
        """
        logger.info(f'Loading data from {file_path}...')
        x, y = np.loadtxt(file_path, usecols=cols, unpack=True)
        x *= xscale
        y *= yscale
        return Spline(x, y, k, ext)

    def norm(self):
        """
        """
        return self.integral(self._xmin, self._xmax)

    def plot(self, num_points: int = 200, logx: bool = False, **kwargs) -> None:
        """
        """
        if logx:
            x = np.geomspace(self._xmin, self._xmax, num_points)
        else:
            x = np.linspace(self._xmin, self._xmax, num_points)
        plt.plot(x, self(x), **kwargs)





def integral(function, xmin: float, xmax: float, args: tuple = (),
    max_err: float = 1.e-6, **kwargs) -> float:
    """Perform a 1-dimensional numerical integration using scipy.integrate.quad

    Parameters
    ----------
    function : callable
        The function to be integrated.

    xmin : float
        The minimum value for the integration interval.

    xmax : float
        The maximum value for the integration interval.

    args : tuple, optional
        Any argument to be passed to the function to be integrated.

    max_err : float
        The maximum allowed estimated relative error on the integral before a
        warning message is printed out.

    Returns
    -------
    float
        The definite integral of the function in the specified domain.
    """
    value, error = quad(function, xmin, xmax, args, **kwargs)
    if error / value > max_err:
        logger.warning(f'Estimated error for the numerical quadrature is {error} / {value}.')
    return value


def find_maximum(function, x0: float, args: tuple = ()) -> float:
    """Return an estimate of the value of the independent variable for which the
    function is maximum.

    Note this uses scipy.optimize.fmin internally, based on a idea at
    https://stackoverflow.com/questions/10146924

    Parameters
    ----------
    function : callable
        The function to be integrated.

    x0 : float
        The initial guess for the maximum.

    args : tuple, optional
        Any argument to be passed to the function.

    Returns
    -------
    float
        The estimate value for which the function is maximum.
    """
    #pylint: disable=invalid-name
    return fmin(lambda x, *args: -function(x, *args), x0, args, disp=0)[0]
