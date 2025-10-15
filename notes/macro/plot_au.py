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

"""Plotting data
from https://articles.adsabs.harvard.edu/pdf/2001JAHH....4...15H
"""

from dataclasses import dataclass
from enum import Enum, auto

import astropy.units as u
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import curve_fit

from plotting import plt, standard_figure, setup_ca, save_cf


def parallax_to_au(value):
    """Convert a value for the Solar parallax to au.

    This is following, verbatim, the prescription in table 1 of the paper
    """
    return u.R_earth.to(u.au) / np.tan(u.arcsec.to(u.rad, value))



class AuMeasType(Enum):

    """Small enum class for the different types of measurements.
    """

    VALUE = auto()
    LOWER_LIMIT = auto()


class AuMeasMethod(Enum):

    """Small enum class for the different methods of measurements.
    """

    LUNAR_DICOTHOMY = 'Lunar dicothomy'
    PLANETARY_DIAMETERS = 'Planetary diameters'
    MARS_PARALLAX = 'Mars parallax'
    MERCURY_TRANSIT = 'Mercury transit'
    VENUS_TRANSIT = 'Venus transit'
    C_ABERRATION = 'c + aberration'
    MOON_PARALLAX_INEQUALITY = 'Parallactic inequality of the Moon'
    JUNO_PARALLAX = 'Juno parallax'
    ASTEROID_PARALLAX = 'Asteroid parallax'


@dataclass
class AuMeasurement:

    """Small container class representing a historical measurement of the AU.
    """
    author : str
    year : int
    value : float
    method : AuMeasMethod = None
    type_ : AuMeasType = AuMeasType.VALUE

    _MARKER_DICT = {
        AuMeasMethod.LUNAR_DICOTHOMY: dict(marker='^', fillstyle='none'),
        AuMeasMethod.MARS_PARALLAX: dict(marker='o', fillstyle='none'),
        AuMeasMethod.VENUS_TRANSIT: dict(marker='s', fillstyle='none'),
        AuMeasMethod.C_ABERRATION: dict(marker='v', fillstyle='none')
    }

    @staticmethod
    def _marker_style(method):
        """Return the proper marker style for a given measurement.
        """
        default = dict(marker='o', fillstyle='full')
        return AuMeasurement._MARKER_DICT.get(method, default)

    def plot(self, ax=None):
        """Plot a single measurement.
        """
        if ax is None:
            ax = plt.gca()
        fmt = dict(color='k')
        if self.type_ == AuMeasType.LOWER_LIMIT:
            ax.errorbar(self.year, self.value, xerr=10, yerr=0.075, lolims=True, **fmt)
            return
        marker_style = self._marker_style(self.method)
        try:
            min_, max_ = self.value
            y = 0.5 * (min_ + max_)
            dy = 0.5 * (max_ - min_)
            ax.errorbar(self.year, y, yerr=dy, **fmt, **marker_style)
        except:
            ax.plot(self.year, self.value, **fmt, **marker_style)


def _range_args(min_parallax, max_parallax):
    """Small convenience function to provide error bars where the original
    measurement is provided as a range in Solar parallax.

    Note that min and max must be swapped.
    """
    return parallax_to_au(max_parallax), parallax_to_au(min_parallax)


measurements = (
    AuMeasurement('Kepler', 1604, 0.14657, type_=AuMeasType.LOWER_LIMIT),
    AuMeasurement('Wendelin', 1626, 0.14657, AuMeasMethod.LUNAR_DICOTHOMY, AuMeasType.LOWER_LIMIT),
    AuMeasurement('Wendelin', 1630, 0.59689),
    AuMeasurement('Wendelin', 1635, 0.62815, AuMeasMethod.PLANETARY_DIAMETERS),
    AuMeasurement('Horrocks', 1640, 0.62815, AuMeasMethod.PLANETARY_DIAMETERS),
    AuMeasurement('Langrenus', 1650, 0.14591),
    AuMeasurement('Riccioli', 1651, 0.31124, AuMeasMethod.LUNAR_DICOTHOMY),
    AuMeasurement('Huygens', 1659, 1.07246),
    AuMeasurement('Hevelius', 1662, 0.21985),
    AuMeasurement('Picard and Cassini', 1669, 0.73286, type_=AuMeasType.LOWER_LIMIT),
    AuMeasurement('Richer and Cassini', 1672, 0.92570, AuMeasMethod.MARS_PARALLAX),
    AuMeasurement('Lahire', 1672, 1.46569, type_=AuMeasType.LOWER_LIMIT),
    AuMeasurement('Flamsteed', 1673, 0.87941, type_=AuMeasType.LOWER_LIMIT),
    AuMeasurement('Halley', 1678, 0.19543, AuMeasMethod.MERCURY_TRANSIT),
    AuMeasurement('Flamsteed', 1681, 0.58628, type_=AuMeasType.LOWER_LIMIT),
    AuMeasurement('Whiston', 1696, 0.58628),
    AuMeasurement('Whiston', 1701, 0.87941),
    AuMeasurement('Maraldi', 1704, 0.87941, 'Mars Parallax'),
    AuMeasurement('Pond and Bradely', 1717, _range_args(9., 12.), AuMeasMethod.MARS_PARALLAX),
    AuMeasurement('Maraldi', 1719, _range_args(11., 15.), AuMeasMethod.MARS_PARALLAX),
    AuMeasurement('De Lacaille', 1751, 0.84418),
    AuMeasurement('Delises', 1760, _range_args(10., 14.), AuMeasMethod.VENUS_TRANSIT),
    AuMeasurement('De Lacaille', 1761, 0.93158),
    AuMeasurement('De Lacaille', 1769, 1.02069),
    AuMeasurement('Delambre', 1770, 1.02258),
    AuMeasurement('Henderson', 1832, 0.96374, AuMeasMethod.MARS_PARALLAX),
    AuMeasurement('Hansen', 1854, 0.98040),
    AuMeasurement('Foucault', 1862, 0.99257, AuMeasMethod.C_ABERRATION),
    AuMeasurement('Hansen', 1863, 0.98589, AuMeasMethod.MOON_PARALLAX_INEQUALITY),
    AuMeasurement('Newcomb', 1867, 0.99391, AuMeasMethod.MARS_PARALLAX),
    AuMeasurement('Leverrier', 1872, 0.98259, AuMeasMethod.MOON_PARALLAX_INEQUALITY),
    AuMeasurement('Stone', 1872, 0.987),
    AuMeasurement('Comu', 1874, 1.00002, AuMeasMethod.C_ABERRATION),
    AuMeasurement('Stone', 1874, _range_args(8.84, 8.92), AuMeasMethod.VENUS_TRANSIT),
    AuMeasurement('Airy', 1874, 1.00459, AuMeasMethod.VENUS_TRANSIT),
    AuMeasurement('Puiseux', 1875, 0.99044, AuMeasMethod.VENUS_TRANSIT),
    AuMeasurement('Todd', 1875, _range_args(8.85, 8.91), AuMeasMethod.VENUS_TRANSIT),
    AuMeasurement('Lindsat and Gill', 1877, 0.99763, AuMeasMethod.JUNO_PARALLAX),
    AuMeasurement('Tupman', 1878, 0.99763, AuMeasMethod.VENUS_TRANSIT),
    AuMeasurement('Hall', 1879, _range_args(8.82, 8.94)),
    AuMeasurement('Downing', 1880, _range_args(8.77, 8.81)),
    AuMeasurement('Todd', 1880, 0.99899, AuMeasMethod.C_ABERRATION),
    AuMeasurement('Gill', 1881, _range_args(8.66, 8.90), AuMeasMethod.MARS_PARALLAX),
    AuMeasurement('Todd', 1881, _range_args(8.85, 8.91), AuMeasMethod.VENUS_TRANSIT),
    AuMeasurement('Faye', 1881, 0.99786, AuMeasMethod.C_ABERRATION),
    AuMeasurement('Newcomb', 1885, 0.99877, AuMeasMethod.C_ABERRATION),
    AuMeasurement('Gill', 1890, 0.99911, AuMeasMethod.ASTEROID_PARALLAX),
)


standard_figure('AU measurements')
for m in measurements:
    m.plot()
setup_ca(xmin=1590., xmax=1910., ymin=0., ymax=1.59, xlabel='Year', ylabel='Astronomical unit [au]')
plt.axhline(1., ls='dashed', color='k')
plt.axhline(0.05, xmax=0.5, ls='dashed', color='k')
plt.text(1600, 0.06, 'Aristarchus (280 BC)')
axins = plt.gca().inset_axes([0.575, 0.1, 0.375, 0.30])
for m in measurements:
    m.plot(axins)
axins.set_xlim(1860, 1900)
axins.set_ylim(0.975, 1.025)
axins.grid(which='both')
plt.gca().indicate_inset_zoom(axins, edgecolor='black', lw=1.)
handles = []
for method, style in AuMeasurement._MARKER_DICT.items():
    handles.append(plt.plot([], [], color='k', label=method.value, linestyle='none', **style)[0])
plt.legend(handles=handles, fontsize='small')
save_cf()


if __name__ == '__main__':
    plt.show()
