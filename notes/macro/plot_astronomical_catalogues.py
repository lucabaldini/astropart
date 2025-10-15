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

"""Poor man attempt at a sensible rendering of the history of astronomical catalogues.
"""

from dataclasses import dataclass

from matplotlib.lines import Line2D
import numpy as np

from plotting import plt, standard_figure, setup_ca, save_cf

@dataclass
class Catalog:

    """Small container class representing a catalog.
    """

    name : str
    observer : str
    year : str
    size : int
    reference : str = None
    notes : str = None

    def plot(self, label=None, **kwargs):
        """Plot the catalog.
        """
        dx = kwargs.get('dx', 8.)
        ha = kwargs.get('ha', 'left')
        va = kwargs.get('va', 'center')
        fillstyle = kwargs.get('fillstyle', 'full')
        x = self.year
        y = self.size
        plt.plot(x, y, 'o', color='k', fillstyle=fillstyle)
        if label is None:
            #label = f'{self.name} ({self.observer})'
            label = f'{self.observer} ({self.year})'
        if ha == 'right':
            dx = -dx
        if va == 'top':
            y *= 0.75
        if va == 'bottom':
            y *= 1.25
        plt.text(x + dx, y, label, ha=ha, va=va, size='small')


HIPPARCHUS = Catalog('Rhodes', 'Hipparchus', -127, 1025)
PTOLEMY = Catalog('Almagest Star Catalog', 'Ptolemy', 150, 1028)
HEVELIUS = Catalog('Catalogus Stellarum Fixarum', 'Hevelius', 1687, 1564)
FLAMSTEED = Catalog('British Catalogue', 'Flamsteed', 1712, 2866)
LACAILLE = Catalog('Coelum Australe Stelliferum', 'Lacaille', 1742, 9766)
PIAZZI = Catalog('Praecipuarium Stellarum Inerrantium', 'Piazzi', 1803, 6748)
BONNER = Catalog('Bonner Durchmusterung', 'Argelander', 1886, 457848)
CAPE = Catalog('Cape Photo Durchmusterung', 'Gill and Kapteyn', 1896, 454875)
CORDOBA = Catalog('Cordoba Durchmusterung', 'Thome', 1932, 578802)
HGSC = Catalog('Hubble GSC 1.1', 'Hubble GSC', 1992, 19000000)
TYCHO = Catalog('Tycho', 'Hipparcos', 1997, 1058332)
TWOMASS = Catalog('2MASS All Sky Catalog of Point Sources', '2MASS', 2003, 470992970)
GAIADR3 = Catalog('Gaia Data Release 3', 'Gaia', 2022, 1800000000)
# See https://www.lsst.org/scientists/keynumbers
# (I summed the number of galaxies and that of individual stars)
LSST = Catalog('LSST DR11', 'LSST', 2034, 37000000000)

# This is from https://heasarc.gsfc.nasa.gov/docs/heasarc/headates/how_many_xray.html
TX = np.array([1962, 1965, 1970, 1974, 1980, 1984, 1990, 2000, 2010, 2018])
NX = np.array([1, 10, 60, 160, 680, 840, 8000, 340000, 780000, 1500000])

TGAMMA = np.array([1970, 1973, 1977, 1981, 1994, 1995, 1996, 1999, 2002, 2010, 2011, 2015, 2020])
NGAMMA = np.array([1, 6, 13, 25, 50, 128, 156, 270, 420, 1451, 1873, 3034, 5064])


standard_figure('Astronomical catalogues')
HEVELIUS.plot(va='top')
FLAMSTEED.plot()
LACAILLE.plot()
BONNER.plot(ha='right', va='top')
CORDOBA.plot(ha='center', va='bottom')
HGSC.plot(ha='right')
TWOMASS.plot(ha='right')
GAIADR3.plot(ha='right')
LSST.plot(ha='right', fillstyle='none')
setup_ca(xlabel='Year', ylabel='Number of objects', xmin=1650, xmax=2050, logy=True)

fmt = dict(color='k', fillstyle='none')
plt.plot(TX, NX, 's', **fmt)
plt.plot(TGAMMA, NGAMMA, '^', **fmt)

handles = []
handles.append(plt.plot([], [], 's', label='Known X-ray sources', **fmt)[0])
handles.append(plt.plot([], [], '^', label='Known $\\gamma$-ray sources', **fmt)[0])
plt.legend(handles=handles, fontsize='small', loc='lower center')


save_cf()


if __name__ == '__main__':
    plt.show()
