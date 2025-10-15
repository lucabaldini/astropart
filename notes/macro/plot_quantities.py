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

"""Orders of magnitude.
"""

import astropy.units as u
from matplotlib.ticker import LogLocator, MultipleLocator, AutoMinorLocator, NullFormatter
import numpy as np

from constants import *
from plotting import plt, setup_ca, save_cf, add_yaxis


def plot_quantity(quantity, unit, label : str = None, yscale : float = 1.):
    """Plot a quantity
    """
    y = quantity.to(unit).value
    if label is None:
        label = quantity.name
    plt.plot((0.15, 0.4), (y, y), color='black', lw=0.8)
    plt.plot((0.4, 0.6), (y, y * yscale), color='black', lw=0.8)
    plt.text(0.7, y * yscale, label, size='small', va='center')

def plot_range(range_, unit, label=None):
    """Plot a range
    """
    ymin = range_.minimum.to(unit).value
    ymax = range_.maximum.to(unit).value
    y = np.sqrt(ymin * ymax)
    if label is None:
        label = range_.name
    plt.plot((0.15, 0.4, 0.4, 0.15), (ymin, ymin, ymax, ymax), color='black', lw=0.8)
    plt.plot((0.4, 0.6), (y, y), color='black', lw=0.8)
    plt.text(0.7, y, label, size='small', va='center')

def data_to_figure(ax, x, y):
    """Convert axes coordinates to figure coordinates.

    This is somewhat more difficult than one might think, and the primary source
    of information is the matplotlib documentation at
    https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html

    Basically the transformation involves passing from data coordinates to
    axes coordinates, and then to figure coordinates.
    """
    return plt.gcf().transFigure.inverted().transform(ax.transData.transform((x, y)))


plt.figure('Astronomical dimensions', figsize=(2., 5.))
ymin = 1.e-16
ymax = 1.e28
ax_m = add_yaxis(0.255, ymin, ymax, logy=True, title='m')
ax_m.yaxis.set_major_locator(LogLocator(numticks=14))
unit = u.m
plot_quantity(PROTON_RADIUS, unit, 'Proton')
plot_quantity(BOHR_RADIUS, unit, 'Hydrogen atom')
plot_quantity(NEUTRON_STAR_RADIUS, unit, 'Neutron star')
plot_quantity(EARTH_RADIUS, unit, 'Earth', yscale=0.5)
plot_range(WHITE_DWARF_RADIUS, unit, 'White dwarf')
plot_quantity(JUPITER_RADIUS, unit, 'Jupiter')
plot_quantity(SUN_RADIUS, unit, 'Sun', yscale=1.5)
plot_range(RED_GIANT_RADIUS, unit, 'Red giant')
plot_quantity(NEPTUNE_APHELION, unit, 'Solar system')
ymin = 2.e17
_, bot = data_to_figure(ax_m, 0., ymin)
ymin = u.m.to(u.pc, ymin)
ymax = u.m.to(u.pc, ymax)
ax_pc = add_yaxis(0.515, ymin, ymax, logy=True, title='pc', bottom=bot)
ax_pc.yaxis.set_major_locator(LogLocator(numticks=10))
unit = u.pc
plot_quantity(MILKY_WAY_RADIUS, unit, 'Milky Way')
plot_quantity(VIRGO_RADIUS, unit, 'Virgo cluster')
plot_quantity(UNIVERSE_RADIUS, unit, 'Universe')
save_cf()


plt.figure('Astronomical distances', figsize=(2.75, 10.))
ymin = 1.e8
ymax = 1.e28
ax_m = add_yaxis(0.205, ymin, ymax, logy=True, title='m')
ax_m.yaxis.set_major_locator(LogLocator(numticks=14))
unit = u.m
plot_quantity(DIST_TO_MOON, unit, 'Moon')
plot_quantity(DIST_TO_SUN, unit, 'Sun')
plot_quantity(DIST_TO_PROXIMA_CEN, unit, 'Proxima Centauri')
plot_quantity(DIST_TO_SIRUS_B, unit, 'Sirius B')
ymin = 2.e17
_, bot = data_to_figure(ax_m, 0., ymin)
ymin = u.m.to(u.pc, ymin)
ymax = u.m.to(u.pc, ymax)
ax_pc = add_yaxis(0.475, ymin, ymax, logy=True, title='pc', bottom=bot)
ax_pc.yaxis.set_major_locator(LogLocator(numticks=12))
unit = u.pc
plot_quantity(DIST_TO_CRAB, unit, 'Crab Nebula')
plot_quantity(DIST_TO_GALACTIC_CENTER, unit, 'Galactic center')
plot_quantity(DIST_TO_LMC, unit, 'LMC')
plot_quantity(DIST_TO_VIRGO, unit, 'Virgo cluster')
plot_quantity(DIST_TO_BL_LAC, unit, 'BL Lacertae')
plot_quantity(DIST_TO_HD1, unit, 'HD1')
save_cf()


if __name__ == '__main__':
    plt.show()
