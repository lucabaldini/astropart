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

"""Plot the Herzprung-Russel diagram.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd

from plotting import DATA_FOLDER_PATH, plt, standard_figure, setup_ca, save_cf,\
    MARGIN_FIGURE_WIDTH, subplot_vertical_stack, margin_figure, THIN_LW, full_figure
import stars

CATALOG_FILE_PATH = DATA_FOLDER_PATH / 'hyg_v35.csv'

df = pd.read_csv(CATALOG_FILE_PATH)
print(df)
# Filter the data frame in order to avoid having too many entries.
mask = np.logical_or(pd.isna(df['hip']), df['mag'] < 5.)
# From the docs: a value of dist >= 100000 indicates missing or dubious
# (e.g., negative) parallax data in Hipparcos.
mask = np.logical_and(mask, df['dist'] < 100000)

df = df[mask]
print(df)
print(f'Number of sources after the selection: {mask.sum()}')

# Cache the necessary columns.
name = df['proper']
distance = df['dist']
color_index = df['ci']
absolute_magnitude = df['absmag']
apparent_magnitude = df['mag']


def draw_star(name, color_index, magnitude, offset=(0., 0.), padding=0.005, marker=False,
    mass=None):
    """Draw the label for a star at a given color index and absolute magnitude.

    Arguments
    ---------
    color_index : float
        The (B - V) color index.

    magnitude : float
        The absolute magnitude.

    offset : 2-element tuple
        The offset of the label wrt the star position in fractional axis coordinates

    padding : float
        The padding for the text in fractional axis coordinates.
    """
    x0, y0 = color_index, magnitude
    dx, dy = offset
    deltax = abs(np.diff(plt.xlim()))
    deltay = abs(np.diff(plt.ylim()))
    theta = np.degrees(np.arctan2(dy, dx))
    dx *= deltax
    dy *= deltay
    if marker:
        plt.scatter(x0, y0, color='k', s=0.5)
    if offset != (0., 0.):
        x1 = x0 + padding * deltax * np.cos(np.radians(theta))
        y1 = y0 - padding * deltay * np.sin(np.radians(theta))
        x2 = x0 + dx
        y2 = y0 - dy
        plt.plot((x1, x2), (y1, y2), color='k', lw=0.5)
    x = x0 + dx
    y = y0 - dy
    kwargs = dict(size='xx-small')
    if theta > -45. and theta <= 45.:
        kwargs['ha'] = 'left'
        kwargs['va'] = 'center'
        x += padding * deltax
    elif theta > 45. and theta <= 135:
        kwargs['ha'] = 'center'
        kwargs['va'] = 'bottom'
        y -= padding * deltay
    elif theta > 135. and theta <= 225:
        kwargs['ha'] = 'right'
        kwargs['va'] = 'center'
        x -= padding * deltax
    else:
        kwargs['ha'] = 'center'
        kwargs['va'] = 'top'
        y += padding * deltax
    plt.text(x, y, name, **kwargs)
    if mass is not None:
        if isinstance(mass, int):
            plt.text(x, y + 0.5, f'({mass} $M_\\odot$)', **kwargs)
        else:
            plt.text(x, y + 0.5, f'({mass:.2f} $M_\\odot$)', **kwargs)

def draw_sn(name):
    """Draw the evolution of SN on the HR diagram.

    Data are copied by hand from https://academic.oup.com/mnras/article/473/1/513/4107767#98486127.
    """
    file_path = DATA_FOLDER_PATH / f'{name}.txt'
    _, t, T, L = np.loadtxt(file_path, unpack=True)
    L = L * 1.e42 / 3.828e33
    mag = stars.luminosity_to_magnitude(L)
    ci = stars.temperature_to_bv(T)
    plt.plot(ci, mag, color='k', lw=0.5)
    for index in (0, -1):
        x, y = ci[index], mag[index]
        plt.scatter(x, y, color='k', s=0.5)
        plt.text(x, y - 0.2, f'{int(t[index])} d', size='xx-small', va='bottom', ha='center')
    #plt.text(ci[-1] + 0.02, mag[-1], name, va='top', ha='left', size='xx-small')
    plt.text(ci[0] - 0.02, mag[0], name, va='center', ha='right', size='xx-small')

def draw_radius_contour(R, label=True):
    """Draw the a contour at a fixed stellar radius (from the Stefan-Boltzmann law)
    on the HR diagram.
    """
    T = np.geomspace(1000., 50000., 200)
    ci = stars.temperature_to_bv(T)
    bc = stars.bolometric_correction(T)
    #ci = flower_formula_ci(T)
    #bc = flower_formula_bc(T)
    L = 4. * np.pi * (R * const.R_sun)**2. * const.sigma_sb / const.L_sun * T**4
    mag = stars.luminosity_to_magnitude(L.value) - bc
    plt.plot(ci, mag, color='darkgray', ls='dashed', lw=THIN_LW)
    if label:
        index = -10
        x = ci[index]
        y = mag[index]
        if R >= 10:
            R = int(R)
        if R == 1:
            R = ''
        plt.text(x, y, f'{R} $R_\\odot$', size='x-small', va='bottom', ha='right',
            bbox=dict(boxstyle='square,pad=0.01', fc='white', ec='none'))


full_figure('Herzprung Russel diagram', height=7.5, bottom=0.1, top=0.9, right=0.85)
cimin, cimax = -0.6, 2.25
magmin, magmax = 18, -20
setup_ca(xmin=cimin, xmax=cimax, xlabel='Color index (B - V)',
    ymin=magmin, ymax=magmax, ylabel='Absolute visual magnitude', grids=False)
# Plot all the stars.
plt.scatter(color_index, absolute_magnitude, color='k', s=0.5)
# The Sun.
sun_ci = 0.656
sun_mag = 4.85
draw_star('Sun', sun_ci, sun_mag, (0.1, 0.02))
# Some famous stars.
draw_star('Proxima Centauri', 1.807, 15.447, (-0.05, 0.015), marker=True, mass=0.1221)
draw_star('Regulus', -0.087, -0.569, (-0.075, 0.025), mass=3.8)
draw_star('Sirius A', 0., 1.43, (-0.05, -0.045), mass=2.063)
draw_star('Vega', 0., 0.582, (-0.095, -0.025), mass=2.11)
draw_star('Altair', 0.221, 2.21, (-0.05, -0.05), mass=1.86)
draw_star('Rigel', -0.03, -6.933, (0.05, 0.05), mass=21)
draw_star('Antares', 1.865, -5.089, (0.05, 0.))
draw_star('Betelgeuse', 1.5, -5.469, (0.05, 0.025))
draw_star('Aldebaran', 1.538, -0.682, (0.1, 0.))
draw_star('Tau Ceti', 0.72, 5.69, (0., -0.05), mass=0.783)
draw_star('Barnard\'s Star', 1.57, 13.23, (0.05, 0.1), mass=0.161)
draw_star('61 Cygni A', 1.139, 7.506, (0.05, 0.05), mass=0.7)
# White dwarfs,
draw_star('40 Eridani B', 0.03, 11.006, (0.0, 0.04))
draw_star('Sirius B', -0.03, 11.337, (-0.04, 0.02))
draw_star('Van Maanen\'s Star', 0.554, 14.222, (0.02, 0.05), marker=True)
# A few SNae.
for sn_name in ('SN2001fa', 'SN2004du', 'SN2003z', 'SN2006bp'):
    draw_sn(sn_name)
# And the grids with the radius from the SB law.
for R in np.logspace(-2, 3, 6):
    draw_radius_contour(R)
for R in (0.001, 10000, 100000):
    draw_radius_contour(R, False)

# Additional axes.
twiny = plt.gca().twiny()
twiny.set_xlabel('Effective temperature [K]')
twiny.set_xlim(cimin, cimax)
temperature_grid = np.array([2000, 3000, 4000, 5000, 7500, 10000, 50000])
ticks = stars.temperature_to_bv(temperature_grid)
labels = [f'{T}' for T in temperature_grid]
twiny.set_xticks(ticks, labels)
twinx = plt.gca().twinx()
twinx.set_ylabel('Bolometric luminosity @ $T_\\odot$ [$L_\\odot$]')
bc = stars.temperature_to_bv(stars.SUN_EFFECTIVE_TEMPERATURE)
twinx.set_ylim(stars.magnitude_to_luminosity(magmin + bc), stars.magnitude_to_luminosity(magmax + bc))
twinx.set_yscale('log')
twinx.set_zorder(-100)
twiny.set_zorder(-100)
save_cf('herzprung_russel_diagram')


if __name__ == '__main__':
    plt.show()
