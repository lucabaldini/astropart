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

"""Plot some relevant quantities specific to the diffusive shock acceleration.
"""


import astropy.units as u
import numpy as np


from astropart.plot import plt, standard_figure, setup_ca, setup_ca_drawing,\
    save_cf, draw_line, draw_arrow, draw_rectangle, _srcp, draw_ellipse, draw_circle


def magnetic_field(max_energy, size, beta=1., Z=1):
    """Return the magnetic field necessary to accelerate particles to a given
    maximum energy, for a given set of parameters.

    Move to library.

    This is essentially B = E / (2 beta c Z e R)
    """
    return max_energy / (beta * 3.e10 * Z * 4.8e-10 * size * 3.09e18)


standard_figure('hillas_plot')

E = np.logspace(-14., 11., 100)
emax = 1.e20
y1 = magnetic_field(emax, E)
y2 = magnetic_field(emax, E, beta=0.01)
plt.plot(E, y1, color='black')
plt.plot(E, y2, color='black', ls='dashed')
plt.fill_between(E, y1, y2, color='gray', alpha=0.5)

plt.text(1.e2, 2.e-1, '$10^{20}$ eV proton', size='small', rotation=-42.,
    ha='center', va='center')
x = 1.e-7
y = magnetic_field(emax, x)
plt.annotate('$\\beta = 1$', (x, y), (1000 * x, 100 * y),
    arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleB=90, angleA=-10'))
x = 5.e-8
y = magnetic_field(emax, x, beta=0.01)
plt.annotate('$\\beta = 0.01$', (x, y), (1000 * x, 100 * y),
    arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleB=90, angleA=-10'))

for emax in np.logspace(14, 19, 6):
    y = magnetic_field(emax, E)
    plt.plot(E, y, color='gray', ls='dashed', lw=1.)


x = 5.e-8
y = magnetic_field(1.e14, x, beta=1.)
plt.annotate('$10^{14}$ eV', (x, y), (1e-6 * x, 1e-2 * y),
    arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleB=90, angleA=-10'))


x = (10. * u.km).to(u.pc).value
draw_arrow((x, 1.e14), (x, 1.e10))
plt.text(2. * x, 5.e13, 'Neutron stars')

x = 1.
y = 1.e-4
plt.text(x, y, 'SNR', ha='center', va='center')

x = 0.0001
y = 1.e4
plt.text(x, y, 'AGN', ha='center', va='center')


setup_ca(xlabel='$L$ [pc]', ylabel='$B$ [G]', xmin=1.e-14, xmax=1.e11,
    ymin=1.e-9, ymax=1.e15, logx=True, logy=True, grids=False)
save_cf()

plt.show()
