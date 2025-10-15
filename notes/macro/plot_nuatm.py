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

"""Plot some relevant quantities for the atmospheric neutrinos.
"""


import numpy as np

from astropart.plot import plt, margin_figure, setup_ca, save_cf, draw_circle


margin_figure('atmospheric nu sketch', height=2.2, left=0.05, bottom=0.05, right=0.95)
draw_circle((0., 0.), 0.9)
draw_circle((0., 0.), 1.0, ls='dashed')
plt.gca().set_aspect(1)
setup_ca(xmin=-1.05, xmax=1.05, ymin=-1.05, ymax=1.05, grids=False)
plt.axis('off')
save_cf()

margin_figure('atmospheric nu path', height=2.25)
R = 6300.
d = 2.5
h = 15.
theta = np.linspace(0., np.pi, 200)
A = (R + h)**2. - (R - d)**2
x1 = ((R - d) + np.sqrt((R + h)**2 + A * np.tan(theta)**2.)) / (1. + np.tan(theta)**2.)
x2 = ((R - d) - np.sqrt((R + h)**2 + A * np.tan(theta)**2.)) / (1. + np.tan(theta)**2.)
c = np.cos(theta)
x = x1 * (c < 0) + x2 * (c >= 0)
y2 = (R + h)**2 - (R - d - x)**2
p = np.sqrt(x**2. + y2)
plt.plot(c, p, color='black')
setup_ca(xlabel='$\\cos\\theta$', ylabel='$\\nu$ path length [km]', logy=True)
save_cf()


plt.show()
