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

import numpy as np

from astropart.plot import plt, margin_figure, setup_ca, save_cf,\
    draw_line, setup_ca_drawing, draw_circle, draw_arc


margin_figure('hard_sphere_xsec', height=1.3, left=0.0, bottom=0.0, right=1., top=1.)
setup_ca_drawing(xside=1., yside=0.75, equal_aspect=True)
r = 0.4
b = 0.25
theta = 2. * np.arcsin(b / r)
x0, y0 = r, -0.5 * r
x1, y1 = x0 - 2.5 * r * np.cos(theta / 2.), y0 + 2.5 * r * np.sin(theta / 2.)
x2, y2 = x0 - r * np.cos(theta / 2.), y0 + b
draw_circle((x0, y0), r)
draw_line((-1., y0), (1., y0), ls='dashed')
draw_line((x0, y0), (x1, y1), ls='dashed')
draw_line((-1., y2), (x2, y2))
draw_line((x2, y2), (x2 - 2.5 * r * np.cos(theta), y2 + 2.5 * r * np.sin(theta)))
draw_line((-0.6, y0), (-0.6, y0 + b), lw=0.75)
plt.text(-0.5, y0 + 0.5 * b, '$b$', ha='center', va='center')
draw_arc((x0, y0), 0.2, 180., -np.degrees(theta / 2.))
draw_arc((x0, y0), 0.25, 180., -np.degrees(theta / 2.))
kwargs = dict(size='small', ha='center', va='center')
plt.text(x0 - 0.15 * r, y0 - 0.45 * r, '$\\frac{\\theta}{2}$', **kwargs)
draw_arc((x2, y2), 0.2, 180., -np.degrees(theta))
plt.text(x2 + 0.1, y2 + 0.2, '$\\theta$', **kwargs)
save_cf()

plt.show()
