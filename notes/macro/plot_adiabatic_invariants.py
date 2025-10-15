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
    draw_line, setup_ca_drawing, draw_ellipse, draw_arrow


margin_figure('adiabatic_invariant', height=1.3, left=0.0, bottom=0.0, right=1., top=1.)
draw_ellipse((0., 0.), 4., 2.)
draw_arrow((-2.5, 0.), (2.5, 0.))
plt.text(2.25, -0.15, 'q', ha='center', va='top')
draw_arrow((0., -1.5), (0., 1.5))
plt.text(-0.15, 1.25, 'p', va='center', ha='right')
setup_ca(xmin=-2.5, xmax=2.5, ymin=-1.5, ymax=1.5, grids=False)
plt.text(-0.1, 0.5, '$\\sqrt{2mE}$', ha='right', va='center')
plt.text(0.85, -0.1, '$\\sqrt{2E/m\\omega^2}$', ha='center', va='top')
plt.axis('off')
save_cf()

plt.show()
