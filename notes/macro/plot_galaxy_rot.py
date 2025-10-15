# Copyright (C) 2025, Luca Baldini (luca.baldini@pi.infn.it)
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

from astropart.plot import plt, margin_figure, setup_ca, save_cf

margin_figure('galaxy rotation', height=2.25, left=0.25)
r = np.linspace(0., 5., 100)
v = r * (r < 1.) + 1. / np.sqrt(r) * (r >= 1.)
plt.plot(r, v, color='black')
plt.axvline(1., color='black', ls='dashed')
setup_ca(xmin=0., ymin=0., xlabel='$r~[R_b]$', ylabel='$v(r)~[a.~u.]$')
save_cf()


plt.show()
