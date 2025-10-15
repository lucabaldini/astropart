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


import numpy as np

from astropart import logger
from astropart.plot import plt, setup_ca, full_figure, margin_figure, standard_figure,\
    save_cf, draw_line


margin_figure('irf_fov', height=2.5)
theta_max = np.linspace(0., 90., 100)
fov = np.pi * (1. - np.cos(np.radians(theta_max)))
plt.plot(theta_max, fov / 4 / np.pi, color='k')
setup_ca(xlabel='$\\theta_\\mathrm{max}$', ylabel='Field of view / $4\\pi$', logy=True,
    xticks=np.linspace(0., 80., 5))
save_cf()


plt.show()
