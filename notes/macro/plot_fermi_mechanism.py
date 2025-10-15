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

"""Plot some relevant quantities specific to Fermi acceleration mechanism.
"""

import numpy as np

from astropart.plot import plt, margin_figure, setup_ca, add_yaxis, save_cf


margin_figure('fermi energy gain', height=2.5)
beta = 1.e-4
c = np.linspace(-1., 1., 7500)
dE = 2. * beta**2. + 2. * beta * c
ave = 3. * beta**2.
plt.plot(c, abs(dE), color='black')
plt.axhline(ave, ls='dashed', color='black')
setup_ca(xlabel='$\\cos\\theta$', ylabel='$\\left| \\frac{\\Delta E}{E} \\right|$',
    grids=True, logy=True, ymin=1.e-8)
plt.text(0.1, ave * 1.2, 'Average')
save_cf()


if __name__ == '__main__':
    plt.show()
