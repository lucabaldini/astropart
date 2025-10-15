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

"""Plot some relevant quantities specific to Fermi gas.
"""

import numpy as np

from plotting import plt, margin_figure, setup_ca, add_yaxis, save_cf


#margin_figure('fermi_gas_pdist')
#plt.plot([0, 1, 1], [1, 1, 0], color='k')
#setup_ca(xmax=1.25, ymin=0., ymax=1.25, xticks=[], yticks=[], xlabel='$p$',
#    ylabel='$\\frac{dN}{dp}$')

margin_figure('fermi_gas_levels', height=2.5)
x1 = 0.255
x2 = 0.755
num_levels = 5
ymin = 0.
ymax = 0.9
yticks_E = np.linspace(ymin, ymax, num_levels)**2. / ymax
ylabels_E = [f'$E_{i}$' for i in range(num_levels)]
yticks_p = np.linspace(ymin, ymax, num_levels)
ylabels_p = [f'$p_{i}$' for i in range(num_levels)]
ax_E = add_yaxis(x1, 0., 1., title='E [a. u.]', top=0.9)
ax_E.yaxis.set_ticks(yticks_E)
ax_E.yaxis.set_ticklabels(ylabels_E)
ax_p = add_yaxis(x2, 0., 1., title='p [a. u.]', top=0.9)
ax_p.yaxis.set_ticks(yticks_p)
ax_p.yaxis.set_ticklabels(ylabels_p)
for y1, y2 in zip(yticks_E, yticks_p):
    plt.plot([-4.75, -1.25], [y1, y2], color='gray', ls='dashed', clip_on=False)
save_cf()


if __name__ == '__main__':
    plt.show()
