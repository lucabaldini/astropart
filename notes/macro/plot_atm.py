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

"""Plot some relevant quantities specific to the Earth atmosphere.
"""

import numpy as np

from astropart.atm import Atmosphere
from astropart.plot import plt, margin_figure, setup_ca, save_cf, add_yaxis, subplot_vertical_stack


hmax = 50.
h = np.linspace(0., hmax, 100)
hticks = np.linspace(0., hmax, 6)
hlabel = 'Altitude [km]'

density = Atmosphere.density(h)
grammage = Atmosphere.grammage(h)

margin_figure('atmospheric density', height=2.25)
plt.plot(h, density, color='k')
setup_ca(xlabel=hlabel, ylabel='Density [g cm$^{-3}$]', logy=True, xticks=hticks)
save_cf()

margin_figure('atmospheric grammage', height=2.25)
plt.plot(h, grammage, color='k')
setup_ca(xlabel=hlabel, ylabel='Grammage [g cm$^{-2}$]', logy=True, xticks=hticks)
save_cf()

margin_figure('atmospheric scales', height=2.75)
hmax = 25.
ax_h = add_yaxis(0.25, 0., hmax, logy=False, title='h[km]', top=0.9)
ax_radl = add_yaxis(0.55, 0., hmax, logy=False, title='$X_0$', top=0.9)
radl = np.array([1., 2., 5., 10., 20.])
ax_radl.set_yticks(Atmosphere.radiation_length_to_altitude(radl), radl)
ax_intl = add_yaxis(0.85, 0., hmax, logy=False, title='$\lambda_p$', top=0.9)
intl = np.array([1., 2., 5., 10.])
ax_intl.set_yticks(Atmosphere.proton_interaction_length_to_altitude(intl), intl)
save_cf()

margin_figure('atmospheric cherenkov', height=3.25)
m_e = 0.511
ax1, ax2 = subplot_vertical_stack((0.5, 0.5))
plt.sca(ax1)
plt.plot(h, Atmosphere.cherenkov_energy_threshold(h, m_e), color='k')
setup_ca(ylabel='Energy threshold [MeV]', logy=True, grids=True, ymin=10., ymax=1000.)
plt.sca(ax2)
plt.plot(h, Atmosphere.cherenkov_angle(h), color='k')
setup_ca(xlabel=hlabel, ylabel='Cherenkov angle [$^\circ$]', logy=True, grids=True,
    ymin=0.01, xticks=hticks)

save_cf()
plt.show()
