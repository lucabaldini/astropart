# Copyright (C) 2026, Luca Baldini (luca.baldini@pi.infn.it)
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

"""Plot the particle gyroradius as a function of rigidity for different magnetic field strengths.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np

from astropart.plot import plt, standard_figure, setup_ca, save_cf



R = 1.e9 * u.V
B = 1. * u.T
rho = R / B / const.c
print(R, B, rho.to(u.m))

R = 1.e15 * u.V
B = 1. * u.G
rho = R / B / const.c
print(R, B, rho.to(u.m))


standard_figure("Gyroradius")
R = np.linspace(1.e9, 1.e21) * u.V
x = 9.e12
for B in [1. * u.uG, 1. * u.G, 1. * u.T, 1. * u.MG, 1. * u.TG]:
    rho = R / B / const.c
    plt.plot(R.to(u.V), rho.to(u.m), label=f'B={B}', color='black')
    y = 2. * (x * u.V / B / const.c).to(u.m).value
    plt.text(x, y, f'{B}', color='black', fontsize=8, rotation=24., va="top", ha="center")
setup_ca(logx=True, logy=True, xlabel='Rigidity [V]', ylabel='Gyroradius [m]',
         ymin=1.e-5, ymax=1.e18)

y = 1. * u.au.to(u.m)
plt.axhline(y, color='black', ls='--')
plt.text(1.e17, 1.5 * y, "1 au")
y = 1. * u.pc.to(u.m)
plt.axhline(y, color='black', ls='--')
plt.text(1.e17, 1.5 * y, "1 pc")

save_cf()
plt.show()