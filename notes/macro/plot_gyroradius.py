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

import astropy.constants as const
import astropy.units as u
import numpy as np

from astropart.plot import plt, margin_figure, setup_ca, save_cf, add_yaxis, subplot_vertical_stack

u.eV_c = u.def_unit('eV / c', u.eV / const.c)



# import astropy.units as u
# from astropy.constants import m_e, e  # Electron mass and charge

# def gyroradius(m, q, v_perp, B):
#     """
#     Calculate the gyroradius (Larmor radius) of a charged particle.

#     Parameters:
#     m : astropy.Quantity
#         Mass of the particle (with units)
#     q : astropy.Quantity
#         Charge of the particle (with units)
#     v_perp : astropy.Quantity
#         Perpendicular velocity (with units)
#     B : astropy.Quantity
#         Magnetic field strength (with units)

#     Returns:
#     astropy.Quantity
#         Gyroradius (with units)
#     """
#     return (m * v_perp) / (abs(q.si) * B)

# # Example: Electron in a 1 Tesla field, moving at 1e6 m/s perpendicular velocity
# B_field = 1 * u.T  # 1 Tesla
# v_perp = 1e6 * u.m / u.s  # 1 million m/s

# r_g = gyroradius(m_e, e, v_perp, B_field)
# print(f"Gyroradius: {r_g.to(u.mm)}")


#margin_figure('gyroradius', height=2.25)

p = 1.e9 * u.eV_c
B = 1. * u.T
print(p, B)

rho = p.to(u.g * u.m / u.s) / const.e.si / B
print(rho.to(u.m))

#r = np.linspace(1.e9, 1.e21) * u.GV
#B = 1. * u.G
#rho = p * const.c / const.e.to(u.C) / B.to(u.T)
#plt.plot(p, rho.to(u.cm), color='k')
#setup_ca(xlog=True, ylog=True, xlabel='Momentum [eV/c]', ylabel='Gyroradius [cm]')
#plt.show()