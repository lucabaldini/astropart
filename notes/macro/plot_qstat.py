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

from plotting import plt, setup_ca, save_cf, margin_figure


margin_figure('Quantum statistics', height=2.5, left=0.125)
mu = 1
kT = 0.1
energy = np.linspace(0, 2, 100)
x = (energy - mu) / kT
be = 1. / (np.exp(x) - 1)
fd = 1. / (np.exp(x) + 1)
mask = x > 0
plt.plot(energy[mask], be[mask], color='black')
plt.plot(energy, fd, color='black')
plt.text(0.05, 1.05, 'Fermi-Dirac', size='small')
plt.text(1.15, 1.25, 'Bose-Einstein', size='small')
plt.axvline(mu, ls='dashed', color='black')
plt.axhline(1., ls='dashed', lw=0.5, color='lightgray')
plt.axhline(0.5, ls='dashed', lw=0.5, color='lightgray')
setup_ca(xlabel='Energy', ylabel='$h^3 dn/d^3p$', xticks=[1], yticks=[], ymin=0., ymax=1.5)
plt.gca().set_xticklabels(['$\\mu$'])
save_cf()


margin_figure('Fermi Dirac', height=2.5, left=0.125)
mu = 1
energy = np.linspace(0, 2, 250)
for kT in (0.1, 0.01):
    x = (energy - mu) / kT
    fd = 1. / (np.exp(x) + 1)
    plt.plot(energy, fd, color='black')
#plt.text(0.05, 1.05, 'Fermi-Dirac', size='small')
#plt.text(1.15, 1.25, 'Bose-Einstein', size='small')
plt.axvline(mu, ls='dashed', color='black')
plt.axhline(1., ls='dashed', lw=0.5, color='lightgray')
plt.axhline(0.5, ls='dashed', lw=0.5, color='lightgray')
setup_ca(xlabel='Energy', ylabel='$h^3 dn/d^3p$', xticks=[1], yticks=[], ymin=0., ymax=1.5)
plt.gca().set_xticklabels(['$\\mu$'])
save_cf()


margin_figure('Planck distribution nu', height=2., left=0.125)
nu = np.linspace(0., 15., 100)
plank = nu**3 / (np.exp(nu) - 1)
plt.plot(nu, plank, color='black')
setup_ca(xlabel='$h\\nu / kT$', ylabel='$u_\\nu(\\nu)$',
         xticks=[0, 2, 4, 6, 8, 10, 12, 14],yticks=[], ymin=0.)
save_cf()

margin_figure('Planck distribution lambda', height=2., left=0.125)
nu = np.linspace(0., 15., 100)
plank = nu**5 / (np.exp(nu) - 1)
plt.plot(nu, plank, color='black')
setup_ca(xlabel='$hc / \\lambda kT$', ylabel='$u_\\lambda(\\lambda)$',
         xticks=[0, 2, 4, 6, 8, 10, 12, 14],yticks=[], ymin=0.)
save_cf()


plt.show()
