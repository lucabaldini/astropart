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

"""Plot some relevant quantities specific to the diffusive shock acceleration.
"""

from dataclasses import dataclass

import astropy.constants as const
import astropy.units as u
import numpy as np

from astropart import logger
from astropart.qstat import CMB
import astropart.ebl as ebl
from astropart.plot import plt, standard_figure, setup_ca, draw_line, save_cf


hz_to_eV = lambda freq: const.h.to_value(u.eV * u.s) * freq


@dataclass
class Budget:

    """Small container class describing the energy budget in a given energy range.
    """

    name: str
    min_energy: float
    max_energy: float
    yref: float

    def integral_energy_density(self):
        return ebl.integral_energy_density(self.min_energy, self.max_energy)

    def integral_number_density(self):
        return ebl.integral_number_density(self.min_energy, self.max_energy)

    def mean_energy(self):
        return np.sqrt(self.min_energy * self.max_energy)

    @staticmethod
    def to_latex(value):
        exponent = int(np.log10(value)) - 1
        mantissa = value / 10.**exponent
        return '%.1f \\times 10^{%d}' % (mantissa, exponent)

    def plot(self):
        draw_line((self.min_energy, self.yref), (self.max_energy, self.yref))
        plt.text(self.mean_energy(), 1.5 * self.yref, self.name, size='small', va='center', ha='center')
        if self.min_energy < 10:
            x = self.max_energy * 1.75
            kwargs = dict(size='small', ha='left')
        else:
            x = self.min_energy * 0.5
            kwargs = dict(size='small', ha='right')
        val = self.to_latex(self.integral_energy_density())
        plt.text(x, self.yref * 1.3, f'$\\varepsilon = {val}$ eV cm$^{{-3}}$', **kwargs)
        val = self.to_latex(self.integral_number_density())
        plt.text(x, self.yref * 0.6, f'$n = {val}$ cm$^{{-3}}$', **kwargs)


standard_figure('ebl')
cmb = CMB()
energy = np.logspace(-7., 10.9, 250)
plt.plot(energy, energy * ebl.differential_energy_density(energy), color='black')
setup_ca(logx=True, logy=True, xlabel='Energy [eV]', ylabel='$E \\times u(E)$ [eV cm$^{-3}$]',
    ymin=2.e-8, ymax=0.5, grids=False)
plt.grid(which='major')
# Plot the CMB
energy = np.geomspace(1.e-6, 7.e-3)
plt.plot(energy, energy * cmb.spectral_energy_density(energy * u.eV), color='black', ls='dashed')
# Plot the other bands, with the energy and number budgets.
ir_budget = Budget('Infrared', hz_to_eV(3.e11), hz_to_eV(3.e13), 2.e-2)
optical_budget = Budget('Optical', hz_to_eV(3.e13), 10., 3.e-3)
xray_budget = Budget('X-ray', 500., 1e6, 2.5e-5)
gamma_budget = Budget('Gamma-ray', 4e6, 5.e10, 1.e-6)
for budget in (ir_budget, optical_budget, xray_budget, gamma_budget):
    budget.plot()
#plt.vlines(bounds, 0., bounds * ebl.differential_energy_density(bounds), color='gray', ls='dashed')
save_cf()

plt.show()
