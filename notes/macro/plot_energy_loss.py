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

from astropart.pdg import MuonEnergyLossColumn, parse_muon_energy_loss, MUON_MASS,\
    critical_energy_gas, critical_energy_solid, multiple_scattering_angle_plane
from astropart.plot import plt, margin_figure, setup_ca, save_cf



data = parse_muon_energy_loss()
momentum = data[MuonEnergyLossColumn.MOMENTUM.value]
ion_loss = data[MuonEnergyLossColumn.IONIZATION_LOSS.value]
rad_loss = data[MuonEnergyLossColumn.TOTAL_RADIATION_LOSS.value]
tot_loss = data[MuonEnergyLossColumn.TOTAL_LOSS.value]
mip_momentum = data['mip_momentum']
mip_betagamma = data['mip_betagamma']


def top_xaxis():
    """Add the top x-axis with beta * gamma.
    """
    xmin, xmax = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xscale('log')
    ax2.set_xlabel('$\\beta\\gamma$')
    betagamma = np.array([1.e-2, 1., 1.e2, 1.e4, 1.e6])
    pc = MUON_MASS * betagamma
    labels = [f'$10^{{{int(x)}}}$' for x in np.log10(betagamma)]
    ax2.set_xticks(pc)
    ax2.set_xticklabels(labels)
    # For some reason that I don't fully understand, this has to be put here.
    ax2.set_xlim(xmin, xmax)


margin_figure('muon_ionization_loss', height=2.75, top=0.825, left=0.275)
plt.plot(momentum, ion_loss, color='k')
plt.axvline(mip_momentum, color='k', ls='dashed')
setup_ca(xlabel='Momentum [MeV/c]', ylabel='Ionization loss [MeV cm$^{2}$ g$^{-1}$]',
    xmin=10., xmax=1.e9, ymin=1., ymax=100., logx=True, logy=True, xticks=[1.e2, 1.e4, 1.e6, 1.e8])
top_xaxis()
fmt = dict(ha='center', va='center', size='small')
plt.text(60., 50., '$\\frac{1}{E}$', **fmt)
plt.text(1.e6, 50., '$\\log{E}$', **fmt)
save_cf()

margin_figure('muon_stopping_power', height=2.75, top=0.825, left=0.275)
plt.plot(momentum, ion_loss, color='k', ls='dashed')
plt.plot(momentum, rad_loss, color='k', ls='dashed')
plt.plot(momentum, tot_loss, color='k')
setup_ca(xlabel='Momentum [MeV/c]', ylabel='Stopping power [MeV cm$^{2}$ g$^{-1}$]',
    xmin=10., xmax=1.e9, ymin=1., ymax=100., logx=True, logy=True, xticks=[1.e2, 1.e4, 1.e6, 1.e8])
top_xaxis()
save_cf()

margin_figure('critical_energy', height=2.5, left=0.275)
Z = np.arange(1, 100)
plt.plot(Z, critical_energy_solid(Z), color='k', label='Solids')
plt.plot(Z, critical_energy_gas(Z), color='k', ls='dashed', label='Gases')
setup_ca(xlabel='Atomic number', ylabel='Critical energy [MeV]', logx=True, logy=True,
    legend=True)
save_cf()

margin_figure('multiple_scattering', height=2.5, left=0.275)
p = np.linspace(100., 1.e6, 100)
m_e = 0.511
m_p = 938.3
for i, t in enumerate((0.1, 0.01)):
    label = 'Electrons' if i == 0 else None
    plt.plot(p, np.degrees(multiple_scattering_angle_plane(m_e, p, t)), color='k', label=label)
    label = 'Protons' if i == 0 else None
    plt.plot(p, np.degrees(multiple_scattering_angle_plane(m_p, p, t)), color='k',
        ls='dashed', label=label)
setup_ca(xlabel='Momentum [MeV/c]', ylabel='Multiple scattering angle [$^\\circ$]',
    logx=True, logy=True, legend=True, xticks=np.geomspace(100., 1.e6, 5),
    ymin=1.e-4, ymax=100., yticks=np.geomspace(1.e-4, 100., 7))
kwargs = dict(size='small')
plt.text(1.e4, 0.05, '10% $X_0$', **kwargs)
plt.text(5.e2, 0.005, '1% $X_0$', **kwargs)
save_cf()



plt.show()
