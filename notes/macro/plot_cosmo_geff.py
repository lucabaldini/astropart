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


import astropy.units as u
import numpy as np

from astropart.cosmo import geff_energy, geff_entropy, cosmic_time_to_temperature
from astropart.plot import plt, standard_figure, margin_figure, setup_ca, save_cf



standard_figure('cosmo_geff', top=0.85)
kT = np.geomspace(1.01e-2, 1e6, 200) * u.MeV
plt.plot(kT, geff_energy(kT), color='black', label='geff')
plt.plot(kT, geff_entropy(kT), color='black', ls='dashed', label='geffs')

# Annotations.
pad = 1.075
kwargs = dict(ha='center', fontsize=8)
x = 2.e5 * u.MeV
plt.text(x.value, pad * geff_energy(x), f'{geff_energy(x):.2f}', va='bottom', **kwargs)
x = 6.e3 * u.MeV
#plt.text(x.value, pad * geff_energy(x), f'{geff_energy(x):.2f}', va='bottom', **kwargs)
plt.text(50, 20, '17.25', va='bottom', **kwargs)
x = 10 * u.MeV
plt.text(x.value, pad * geff_energy(x), f'{geff_energy(x):.2f}', va='bottom', **kwargs)
x = 0.025 * u.MeV
plt.text(x.value, 1. / pad * geff_energy(x), f'{geff_energy(x):.2f}', va='top', **kwargs)
plt.text(x.value, pad * geff_entropy(x), f'{geff_entropy(x):.2f}', va='bottom', **kwargs)

plt.gca().annotate('EW trans.', xytext=(1e5, 30), xy=(1e5, 100),
    arrowprops=dict(arrowstyle="->"), **kwargs)
plt.gca().annotate('QCD trans.', xytext=(150, 5), xy=(150, 20),
    arrowprops=dict(arrowstyle="->"), **kwargs)

mpi = 139
plt.gca().annotate('$m_\\pi$', xytext=(mpi, 120), xy=(mpi, 60),
    arrowprops=dict(arrowstyle="->"), **kwargs)
mmu = 105
plt.gca().annotate('$m_\\mu$', xytext=(mmu, 50), xy=(mmu, 25),
    arrowprops=dict(arrowstyle="->"), **kwargs)
me = 0.511
plt.gca().annotate('$m_e$', xytext=(me, 25), xy=(me, 12.5),
    arrowprops=dict(arrowstyle="->"), **kwargs)

plt.gca().xaxis.set_inverted(True)
setup_ca(xlabel='$k_B T$ [MeV]', ylabel='$g_*(T)$', logx=True, logy=True, ymin=1, ymax=200)
top_axis = plt.gca().secondary_xaxis('top')
top_axis.set_xlabel('Age of the universe')
ticks = [1 * u.ps, 1 * u.ns, 1 * u.us, 1. * u.ms, 1 * u.s, 1 * u.min, 1 * u.hour]
labels = [f'{tick.value:.0f} {tick.unit:latex}' for tick in ticks]
ticks = [cosmic_time_to_temperature(tick).to(u.MeV).value for tick in ticks]
top_axis.set_xticks(ticks, labels)
top_axis.set_xticks([], minor=True)
save_cf()


plt.show()
