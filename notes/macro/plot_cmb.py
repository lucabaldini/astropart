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

import numpy as np
import astropy.units as u

from astropart.qstat import CMB
from plotting import plt, setup_ca, save_cf, standard_figure, margin_figure


cmb = CMB()

margin_figure('cmb', height=2.5, left=0.37)
#E = np.linspace(0.0000001, 0.0022, 200) * u.eV
E = np.linspace(0.0001, 2.2, 200) * u.meV
plt.plot(E, cmb.spectral_number_density(E).to(u.eV**-1 * u.cm**-3), color='black')
average_energy = cmb.average_energy().to(u.meV).value
plt.axvline(average_energy, color='black', ls='dashed')
# Peak energy
x = 0.374
y = cmb.spectral_number_density(x * u.meV).value
plt.plot(x, y, 'o', color='black')
plt.text(x, y, ' $E_{p} = 0.374$ eV')
# Average energy.
x, y = average_energy + 0.05, 0.25e5
plt.text(x, y, f'$\\left<E\\right> = {average_energy:.3f}$ meV', rotation=90)
setup_ca(xlabel='Energy [meV]', ylabel='Spectral number density [cm$^{-3}$ eV$^{-1}$]',
         xmin=0., ymin=0., ymax=5.e5, xticks=np.linspace(0., 2., 5))
plt.gca().set_yticklabels([0, '$1\\times 10^5$', '$2\\times 10^5$', '$3\\times 10^5$',
                           '$4\\times 10^5$', '$5\\times 10^5$'])

save_cf()

plt.show()
