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
import xcom

from astropart.plot import plt, standard_figure, setup_ca, save_cf


standard_figure('photon_xsec')
data = xcom.calculate_cross_section(14)
energy = 1.e-6 * data['energy']
plt.plot(energy, data['total'], color='k')
plt.plot(energy, data['photoelectric'], color='k', ls='dashed')
plt.plot(energy, data['incoherent'], color='k', ls='dashed')
plt.plot(energy, data['pair_atom'] + data['pair_electron'], color='k', ls='dashed')
setup_ca(xlabel='Energy [MeV]', ylabel='$\\sigma$ [barn/atom]', logx=True, logy=True,
    xticks=np.geomspace(1.e-3, 1.e5, 9))
plt.text(1.e-2, 2.e3, '$\\sigma_{\\mathrm{photoelectric}}$')
plt.text(2.5e3, 4.e-7, '$\\sigma_{\\mathrm{photoelectric}}$')
plt.text(2.e-3, 6.e-1, '$\\sigma_{\\mathrm{Compton}}$')
plt.text(1.e3, 2.e-2, '$\\sigma_{\\mathrm{Compton}}$')
plt.text(1.5, 2.e-7, '$\\sigma_{\\mathrm{pair}}$')
plt.text(1.e4, 4., '$\\sigma_{\\mathrm{pair}}$')
save_cf()
plt.show()
