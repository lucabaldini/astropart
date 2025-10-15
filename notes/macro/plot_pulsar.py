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


import astropy.units as u
import numpy as np

from astropart.ns import CanonicalPulsar, PulsarCatalog
from astropart.plot import plt, standard_figure, margin_figure, setup_ca, save_cf


psrcat = PulsarCatalog()
P = np.linspace(1.e-3, 10.) * u.s


standard_figure('Pulsar period pdot')
CanonicalPulsar.ppdot_figure()
psrcat.plot_ppdot(max_distance=2., color='black', ms=2)
row = psrcat.find_row('crab')
plt.text(psrcat.period[row], 2. * psrcat.pdot[row], 'Crab', ha='right', va='bottom')
save_cf()

margin_figure('Pulsar minimum density', height=2.5, left=0.3)
rho = CanonicalPulsar.minimum_density(P)
plt.plot(P, rho, color='black')
setup_ca(xlabel='P [s]', ylabel='Minimum density [g cm$^{-3}$]', logx=True, logy=True,
         xticks=np.geomspace(1.e-3, 10., 5), yticks=np.geomspace(1.e6, 1.e14, 9),
         ymin=1.e6)
save_cf()


margin_figure('Pulsar rotational energy', height=2.5, left=0.3)
E = CanonicalPulsar.rotational_energy(P)
plt.plot(P, E, color='black')
setup_ca(xlabel='P [s]', ylabel='Rotational energy [erg]', logx=True, logy=True,
        xticks=np.geomspace(1.e-3, 10., 5), yticks=np.geomspace(1.e44, 1.e52, 9),
        ymin=1.e44)
save_cf()


if __name__ == '__main__':
    plt.show()
