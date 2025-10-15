# Copyright (C) 2024-2025, Luca Baldini (luca.baldini@pi.infn.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Test suite for astropart.ebl
"""

import pytest

#from astropy import constants as const
from astropy import units as u
import numpy as np

from astropart import ASTROPART_DATA
import astropart.ebl as ebl
from astropart.plot import plt, setup_ca
from astropart.qstat import CMB


def test_cmb():
    """Small test to verify that the CMB component in the spectrum at
    https://indico.in2p3.fr/event/10225/contributions/1441/attachments/994/1289/GIF20140909_CIB_HDole.pdf
    plays well with our CMB library.

    Note that the EBL data points are taken by hand, and we cannot expect a level
    of agreement to better than ~10%.
    """
    cmb = CMB()
    # Load the CMB test data.
    file_path = ASTROPART_DATA / 'ebl_universe_sed_cmb.csv'
    nu, spectral_density = ebl._load_spectral_density(file_path)
    energy, energy_density = ebl._energy_density_data((nu, spectral_density))
    number_density = energy_density / energy
    #delta = (spectral_density - cmb.spectral_energy_density(nu * u.GHz)) / spectral_density
    #assert max(delta) < 0.5
    #delta = (energy_density - cmb.differential_energy_density(energy)) / energy_density
    #assert max(delta) < 0.5
    #delta = (number_density - cmb.differential_number_density(energy)) / number_density
    #assert max(delta) < 0.5

def test_plot():
    """Some relevant plots.
    """
    nu = np.logspace(6., 25., 250)
    energy = np.logspace(-8., 10., 250)
    plt.figure('ebl_spectral_energy_density_nu')
    plt.plot(nu, ebl.spectral_energy_density_nu(nu))
    setup_ca(logx=True, logy=True, xlabel='$\\nu$ [Hz]',
        ylabel='Spectral energy density [erg Hz$^{-1}$ cm$^{-3}$]')
    plt.figure('ebl_energy_density')
    plt.plot(energy, ebl.differential_energy_density(energy))
    setup_ca(logx=True, logy=True, xlabel='Energy [eV]',
        ylabel='Differential energy density [cm$^{-3}$]')
    plt.figure('ebl_number_density')
    plt.plot(energy, ebl.differential_number_density(energy))
    setup_ca(logx=True, logy=True, xlabel='Energy [eV]',
        ylabel='Differential number density [eV^$^{-1}$ cm$^{-3}$]')



if __name__ == '__main__':
    test_plot()
    test_cmb()
    plt.show()
