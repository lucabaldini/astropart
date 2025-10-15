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

"""Interface to the PDG data and formulae.
"""

from enum import Enum

import numpy as np
import scipy.special

from astropart import ASTROPART_DATA, logger



class MuonEnergyLossColumn(Enum):

    """Enum class provinding the intent of the columns in the PDG files
    with the muon energy loss, e.g.
    https://pdg.lbl.gov/2023/AtomicNuclearProperties/MUE/muE_silicon_Si.txt
    """

    KINETIC_ENERGY = 'kinetic_energy'
    MOMENTUM = 'momentum'
    IONIZATION_LOSS = 'ionization_loss'
    BREMSSTRAHLUNG_LOSS = 'bremsstralung_loss'
    PAIR_PRODUCTION_LOSS = 'pair_production_loss'
    PHOTONUCLEAR_LOSS = 'photonuclear_loss'
    TOTAL_RADIATION_LOSS = 'total_radiation_loss'
    TOTAL_LOSS = 'total_loss'
    CSDA_RANGE = 'csda_range'
    DELTA = 'delta'
    BETA = 'beta'
    dE_dx_R = 'boh'


MUON_ENERGY_LOSS_COLUMNS = [item.value for item in MuonEnergyLossColumn]
NUM_MUON_ENERGY_LOSS_COLUMNS = len(MUON_ENERGY_LOSS_COLUMNS)
MUON_MASS = 105.66 # MeV / c^2


def parse_muon_energy_loss(suffix: str = 'silicon_Si') -> dict:
    """Parse a PDG file containing the muon energy loss in a given material.

    I have to say the format in which the file is saved is definitely not the
    most handy to be parsed programmatically, as the header is not guarded with
    any comment sign, and there are two special lines in the middle of the data
    indicating the minimum ionization and critical energy. This is why this piece
    of code is so convoluted.
    """
    file_path = ASTROPART_DATA / f'muE_{suffix}.txt'
    logger.info(f'Loading muon energy loss data from {file_path}...')
    data = []
    with open(file_path, 'r') as input_file:
        for line in input_file:
            # Parse the line with minimum ionization...
            if 'Minimum ionization' in line:
                pmin = float(line.split()[1])
                bgmin = pmin / MUON_MASS
                logger.info(f'Minimum ionization @ p = {pmin} MeV / c, betagamma = {bgmin}')
            # Parse the line with the critical energy...
            elif 'Muon critical energy' in line:
                ecrit = float(line.split()[0])
                logger.info(f'Critical kinetic energy = {ecrit} MeV')
            # Extract the data from the rest.
            else:
                try:
                    row = [float(value) for value in line.split()]
                except ValueError:
                    continue
                # We want each row to contain the target number of columns,
                # and we want to discard the fist row (at 1 MeV), which the
                # documentation say is unreliable.
                if len(row) == NUM_MUON_ENERGY_LOSS_COLUMNS and row[0] != 1.:
                    data.append(row)
    data = np.array(data)
    columns = [data[:, col] for col in range(NUM_MUON_ENERGY_LOSS_COLUMNS)]
    data = {key: value for key, value in zip(MUON_ENERGY_LOSS_COLUMNS, columns)}
    data['mip_momentum'] = pmin
    data['mip_betagamma'] = bgmin
    data['critical_energy'] = ecrit
    return data


def critical_energy_solid(Z: int) -> float:
    """Return the approximate electron critical energy (in MeV) for a given atomic
    number of the target solid, according to the PDG parametrization (see figure
    33.14 in the 2016 edition).

    Arguments
    ---------
    Z : array_like
        The atomic number of the target material.
    """
    return 610. / (Z + 1.24)

def critical_energy_gas(Z: int) -> float:
    """Return the approximate electron critical energy (in MeV) for a given atomic
    number of the target gas, according to the PDG parametrization (see figure
    33.14 in the 2016 edition).

    Arguments
    ---------
    Z : array_like
        The atomic number of the target material.
    """
    return 710. / (Z + 0.92)

def multiple_scattering_angle_plane(mass: float, momentum: float, thickness: float,
    z: int = 1) -> float:
    """Return the multiple scattering angle in the plane in the gaussian approximation
    (see PDG, eq. 33.15 in the 2016 edition).
    """
    beta = np.sqrt(1. / (1. + mass / momentum))
    return 13.6 / beta / momentum * z * np.sqrt(thickness) * (1. + 0.038 * np.log(thickness))

def shower_longitudinal_profile(t: float, E0: float, critical_energy: float,
    b: float = 0.5, t0: float = -0.5):
    """Parametrization of the longitudinal shower profile, see equation 33.36 in
    the 2016 edition of the PDG.
    """
    tmax = np.log(E0 / critical_energy) + t0
    a = b * tmax + 1
    return E0 * b * (b * t)**(a - 1) * np.exp(-b * t) / scipy.special.gamma(a)



if __name__ == '__main__':
    parse_muon_energy_loss()
