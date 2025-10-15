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

"""Atomic Mass Data Center interface.
"""

from dataclasses import dataclass

import numpy as np

from astropart import ASTROPART_DATA, logger
from astropart.plot import plt, setup_ca


# Dowloaded from https://www-nds.iaea.org/amdc/
_AMDC_FILE_PATH = ASTROPART_DATA / 'mass_1.mas20.txt'
_AMDC_COL_WIDTH = (1, 3, 5, 5, 5, 4, 6, 14, 13, 13, 10, 3, 13, 11, 17, 13)
_float = lambda x: None if x.strip() in ('*', '') else float(x.strip('#'))
_am = lambda x: 1.e-6 * float(x.strip('#').replace(' ', ''))
_AMDC_COL_FMT = (int, int, int, int, str, str, _float, _float, _float, _float, str, _float, _float, _am, _float)

# Downloaded from https://raw.githubusercontent.com/pyne/pyne/develop/pyne/dbgen/abundances.txt
_IUPAC_ABUND_FILE_PATH = ASTROPART_DATA / 'iupac_isotopic_abundance.txt'


@dataclass
class NuclearMassData:

    """Small containter class holding a row in the input data file.
    """

    asymmetry: int
    N: int
    Z: int
    A: int
    element: str
    o: str
    mass_excess: float
    mass_excess_unc: float
    binding_energy: float
    binding_energy_unc: float
    beta_marker: str
    beta_decay_energy: float
    beta_decay_energy_unc: float
    atomic_mass: float
    atomic_mass_unc: float

    def __post_init__(self):
        """Basic integrity checks.
        """
        assert self.asymmetry == self.N - self.Z
        assert self.beta_marker == 'B-'
        assert abs(self.atomic_mass - self.A) < 0.5

    def latex_symbol(self):
        """
        """
        return '{}_{%d}^{%d}\\mathrm{%s}' % (self.Z, self.A, self.element)



def _load_nuclear_mass_data(file_path: str = _AMDC_FILE_PATH) -> dict:
    """Load the nuclear data from the underlying file.

    col 1     :  Fortran character control: 1 = page feed  0 = line feed
    format    :  a1,i3,i5,i5,i5,1x,a3,a4,1x,f14.6,f12.6,f13.5,1x,f10.5,1x,a2,f13.5,f11.5,1x,i3,1x,f13.6,f12.6
                 cc NZ  N  Z  A    el  o     mass  unc binding unc      B  beta  unc    atomic_mass   unc
    Warnings  :  this format is not identical to that used in AME2016;
                 one more digit is added to the "BINDING ENERGY/A", "BETA-DECAY ENERGY" and "ATOMIC-MASS" values and their uncertainties;
                 # in a place of decimal point : estimated (non-experimental) value;
                 * in a place of value : the not calculable quantity
    """
    logger.info(f'Loading AMDC data from {file_path}...')
    with open(file_path, 'r') as input_file:
        while not input_file.readline().startswith('1N-Z'):
            pass
        input_file.readline()
        lines = input_file.readlines()
    markers = np.cumsum(_AMDC_COL_WIDTH)
    data = {}
    for line in lines:
        fields = [fmt(line[start:stop].strip()) for \
            start, stop, fmt in zip(markers, markers[1:], _AMDC_COL_FMT)]
        _data = NuclearMassData(*fields)
        data[(_data.Z, _data.A)] = _data
    logger.info(f'Done, {len(data)} isotope(s) found.')
    return data


def _load_isotopic_abundance_data(file_path: str = _IUPAC_ABUND_FILE_PATH) -> dict:
    """Load the isotopic abundance data.
    """
    logger.info(f'Loading IUPAC isotopica abundance data from {file_path}...')
    data = {}
    with open(file_path, 'r') as input_file:
        lines = input_file.readlines()
    for line in lines:
        if line.startswith('#'):
            continue
        Z, _, A, abundance = line.split()
        Z = int(Z)
        A = int(A)
        abundance = float(abundance)
        data[Z, A] = abundance
    logger.info(f'Done, {len(data)} isotope(s) found.')
    return data



_NUCLEAR_MASS_DATA_DICT = _load_nuclear_mass_data()
_ISOTOPIC_ABUNDANCE_DATA_DICT = _load_isotopic_abundance_data()

# Populate a list of stable isotopes, in the form of (Z, A) tuples, and order it
# by mass number.
STABLE_ISOTOPES = list(_ISOTOPIC_ABUNDANCE_DATA_DICT.keys())
STABLE_ISOTOPES.sort(key=lambda x: x[1])

def nuclear_mass_data(Z: int, A: int) -> NuclearMassData:
    """Return the nuclear mass data for a given isotope.
    """
    return _NUCLEAR_MASS_DATA_DICT[(Z, A)]

def isotopic_abundance(Z: int, A: int) -> float:
    """Return the isotopic abundance for a given isotope.
    """
    return _ISOTOPIC_ABUNDANCE_DATA_DICT[(Z, A)]
