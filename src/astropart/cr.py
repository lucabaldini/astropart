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

"""Simple interface to the cosmic-ray database at https://lpsc.in2p3.fr/crdb/
"""

from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from astropart import ASTROPART_DATA, logger


_CRDB_DUMP_FILE_PATH = ASTROPART_DATA / 'csv_all_database.dat'
_CRDB_PICKLE_FILE_PATH = ASTROPART_DATA / 'crdb.pickle'



class EnergyType(Enum):

    """Enum class for the ``EAXIS TYPE`` column in the cosmic-ray database.
    """

    KINETIC_ENERGY_PER_NUCLEON = 'EKN'
    KINETIC_ENERGY = 'EK'
    RIGIDITY = 'R'
    TOTAL_ENERGY = 'ETOT'
    TOTAL_ENERGY_PER_NUCLEON = 'ETOTN'



class ParticleType(Enum):

    """Enum class for the ``QUANTITY NAME`` column in the cosmic-ray database.
    """

    ALL_PARTICLES = 'AllParticles'
    NUCLEI = (
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
        'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
        'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
        'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
        'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
        'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
        'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es'
        )
    NUCLEI_GROUPS = (
        'H-He-group', 'N-group', 'O-group', 'Al-group', 'Si-group', 'Fe-group',
        'O-Fe-group', 'C-Fe-group', 'He-C-group', 'Si-Fe-group', 'C-group', 'CNO-group',
        'NeMgSiS-group', 'Zgeq1', 'Zgeq2', 'Zgeq3', 'Zgeq4', 'Zgeq5', 'Zgeq6', 'Zgeq7',
        'Zgeq8', 'Zgeq55', 'Zgeq65', 'Zgeq70', 'Zgeq88', 'Zgeq94', 'Zgeq110', 'Z_3-5',
        'Z_16-24', 'Z_17-20', 'Z_17-25', 'Z_21-23', 'Z_21-24', 'Z_21-25', 'Z_33-34',
        'Z_35-36', 'Z_37-38', 'Z_39-40', 'Z_41-42', 'Z_43-44', 'Z_45-46', 'Z_47-48',
        'Z_49-50', 'Z_51-52', 'Z_53-54', 'Z_55-56', 'Z_57-58', 'Z_59-60', 'Z_62-69',
        'Z_70-73', 'Z_74-80', 'Z_81-87', 'Z_74-87'
        )
    ISOTOPES = (
        '1H', '2H', '3He', '4He', '6Li', '7Li', '7Be', '9Be', '7Be+9Be', '9Be+10Be',
        '10B', '10Be', '11B', '12C', '13C', '14N', '14C', '15N', '16O', '17O', '18O',
        '19F', '20Ne', '21Ne', '22Ne', '23Na', '24Mg', '25Mg', '26Mg', '26Al', '27Al',
        '28Si', '29Si', '30Si', '31P', '32S', '33S', '34S', '35Cl', '36S', '36Ar',
        '36Cl', '37Cl', '37Ar', '38Ar', '39K', '40Ar', '40Ca', '40K', '41K', '41Ca',
        '42Ca', '43Ca', '44Ca', '44Ti', '45Sc', '46Ti', '46Ca', '47Ti', '48Ti',
        '48Ca', '48Cr', '49Ti', '49V', '50Ti', '50Cr', '50V', '51V', '51Cr', '52Cr',
        '53Cr', '53Mn', '54Cr', '54Fe', '54Mn', '55Mn', '55Fe', '56Fe', '56Ni',
        '57Fe', '57Co', '58Fe', '58Ni', '59Co', '59Ni', '60Ni', '60Fe', '61Ni',
        '62Ni', '63Cu', '64Ni', '64Zn', '65Cu', '66Zn', '67Zn', '68Zn', '70Zn',
        '67Ga', '68Ge', '69Ga', '70Ge', '71Ga', '71Ge', '72Ge', '72Se', '73Ge',
        '73As', '74Ge', '74Se', '74+75+76+77+78Se', '75As', '75Se', '76Ge', '76Se',
        '76Kr', '77Se', '78Se', '78Kr', '78+80+81+82Kr', '79Se', '79Br', '80Se',
        '80+82Se', '80Kr', '81Br', '81Kr', '82Se', '82Sr', '82Kr', '83Kr', '83+84+86Kr',
        '83Rb', '84Kr', '84Sr', '84+85+86Sr', '85Rb', '85Sr', '86Kr', '86Sr', '87Rb',
        '87Sr', '87+88Sr', '88Sr'
        )
    ANTINUCLEI = (
        'H-bar', 'He-bar', 'Li-bar', 'Be-bar', 'B-bar', 'C-bar', 'N-bar', 'O-bar'
        )
    ANTINUCLEI_GROUPS = (
        'Zgeq1-bar', 'Zgeq2-bar', 'Zgeq3-bar', 'Zgeq4-bar', 'Zgeq5-bar', 'Zgeq6-bar',
        'Zgeq7-bar', 'Zgeq8-bar'
        )
    ANTIISOTOPES = (
        '1H-bar', '2H-bar', '3He-bar', '4He-bar', '6Li-bar', '9Be-bar', '11B-bar',
        '12C-bar', '14N-bar', '16O-bar'
        )
    LEPTONS = (
        'e-', 'e+', 'e-+e+'
        )
    NEUTRINOS = (
        'NU_E', 'NU_M', 'NU_T'
        )
    GAMMA = 'GAMMA'
    MISC = (
        '<LnA>', '<X_max>', 'X_mu_max', '<rho_mu_600>', '<rho_mu_800>', '<R_mu>',
        'DipoleAmplitude', 'DipolePhase'
        )



@dataclass
class Record:

    """Small dataclass representing a single record in the cosmic-ray database.

    According to the documentation in the csv file, the columns represent:

    * Col. 1: QUANTITY NAME (case insensitive)
    * Col. 2: SUB-EXP NAME (case insensitive, no space)
    * Col. 3: EAXIS TYPE: EKN, EK, R, ETOT, or ETOTN
    * Col. 4: <E>: mean value bin mean value bin (all in [GeV] except for R in [GV])
    * Col. 5: EBIN_LOW
    * Col. 6: EBIN_HIGH
    * Col. 7: QUANTITY VALUE: [#/sr/s/m2/EAxis] if flux , no unit if ratio
    * Col. 8: ERR_STAT-
    * Col. 9: ERR_STAT+
    * Col.10: ERR_SYST-
    * Col.11: ERR_SYST+
    * Col.12: ADS URL FOR PAPER REF (no space)
    * Col.13: phi [MV]
    * Col.14: DISTANCE EXP IN SOLAR SYSTEM [AU]
    * Col.15: DATIMES: format = yyyy/mm/dd-hhmmss:yyyy/mm/dd-hhmmss;...
    * Col.16: IS UPPER LIMIT: format = 0 or 1
    """

    quantity: str
    experiment: str
    energy_type: EnergyType
    energy: float
    energy_lo: float
    energy_hi: float
    value: float
    stat_err_neg: float
    stat_err_pos: float
    sys_err_neg: float
    sys_err_pos: float
    ads_id: str
    modulation_potential: float
    distance: float
    dates: tuple
    upper_limit: bool

    _DATE_FMT = '%Y/%m/%d-%H%M%S'

    @staticmethod
    def _parse_upper_limit(value: str) -> bool:
        """Parse the upper limit field in a database record.
        """
        if value == '0':
            return False
        if value == '1':
            return True
        raise ValueError(f'Invalid bool "{value}" in cosmic-ray databse record')

    @staticmethod
    def _format_arguments(arguments: list, formatter: Callable, *indices: int) -> None:
        """Format a specific set of indices in a given list of arguments.
        """
        for i in indices:
            arguments[i] = formatter(arguments[i])

    @classmethod
    def from_csv_line(cls, line: str, separator: str = ',') -> Record:
        """Create a Record object from a single text line in the input csv file.
        """
        args = [item.strip('""') for item in line.strip('\n').split(separator)]
        cls._format_arguments(args, EnergyType, 2)
        cls._format_arguments(args, float, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13)
        cls._format_arguments(args, cls._parse_upper_limit, 15)
        return cls(*args)

    def unique_key(self) -> str:
        """Return a unique key for the record.
        """
        return f'{self.experiment} {self.quantity}'



class DataSet:

    """Small class representing a single dataset.
    """

    def __init__(self, record: Record) -> None:
        """Constructor.
        """
        self.unique_key = record.unique_key()
        self.quantity = record.quantity
        self.experiment = record.experiment
        self.energy_type = record.energy_type
        self.ads_id = record.ads_id
        self.modulation_potential = record.modulation_potential
        self.distance = record.distance
        self.dates = record.dates
        self.energy = np.array([])
        self.energy_lo = np.array([])
        self.energy_hi = np.array([])
        self.value = np.array([])
        self.stat_err = np.array([[], []])
        self.sys_err = np.array([[], []])
        self.upper_limit = np.array([])
        self.add_record(record)

    def add_record(self, record: Record) -> None:
        """Add a single record to the dataset.
        """
        self.energy = np.append(self.energy, record.energy)
        self.energy_lo = np.append(self.energy_lo, record.energy_lo)
        self.energy_hi = np.append(self.energy_hi, record.energy_hi)
        self.value = np.append(self.value, record.value)
        self.stat_err = np.append(self.stat_err, [[record.stat_err_neg], [record.stat_err_pos]], axis=1)
        self.sys_err = np.append(self.sys_err, [[record.sys_err_neg], [record.sys_err_pos]], axis=1)
        self.upper_limit = np.append(self.upper_limit, record.upper_limit)

    def min_energy(self) -> float:
        """Return the minimum energy in the dataset.
        """
        return self.energy_lo.min()

    def max_energy(self) -> float:
        """Return the maximum energy in the dataset.
        """
        return self.energy_hi.max()

    def energy_range(self) -> tuple[float, float]:
        """Return the energy range for the dataset.
        """
        return self.min_energy(), self.max_energy()

    def energy_units(self) -> str:
        """Return the energy units for the dataset.
        """
        if self.energy_type == EnergyType.RIGIDITY:
            return 'GV'
        return 'GeV'

    def copy(self) -> DataSet:
        """Create a copy of the object.
        """
        return copy.deepcopy(self)

    def scale_energy(self, scale: float) -> None:
        """Scale the energy in place.
        """
        self.energy *= scale

    def scale_values(self, scale: float) -> None:
        """Scale the values and the errors in place.
        """
        self.value *= scale
        self.stat_err *= scale
        self.sys_err *= scale

    def convert_energy(self, scale: float) -> DataSet:
        """
        """
        dataset = self.copy()
        dataset.scale_energy(scale)
        dataset.scale_values(1. / scale)
        return dataset

    def rigidity_to_kinetic_energy(self, Z, m) -> DataSet:
        """
        """
        assert self.energy_type == EnergyType.RIGIDITY
        R = self.energy
        p = Z * R
        Ek = np.sqrt(m**2. + p**2.) - m
        dataset = self.convert_energy(Ek / R)
        dataset.energy_type = EnergyType.KINETIC_ENERGY
        return dataset

    def __add__(self, other) -> DataSet:
        """
        """
        emin = max(self.min_energy(), other.min_energy())
        emax = min(self.max_energy(), other.max_energy())
        dataset = self.copy()
        mask = np.logical_and(dataset.energy >= emin, dataset.energy <= emax)
        spline = InterpolatedUnivariateSpline(other.energy, other.value, k=1)
        dataset.energy = dataset.energy[mask]
        dataset.value = dataset.value[mask]
        dataset.value += spline(dataset.energy)
        shape = (2, len(dataset.energy))
        dataset.stat_err = np.zeros(shape)
        dataset.sys_err = np.zeros(shape)
        return dataset

    def __or__(self, other) -> DataSet:
        """
        """
        dataset = self.copy()
        dataset.energy = np.append(self.energy, other.energy)
        dataset.energy_lo = np.append(self.energy_lo, other.energy_lo)
        dataset.energy_hi = np.append(self.energy_hi, other.energy_hi)
        dataset.value = np.append(self.value, other.value)
        shape = (2, len(dataset.energy))
        dataset.stat_err = np.zeros(shape)
        dataset.sys_err = np.zeros(shape)
        return dataset

    def plot(self, xscale: float = 1., label: str = None, **kwargs):
        """
        """
        plt.errorbar(self.energy * xscale, self.value, self.stat_err, label=label, **kwargs)

    def __str__(self):
        """String formatting.
        """
        energy_range = f'{self.min_energy():.1e}--{self.max_energy():.1e} {self.energy_units()}'
        return f'[{self.experiment:.20}] {self.quantity} ({self.ads_id}) {energy_range}'



class DataBase(dict):

    """Small contaire class representing a database of datasets.

    This is basically a dictionary, on top of which we build some additional
    facility.
    """

    def filter(self, quantity: str, pattern: str = None) -> list[DataSet]:
        """Filter the database and return a list of datasets for a given
        quantity and (optionally) matching a given experiment pattern.
        """
        return [dataset for dataset in self.values()\
            if dataset.quantity == quantity and\
            (pattern is None or pattern.lower() in dataset.experiment.lower())]



def pickle_crdb_dump():
    """Parse the content of the full dump of the cosmic-ray database file
    in cvs format and serialize it to a pickle file.
    """
    logger.info(f'Reading raw cosmic-ray database from {_CRDB_DUMP_FILE_PATH}...')
    with open(_CRDB_DUMP_FILE_PATH, 'r') as input_file:
        data = input_file.readlines()
    logger.info('Reassembling data sets...')
    data_set_dict = {}
    for line in data:
        if line.startswith('#'):
            continue
        record = Record.from_csv_line(line)
        key = record.unique_key()
        if key in data_set_dict:
            data_set_dict[key].add_record(record)
        else:
            data_set_dict[key] = DataSet(record)
    logger.info(f'Done, {len(data_set_dict)} distincts data set(s) found.')
    logger.info(f'Writing cosmic-ray datasets to {_CRDB_PICKLE_FILE_PATH}...')
    with open(_CRDB_PICKLE_FILE_PATH, 'wb') as output_file:
        pickle.dump(data_set_dict, output_file)
    logger.info('Done.')

def load_crdb():
    """
    """
    logger.info(f'Loading cosmic-ray data from {_CRDB_PICKLE_FILE_PATH}...')
    with open(_CRDB_PICKLE_FILE_PATH, 'rb') as input_file:
        data_set_dict = pickle.load(input_file)
    return DataBase(data_set_dict)




if __name__ == '__main__':
    #pickle_crdb_dump()
    db = load_crdb()
    for dataset in db.filter('AllParticles', 'Auger'):
        print(dataset)
        print(f'"{dataset.unique_key}"')
