# Copyright (C) 2026, Luca Baldini (luca.baldini@pi.infn.it)
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

"""Basic information about some cosmic-ray instruments.
"""


from dataclasses import dataclass


@dataclass(frozen=True)
class Detector:

    """Small class describing a generic cosmic-ray detector.
    """

    name: str
    technique: str
    energy_range: tuple[float, float]
    acceptance: float
    fov: float
    references: list[str]



@dataclass(frozen=True)
class SpaceDetector(Detector):

    """Small class describing a generic cosmic-ray detector in space.
    """

    launch_date: str
    decommission_date: str | None = None



@dataclass(frozen=True)
class GroundDetector(Detector):

    """Small class describing a generic cosmic-ray detector on the ground.
    """

    pass



AMS_02 = SpaceDetector(
    name='AMS-02',
    technique='magnetic spectrometer',
    energy_range=(0.1, 1000),  # GeV
    acceptance=0.5,  # m^2 sr
    fov=2 * 3.14159,  # sr
    references=[]
)


if __name__ == '__main__':
    print(AMS_02)