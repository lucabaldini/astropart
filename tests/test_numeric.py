# Copyright (C) 2024, Luca Baldini (luca.baldini@pi.infn.it)
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

"""Test suite for numeric.py
"""

import pytest

import numpy as np

from astropart.numeric import integral, find_maximum


def test_integral():
    """Test the integral of a few know functions.
    """
    assert integral(lambda x: 1., 0., 1.) == pytest.approx(1.)
    assert integral(lambda x: x, 0., 1.) == pytest.approx(0.5)
    assert integral(lambda x: x**2., 0., 1.) == pytest.approx(1. / 3.)
    assert integral(lambda x: np.exp(-x**2.), -np.inf, np.inf) == pytest.approx(np.sqrt(np.pi))


def test_find_maximum():
    """Test the find_maximum() routine.
    """
    for mu in (-1., 0., 1.):
        assert find_maximum(lambda x: np.exp(-(x - mu)**2.), 10.) == pytest.approx(mu)
