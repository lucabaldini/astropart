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

"""Plot the Borexino results.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from astropart.plot import plt, standard_figure, setup_ca, save_cf

# Data taken from Nature, 496 (2018) 505
E = np.array([0.2675, 0.8628, 1.4439, 8.0913])
Pee = np.array([0.5660, 0.5316, 0.4286, 0.3717])
Pee_p1s = np.array([0.6581, 0.5867, 0.5416, 0.4494])
labels = ['pp', 'Be', 'pep', 'B']
sigma_Pee = Pee_p1s - Pee



xmsw = np.array([0.1, 0.30170615356996516, 0.6214506591171614, 1.0931838142488728, 1.6837020970221896, 2.829517864265699, 4.561669876646584, 6.996685212973025, 9.089033780120868, 12.829645309323372, 20.0])
ymsw = np.array([0.5460843373493975, 0.5433734939759035, 0.5325301204819276, 0.5180722891566264, 0.4981927710843373, 0.45662650602409627, 0.4051204819277107, 0.3599397590361445, 0.341867469879518, 0.32560240963855414, 0.32])
spline = InterpolatedUnivariateSpline(xmsw, ymsw, k=3)
x = np.geomspace(0.1, 20., 100)
y = spline(x)


standard_figure('solar nu survival')
plt.errorbar(E, Pee, yerr=sigma_Pee, fmt='o', color='black')
plt.plot(x, y, color='black')
for x, y, label in zip(E, Pee_p1s, labels):
    plt.text(x, y, label)
plt.text(15, 0.55, "Vacuum LMA", ha="right")
plt.text(15, 0.25, "MSW LMA", ha="right")
plt.axhline(0.538, color='black', ls='dashed')
setup_ca(xlabel='Energy [MeV]', ylabel='$P_{ee}$', logx=True, xmin=0.1, xmax=20.0,
         ymin=0.2, ymax=0.8)
save_cf()

plt.show()