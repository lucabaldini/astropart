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

"""Plotting data
from https://ui.adsabs.harvard.edu/abs/1912HarCi.173....1L%2F/abstract
"""

import numpy as np
from scipy.optimize import curve_fit

from plotting import plt, residual_figure, setup_ca, save_cf, DATA_FOLDER_PATH

file_path = DATA_FOLDER_PATH / 'leavitt_1912_tab1.txt'
_, max_brightness, min_brightness, _, period = np.loadtxt(file_path, unpack=True)
mean_brightness = np.sqrt(min_brightness * max_brightness)


def fit_function(x, norm, index):
    """Custom power law, where the y axis is expressed in magnitude.
    """
    return np.log10(norm * x**-index)


_, ax1, ax2 = residual_figure('Leavitt 1912')
fmt = dict(color='gray', ls='none', fillstyle='none')
ax1.errorbar(period, mean_brightness, fmt='o', color='k', label='Geometric mean')
ax1.errorbar(period, max_brightness, marker=7, label='Maximum', **fmt)
ax1.errorbar(period, min_brightness, marker=6, label='Minimum', **fmt)
ax1.invert_yaxis()
x = np.geomspace(1., 200., 100)
popt, pcov = curve_fit(fit_function, period, mean_brightness)
ax1.plot(x, fit_function(x, *popt), color='k')
index = popt[1]
sigma_index = np.sqrt(pcov[1, 1])
print(index, sigma_index)
setup_ca(xlabel='Period [days]', ylabel='Brightness', logx=True, ymin=17., ymax=11., legend=True)
plt.sca(ax2)
ax2.errorbar(period, mean_brightness - fit_function(period, *popt), fmt='o', color='k')
plt.axhline(0., color='k')
x = 100.
y = np.log10(2.)
plt.axhline(y, color='k', ls='dashed')
plt.text(x, y * 1.25, '$\\times 2$')
y = np.log10(0.5)
plt.axhline(y, color='k', ls='dashed')
plt.text(x, y * 1.25, '$\\times 1/2$', va='top')
setup_ca(xlabel='Period [days]', ylabel='Reduals', grids=True, ymin=-0.75, ymax=0.75)



save_cf()


if __name__ == '__main__':
    plt.show()
