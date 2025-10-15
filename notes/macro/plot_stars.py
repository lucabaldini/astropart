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

import numpy as np

from plotting import plt, setup_ca, save_cf, margin_figure, subplot_vertical_stack,\
    THIN_LW

import stars


def draw_filter(center, fwhm, label=None, ylabel=1.05):
    """Draw a filter at a given center and FWHM.
    """
    kwargs = dict(ec='k', color='lightgray', lw=THIN_LW)
    plt.axvspan(center - 0.5 * fwhm, center + 0.5 * fwhm, **kwargs)
    if label:
        kwargs = dict(ha='center', va='center', size='x-small')
        plt.text(center, ylabel, label, **kwargs)

def plot_planck(lambda_, T):
    """Small convenience function to draw the Plank spectral radiance at a
    given temperature.
    """
    radiance = stars.planck_lambda(lambda_, T)
    _max = radiance.max()
    plt.plot(lambda_, radiance / _max, 'k')
    plt.text(1500, 1.1, f'{T} $^\\circ$K', ha='right')
    draw_filter(stars.B_FILTER_CENTER, stars.B_FILTER_FWHM, 'B')
    draw_filter(stars.V_FILTER_CENTER, stars.V_FILTER_FWHM, 'V')
    kwargs = dict(ls='dashed', color='gray', lw=THIN_LW)
    x0 = 1100.
    Fb = stars.planck_lambda(stars.B_FILTER_CENTER, T) / _max
    Fv = stars.planck_lambda(stars.V_FILTER_CENTER, T) / _max
    plt.hlines(Fb, stars.B_FILTER_CENTER, x0, **kwargs)
    plt.text(x0, Fb, '$F_\\mathrm{B}$', va='center')
    plt.hlines(Fv, stars.V_FILTER_CENTER, x0, **kwargs)
    plt.text(x0, Fv, '$F_\\mathrm{V}$', va='center')
    #x0 = 1150.
    #bv = stars.temperature_to_bv(T)
    #plt.text(x0, 0.5 * (Fb + Fv), f'${bv:.3}$', va='center', size='small')


margin_figure('Wien law', height=2.5, left=0.3)
T = stars.DEFAULT_T_GRID
lambda_peak = stars.wien_lambda(T)
plt.plot(T, lambda_peak, color='k')
lambda_min, lambda_max = stars.VISIBLE_RANGE
plt.axhspan(lambda_min, lambda_max, ec='k', color='lightgray', lw=THIN_LW)
lambda_center = np.sqrt(lambda_min * lambda_max)
kwargs = dict(size='small', va='center', ha='left')
plt.text(7500., lambda_center, 'Visible light', **kwargs)
setup_ca(logx=True, logy=True, xmin=1000., xmax=100000.,
    xlabel='Temperature [$^\\circ$K]', ylabel='$\\lambda_\\mathrm{peak}$ [nm]')
save_cf()

margin_figure('Bolometric correction', height=2.5, left=0.3)
T = stars.DEFAULT_T_GRID
plt.plot(T, stars.bolometric_correction(T), color='k')
setup_ca(logx=True, xmin=1000., xmax=100000.,
    xlabel='Temperature [$^\\circ$K]', ylabel='Bolometric correction')
plt.plot(stars.SUN_EFFECTIVE_TEMPERATURE, stars.SUN_BOLOMETRIC_CORRECTION, 'o', color='k', ms=3.)
plt.text(stars.SUN_EFFECTIVE_TEMPERATURE, stars.SUN_BOLOMETRIC_CORRECTION + 0.1, 'Sun',
    va='bottom', ha='right', size='small')
save_cf()

margin_figure('Color index', height=2.5, left=0.3)
T = stars.DEFAULT_T_GRID
plt.plot(T, stars.temperature_to_bv(T), color='k')
setup_ca(logx=True, xmin=1000., xmax=100000.,
    xlabel='Temperature [$^\\circ$K]', ylabel='(B - V) color index')
plt.plot(stars.SUN_EFFECTIVE_TEMPERATURE, stars.SUN_BU_COLOR_INDEX, 'o', color='k', ms=3.)
plt.text(stars.SUN_EFFECTIVE_TEMPERATURE, stars.SUN_BU_COLOR_INDEX + 0.1, 'Sun',
    va='bottom', size='small')
save_cf()

margin_figure('Planck color index', left=0.15)
ax1, ax2 = subplot_vertical_stack((1., 1.))
lambda_min, lambda_max = 10., 1550.
lambda_ = np.linspace(lambda_min, lambda_max, 200)
kwargs = dict(xmin=0., ymin=0., ymax=1.25, yticks=[], ylabel='Radiance [a. u.]', grids=False)
plot_planck(lambda_, 4500)
setup_ca(**kwargs)
plt.sca(ax2)
plot_planck(lambda_, 12000)
setup_ca(xlabel='Wavelength [nm]', xmax=lambda_max, xticks=(0, 500, 1000, 1500), **kwargs)
save_cf()

plt.show()
