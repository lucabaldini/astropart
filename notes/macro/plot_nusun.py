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

"""Plot some relevant quantities for the solar neutrinos.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.special import erf, erfc
from scipy.stats import norm

from astropart import amdc, nusun, numix
from astropart.ssm import BS05
from astropart.plot import plt, margin_figure, standard_figure, setup_ca, save_cf


def _draw_isotope(Z, A, hscale=1.32, voffset=0., annotate_energy=False):
    """
    """
    data = amdc.nuclear_mass_data(Z, A)
    energy = 1.e-3 * data.binding_energy
    plt.plot(A, energy, 'o', color='black', ms=3.)
    label = f'${data.latex_symbol()}$'
    if annotate_energy:
        label = f'{label} ({energy:.3f})'
    plt.text(A * hscale, energy + voffset, label, ha='center', va='center')


def _plot_line(energy, flux, **kwargs):
    """
    """
    plt.plot((energy, energy), (1.e-10, flux), **kwargs)


standard_figure('nucleon binding energy')
isotopes = amdc.STABLE_ISOTOPES[1:]
A = np.array([A for (Z, A) in isotopes])
binding_energy = np.array([amdc.nuclear_mass_data(Z, A).binding_energy for (Z, A) in isotopes])
plt.plot(A, 1.e-3 * binding_energy, color='black')
setup_ca(xlabel='Mass number', logx=True, ylabel='Binding enegry per nucleon [MeV]',
    xmin=1., ymin=0., ymax=10.)
_draw_isotope(1, 2, 1., -0.4, annotate_energy=False)
_draw_isotope(2, 3)
_draw_isotope(2, 4, 1., 0.3)
_draw_isotope(3, 6, 1., -0.4)
_draw_isotope(3, 7)
_draw_isotope(4, 9, 1.25, -0.35)
_draw_isotope(5, 11)
_draw_isotope(6, 12, 0.8, 0.3)
_draw_isotope(8, 16, 1., 0.3)
_draw_isotope(26, 56, 1., 0.5)
save_cf()


margin_figure('pp fusion', height=2.25)
r = np.linspace(1., 5., 100)
plt.plot(r, 1./r, color='black')
plt.hlines(-0.5, 0., 1., color='black')
plt.vlines(1., -0.5, 1., color='black')
plt.axhline(0.4, color='black', ls='dashed')
plt.text(3., 0.42, '$E$')
setup_ca(xlabel='$r$ [a.u]', ylabel='$U(r)$ [a. u.]', xticks=(1.,), yticks=(-0.5, 0., 1.))
plt.gca().set_xticklabels(('$r_0$',))
plt.gca().set_yticklabels(('$-U_0$', '0', '$U_\\mathrm{max}$'))
U_max = (const.e.gauss**2 / u.fm).to(u.eV)
p_c = (2. / np.pi * const.G * const.M_sun**2. / const.R_sun**4.).to(u.Pa)
#T_c = (4. * const.G * const.M_sun / 3. / const.R_sun / const.R ).to(u.K)
print(U_max)
print(f'{p_c.to_value() / 101325.:.2e}')
#print(f'{T_c:%.3e}')
save_cf()


margin_figure('gamow peak', height=2.25)
kT = 1.5
E_G = 600.
E = np.geomspace(.1, 1000., 100)
y1 = np.exp(-np.sqrt(E_G / E))
y2 = np.exp(-E / kT)
plt.plot(E, y1, ls='dashed', color='black')
plt.plot(E, y2, ls='dashed', color='black')
plt.plot(E, 1.e8 * y1 * y2, color='black')
setup_ca(xlabel='Energy [keV]', logx=True, logy=True, ymin=1.e-8, ymax=1.e3)
save_cf()

margin_figure('gamow peak gaussian', height=2.25)
kT = 1.5
E_G = 600.
E_0 = E_G**0.3333 * (kT / 2.)**0.6666
tau = 3. * E_0 / kT
sigma = 2. * E_0 / np.sqrt(tau)
print(E_0, tau, sigma)
E = np.linspace(0, 30., 200)
y1 = np.exp(-np.sqrt(E_G / E))
y2 = np.exp(-E / kT)
yg = tau / (sigma * np.sqrt(2. * np.pi)) * np.exp(-(E - E_0)**2. / 2. / sigma**2.)
#plt.plot(E, y1, ls='dashed', color='black')
#plt.plot(E, y2, ls='dashed', color='black')
plt.plot(E, 1.e8 * y1 * y2, color='black')
plt.plot(E, 60. * yg, color='black', ls='dashed')
setup_ca(xlabel='Energy [keV]')#, logx=True, logy=True, ymin=1.e-8, ymax=1.e3)
save_cf()

standard_figure('sun neutrino spectra')
print(f'Total flux from pp cycle: {nusun.FLUX_PP_CYCLE:.3e} cm^{-2} s^{-1}')
print(f'Total flux from CNO cycle: {nusun.FLUX_CNO_CYCLE:.3e} cm^{-2} s^{-1}')
print(f'Total flux: {nusun.FLUX_TOTAL:.3e} cm^{-2} s^{-1}')
pp_fmt = dict(color='black')
cno_fmt = dict(color='black', ls='dashed')
nusun.spec_pp.plot(color='black')
plt.text(0.3, 5.e10, f'\\textit{{pp}} ({100. * nusun.FLUX_PP / nusun.FLUX_TOTAL:.1f}%)')
_plot_line(nusun.ENERGY_PEP, nusun.FLUX_PEP, **pp_fmt)
plt.text(nusun.ENERGY_PEP, 2. * nusun.FLUX_PEP, f'\\textit{{pep}} ({100. * nusun.FLUX_PEP / nusun.FLUX_TOTAL:.1f}%)')
nusun.spec_hep.plot(**pp_fmt)
plt.text(10, 1.5e2, f'\\textit{{hep}}', ha='right')
for energy, yield_ in zip(nusun.ENERGIES_BE7, nusun.YIELDS_BE7):
    flux = nusun.FLUX_BE7 * yield_
    _plot_line(energy, flux, **pp_fmt)
    plt.text(energy, 1.5 * flux, f'${{}}^7Be$ ({100. * flux / nusun.FLUX_TOTAL:.1f}%)',
        ha='right' if energy < 0.5 else 'left')
nusun.spec_b8.plot(**pp_fmt)
plt.text(7., 1.e5, f'${{}}^8B$ ({100. * nusun.FLUX_B8 / nusun.FLUX_TOTAL:.3f}%)')
nusun.spec_n13.plot(**cno_fmt)
nusun.spec_o15.plot(**cno_fmt)
nusun.spec_f17.plot(**cno_fmt)
_plot_line(nusun.ENERGY_ECN, nusun.FLUX_ECN, **cno_fmt)
_plot_line(nusun.ENERGY_ECO, nusun.FLUX_ECO, **cno_fmt)
setup_ca(xlabel='Energy [MeV]', ylabel='Flux [cm$^{-2}$ s$^{-1}$ (100 keV)$^{-1}$ or cm$^{-2}$ s$^{-1}$]',
    xmin=0.1, xmax=30., ymin=1., ymax=1.e12, logx=True, logy=True)
save_cf()

standard_figure('sun neutrino location')
BS05.pdf_flux_pp.plot(color='black')
plt.text(0.13, 0.003, '\\textit{pp}')
BS05.pdf_flux_B8.plot(color='black')
plt.text(0.055, 0.007, '$^8B$')
BS05.pdf_flux_Be7.plot(color='black')
plt.text(0.075, 0.005, '$^7Be$')
BS05.pdf_flux_pep.plot(color='black')
plt.text(0.10, 0.0038, '\\textit{pep}')
#BS05.pdf_flux_hep.plot(color='black')
setup_ca(xlabel='$r$ [$R_\\odot$]', ylabel='Neutrino production [a. u.]', xmax=0.275, ymin=0.)
save_cf()


plt.show()
