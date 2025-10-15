# Copyright (C) 2025, Luca Baldini (luca.baldini@pi.infn.it)
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


from astropy.cosmology import Planck18 as cosmo
from matplotlib import pyplot as plt
import numpy as np

from astropart.plot import setup_ca, standard_figure, subplot_vertical_stack, save_cf
import astropy.units as u
from scipy.interpolate import InterpolatedUnivariateSpline

z = np.geomspace(0.05, 80000., 200)
omega_gamma = cosmo.Ogamma(z)
omega_nu = cosmo.Onu(z)
omega_nurad = omega_gamma * (omega_nu[-1] / omega_gamma[-1])
omega_numat = omega_nu - omega_nurad
omega_r = omega_gamma + omega_nurad
omega_m = cosmo.Om(z)
omega_b = cosmo.Ob(z)
omega_dm = cosmo.Odm(z)
omega_lambda = cosmo.Ode(z)
age = cosmo.age(z).to(u.yr).value
age_spline = InterpolatedUnivariateSpline(np.flip(age), np.flip(z), k=3)

#assert np.allclose(omega_r + omega_m + omega_lambda, 1.)
assert np.allclose(omega_b + omega_dm, omega_m, 1.)


print(f'Cosmological parameters for {cosmo} today')
print(f'H_O = {cosmo.H0}')
print(f'Neutrino masses = {cosmo.m_nu}')
print(f'Age of the universe = {cosmo.age(0)}')
print(f'Omega_gamma = {cosmo.Ogamma(0)}')
print(f'Omega_nu = {cosmo.Onu(0)}')
print(f'Omega_m = {cosmo.Om(0)}')
print(f'Omega_b = {cosmo.Ob(0)}')
print(f'Omega_dm = {cosmo.Odm(0)}')
print(f'Omega_lambda = {cosmo.Ode(0)}')
print(f'Sum: {cosmo.Ogamma(0) + cosmo.Om(0) + cosmo.Onu(0) + cosmo.Ode(0)}')

z_mr = 3386.85
t_mr = cosmo.age(z_mr).to(u.yr)
a_mr = cosmo.scale_factor(z_mr)
z_lm = 0.305405
t_lm = cosmo.age(z_lm)
a_lm = cosmo.scale_factor(z_lm)
print(f'Radiation-matter equality at z = {z_mr}, t = {t_mr}, a = {a_mr}')
print(f'Lambda-matter equality at z = {z_lm}, t = {t_lm}, a = {a_lm}')

standard_figure('Benchmark density parameters', height=4.5, top=0.9)
ax1, ax2 = subplot_vertical_stack((0.7, 0.5))
# Radiation
plt.plot(z, omega_gamma, color='black', label=r'$\Omega_\gamma$', ls='dotted')
plt.plot(z, omega_nu, color='black', label=r'$\Omega_\nu$', ls='dotted')
plt.plot(z, omega_r, color='black', label=r'$\Omega_r$')
plt.text(2.e4, 1.25, r'$\Omega_r$')
plt.text(1., 9e-5, r'$\Omega_\gamma$')
plt.text(0.1, 8e-4, r'$\Omega_\nu$')

# Matter
plt.plot(z, omega_b, color='black', label=r'$\Omega_b$', ls='dotted')
plt.plot(z, omega_dm, color='black', label=r'$\Omega_dm$', ls='dotted')
plt.plot(z, omega_m, color='black', label=r'$\Omega_m$')
plt.text(50, 1.25, r'$\Omega_m$')
plt.text(100, 0.2, r'$\Omega_b$')
plt.text(10, 0.4, r'$\Omega_{dm}$')

# Dark energy
plt.plot(z, omega_lambda, color='black', label=r'$\Omega_\Lambda$')
plt.text(25, 2.e-4, r'$\Omega_\Lambda$')

plt.axvline(z_lm, color='black', ls='dashed')
plt.axvline(z_mr, color='black', ls='dashed')
plt.text(1.1 * z_lm, 2.5, r'$z=0.305$')
plt.text(1.1 * z_mr, 2.5, r'$z=3390$')

setup_ca(ylabel='Density parameters', logx=True, logy=True, ymin=5e-5)

plt.sca(ax2)
plt.plot(z, cosmo.scale_factor(z), color='black')
plt.axvline(z_lm, color='black', ls='dashed')
plt.axvline(z_mr, color='black', ls='dashed')
plt.annotate(r'$\frac{a}{a0} \approx 0.77$', (z_lm, a_lm), (2., 1e-2),
     arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleB=90, angleA=0'))
plt.annotate(r'$\frac{a}{a0} \approx 2.95 \times 10^{-4}$', (z_mr, a_mr), (10., 3e-1),
     arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleB=90, angleA=0'))


setup_ca(xlabel='$z$', ylabel='$a$/$a_0$', logx=True, logy=True, ymin=1e-5,
    yticks=np.geomspace(1e-5, 1., 6))

def z_to_age(z):
    return cosmo.age(z).to(u.yr).value

def age_to_z(age):
    return age_spline(age)

ax_top = ax1.secondary_xaxis('top', functions=(z_to_age, age_to_z))
ax_top.set_ticks((1e1, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10))
ax_top.set_xlabel('Age [yr]')

save_cf()


# plt.figure('Lookback time')
# plt.plot(z, cosmo.lookback_time(z))
# setup_ca(xlabel='$z$', ylabel='Lookback time [Gyr]', logx=True, logy=True)


# plt.figure('Scale factor vs. z')
# plt.plot(z, cosmo.scale_factor(z))
# setup_ca(xlabel='$z$', ylabel='$a / a_0$', logx=True, logy=True)


# plt.figure('Scale factor vs. time')
# plt.plot(cosmo.age(0) - cosmo.lookback_time(z), cosmo.scale_factor(z))
# setup_ca(xlabel='Lookback time [Gyr]', ylabel='$a / a_0$', logx=True, logy=True)




plt.show()
