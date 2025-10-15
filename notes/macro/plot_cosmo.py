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

"""Plot some relevant quantities specific to the diffusive shock acceleration.
"""


import astropy.units as u
import numpy as np


from astropart.plot import plt, standard_figure, setup_ca, setup_ca_drawing,\
    save_cf, draw_line, draw_arrow, draw_rectangle, _srcp, draw_ellipse, draw_circle


standard_figure('single_component_scale')
a0 = 1.
H0 = 1.
t0 = 1.
t = np.linspace(0., 10., 500)

#x1, x2, y1, y2 = -1., 0., 0., 1.  # subregion of the original image
#axins = plt.gca().inset_axes([0.1, 0.65, 0.3, 0.3], xlim=(x1, x2), ylim=(y1, y2))
#plt.gca().indicate_inset_zoom(axins, edgecolor="black")


# Empty universe.
a_e = a0 * H0 * t
plt.plot(H0 * (t - t0), a_e, color='black', ls='dashed')
#axins.plot(H0 * (t - t0), a_e, color='black', ls='dashed')

# Matter dominated universe.
t0 = 2. / 3. / H0
a_m = a0 * (t / t0)**(2. / 3)
plt.plot(H0 * (t - t0), a_m, color='black')
plt.text(3., 3.4, '$a_m$')
#axins.plot(H0 * (t - t0), a_m, color='black')

# Radiation dominated universe.
t0 = 1. / 2. / H0
a_r = a0 * (t / t0)**(1. / 2)
plt.plot(H0 * (t - t0), a_r, color='black')
plt.text(4., 2.75, '$a_r$')
#axins.plot(H0 * (t - t0), a_r, color='black')

# Lambda dominated universe.
t = np.linspace(-2., 10., 500)
a_l = a0 * np.exp(H0 * t)
plt.plot(H0 * t, a_l, color='black')
plt.text(1., 4., '$a_\\Lambda$')
#axins.plot(H0 * t, a_l, color='black')

setup_ca(xlabel='$t - t_0~[H_0^{-1}]$', ylabel='$a / a_0$', grids=True,
    xmin=-2., xmax=5., ymin=0., ymax=5.)
save_cf()


standard_figure('single_component_lookback')
def _lookback_time(z, b):
    return (1. - (1 + z)**(1 - b)) / (b - 1)

z = np.geomspace(0.001, 100, 500)
plt.plot(z, _lookback_time(z, 3), color='black')
plt.plot(z, _lookback_time(z, 2.5), color='black')
setup_ca(xlabel='z', ylabel='Lookback time $[H_0^{-1}]$', logx=True, logy=True)



standard_figure('cosmo_map')

om = np.linspace(0.0001, 2., 100)

# Curvature.
_line_value = lambda x: 1 - x
plt.plot(om, _line_value(om), color='black', ls='dashed')
kwargs = dict(rotation=-33, ha='center', va='center')
#plt.text(0.25, 0.65, '$k = 0$', rotation=-33)
x = 0.3
y = _line_value(x)
dy = 0.075
plt.text(x, y + dy, 'positive curvature', **kwargs)
plt.text(x, y - dy, 'negative curvature', **kwargs)

# Acceleration.
_line_value = lambda x: 0.5 * x
plt.plot(om, _line_value(om), color='black', ls='dotted')
#plt.text(0.25, 0.175, '$q_0 = 0$', rotation=13)
kwargs = dict(rotation=17, ha='center', va='center')
x = 1.25
y = _line_value(x)
dy = 0.07
plt.text(x, y + dy, 'accelerating', **kwargs)
plt.text(x, y - dy, 'decelerating', **kwargs)

# Expansion.
def expansion_limit(om):
    neg = om[om <= 1]
    pos = om[om > 1]
    arg = 1. / 3. * np.arccos((1. - pos) / pos) + 4. * np.pi / 3.
    return np.concatenate((np.zeros(neg.shape), 4. * pos * np.cos(arg)**3.))
plt.plot(om, expansion_limit(om), color='black')
kwargs = dict(ha='center', va='center')
x =0.6
y = 0
dy = 0.07
plt.text(x, y + dy, 'expanding forever', **kwargs)
plt.text(x, y - dy, 're-collapsing', **kwargs)

x1, x2, y1, y2 = 1., 1.5, -0.0125, 0.0125  # subregion of the original image
axins = plt.gca().inset_axes([0.6, 0.725, 0.375, 0.25], xlim=(x1, x2), ylim=(y1, y2))
axins.plot(om, expansion_limit(om), color='black')
axins.grid()
plt.gca().indicate_inset_zoom(axins, edgecolor="black")

# Big bang.
def big_bang_limit(om):
    return 4. * om * np.cosh(1. / 3. * np.arccosh((1. - om) / om))**3.
plt.plot(om, big_bang_limit(om), color='black')
plt.gca().fill_between(om, big_bang_limit(om), 2, edgecolor='black', facecolor='none', hatch='\\\\\\')
kwargs = dict(ha='center', va='center')
plt.annotate('No big bang', (0.1, 1.35), (0.25, 1.1),
     arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleB=90, angleA=0'))



setup_ca(xlabel='$\\Omega_{m,0}$', ylabel='$\\Omega_{\\Lambda,0}$', xmin=0., xmax=1.5,
    ymin=-0.5, ymax=1.5, xticks=(-1, -0.5, 0., 0.5, 1., 1.5),
    yticks=(-0.5, 0., 0.5, 1., 1.5), grids=True)
save_cf()

plt.show()
