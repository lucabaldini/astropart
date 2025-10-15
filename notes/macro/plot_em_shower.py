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

import numpy as np

from astropart.pdg import shower_longitudinal_profile
from astropart.plot import plt, margin_figure, setup_ca, save_cf,\
    draw_line, draw_wavy_line, setup_ca_drawing, draw_circle


margin_figure('em_shower_naive', left=0.05, bottom=0.05, right=0.95, top=0.95)
setup_ca_drawing(yside=2., equal_aspect=True)
vspace = 0.75
y0 = 2. - vspace
x0 = 0.
draw_line((x0, 2.), (x0, y0))
plt.text(0.05, 2. - 0.5 * vspace, '$e^{-}$', va='center')
vertices = [x0]
particles = ['e']
for t in range(4):
    y1 = y0 - vspace
    hspace = 0.5 / (t**1.5 + 1)
    _next_vertices = []
    _next_particles = []
    for x, p in zip(vertices, particles):
        x0 = x - hspace
        x1 = x + hspace
        if p == 'e':
            p0, p1 = 'g', 'e'
            draw_wavy_line((x, y0), (x0, y1))
        else:
            p0, p1 = 'e', 'e'
            draw_line((x, y0), (x0, y1))
        draw_line((x, y0), (x1, y1))
        _next_vertices += [x0, x1]
        _next_particles += [p0, p1]
    plt.axhline(y0, color='lightgray', ls='dashed')
    plt.text(0.9, y0 + 0.02, f'$t = {t + 1}$')
    if t == 0:
        plt.text(-0.45, 0.5 * (y0 + y1), '$\gamma$', va='center')
    if t == 1:
        plt.text(-0.875, 0.5 * (y0 + y1), '$e^+$', va='center')
        plt.text(-0.3, 0.5 * (y0 + y1), '$e^-$', va='center')
    y0 -= vspace
    vertices = _next_vertices
    particles = _next_particles
save_cf()


margin_figure('em_shower_feynman', left=0.05, bottom=0.05, right=0.95, top=0.95)
setup_ca_drawing(yside=2., equal_aspect=True)
l = 0.7
y0, y1, y2, y3 = 0.25, 0.55, 1.45, 1.75
kwargs = dict(va='center')
# Bremsstrahlung...
draw_line((-l, y3), (0., y2))
draw_line((0., y2), (0., y1))
draw_wavy_line((-l, y0), (0., y1))
draw_line((0., y1), (l, y0))
draw_wavy_line((0., y2), (l, y3))
plt.text(-l - 0.2, y3, '$e^-$', **kwargs)
plt.text(l + 0.05, y0, '$e^-$', **kwargs)
plt.text(l + 0.05, y3, '$\\gamma$', **kwargs)
draw_circle((-l, y0), 0.1, zorder=10)
plt.scatter(-l, y0, color='k', s=75, marker='x', zorder=11)
plt.text(-l, y0 + 0.15, '$N$', ha='center')

# ... and pair production.
draw_wavy_line((-l, -y3), (0., -y2))
draw_line((0., -y2), (0., -y1))
draw_wavy_line((-l, -y0), (0., -y1))
draw_line((0., -y1), (l, -y0))
draw_line((0., -y2), (l, -y3))
plt.text(-l - 0.2, -y0, '$\\gamma$', **kwargs)
plt.text(l + 0.05, -y0, '$e^+$', **kwargs)
plt.text(l + 0.05, -y3, '$e^-$', **kwargs)
draw_circle((-l, -y3), 0.1, zorder=10)
plt.scatter(-l, -y3, color='k', s=75, marker='x', zorder=11)
plt.text(-l, -y3 + 0.15, '$N$', ha='center')
save_cf()


margin_figure('shower_longitudinal_profile', height=2.25)
t = np.linspace(0., 30., 100)
E0 = 100
critical_energy = 0.0101
dE = shower_longitudinal_profile(t, E0, critical_energy, t0=-0.5)
plt.plot(t, dE / E0, color='k', label='Electrons')
dE = shower_longitudinal_profile(t, E0, critical_energy, t0=0.5)
plt.plot(t, dE / E0, color='k', ls='dashed', label='Photons')
setup_ca(xlabel='$t$ [$X_0$]', ylabel='$\\frac{1}{E_0}\\frac{dE}{dt}$ [GeV $X_0^{-1}$]',
    ymin=0., ymax=0.14, xticks=np.linspace(0., 30., 7), legend=True,
    yticks=np.linspace(0., 0.14, 8))
save_cf()


plt.show()
