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

from astropart.plot import plt, margin_figure, setup_ca, save_cf,\
    draw_line, setup_ca_drawing, draw_circle, draw_arc


def draw_cherenkov(x0, y0, speed, label):
    """
    """
    scale = 0.2
    draw_line((-1., y0), (1., y0), ls='dashed')
    for t in np.linspace(0., 4., 8):
        x = x0 - scale * speed * t
        r = scale * t
        draw_circle((x, y0), r, edgecolor='gray', facecolor='none')
    plt.text(-0.95, y0 + 0.8, label)
    plt.plot(x0, y0, 'o', color='black')
    if speed == 1:
        l = 1.2 * r
        draw_line((x0, -l), (x0, l))
    if speed > 1:
        _cos = 1. / speed
        theta = np.arccos(_cos)
        _sin = np.sin(theta)
        l = 1.2 * r / _cos
        draw_line((x0, y0), (x0 - l * _sin, y0 + l * _cos))
        draw_line((x0, y0), (x0 - l * _sin, y0 - l * _cos))


margin_figure('cherenkov_effect', height=5., left=0.0, bottom=0.0, right=1., top=1.)
setup_ca_drawing(xside=1., yside=3., equal_aspect=True)
draw_cherenkov(0.6, 2., 0.6, '$\\beta < \\frac{1}{n}$')
draw_cherenkov(0.9, 0., 1.0, '$\\beta = \\frac{1}{n}$')
draw_cherenkov(0.9, -2., 1.2, '$\\beta > \\frac{1}{n}$')
save_cf()

plt.show()
