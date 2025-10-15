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

import numpy as np


from astropart.plot import plt, margin_figure, setup_ca, setup_ca_drawing,\
    save_cf, draw_line, draw_arrow, draw_rectangle, _srcp


def draw_shock(speed: str = None, label: bool = True, notes: str=None):
    """Draw the basic setup for the sketch of the shock.
    """
    l = 0.9
    draw_line((0., -l), (0., l))
    # See this interesting post as to why we need to set the alpha value explicitely
    # https://stackoverflow.com/questions/5195466
    kwargs = dict(edgecolor=None, facecolor='none', hatch='//', alpha=0.2)
    draw_rectangle((-1., -l), 1., 2. * l, **kwargs)
    plt.text(0., 0.95, 'shock front', ha='center')
    y = -1.05
    plt.text(0.15, y, 'upstream', ha='left')
    plt.text(-0.15, y, 'downstream', ha='right')
    if notes is not None:
        y = -1.325
        plt.text(0, y, notes, ha='center')
    draw_arrow((-1., 0.), (0.75, 0.))
    draw_line((0.7, 0.), (1.,0.))
    plt.text(0.7, -0.075, '$x$', va='top', ha='center')
    if speed is not None:
        l = 0.3
        y = 0.75
        draw_arrow((0., y), (l, y))
        plt.text(l, y, speed, va='center', ha='left')
    if label:
        y = -0.5
        plt.text(0.5, y, '$\\varrho_1,~P_1,~T_1$', ha='center')
        plt.text(-0.5, y, '$\\varrho_2,~P_2,~T_2$', ha='center')

def draw_velocity(label: str, x1: float, x2: float, y: float = 0.5):
    """Draw an annotated arrow representing the gas velocity.
    """
    draw_arrow((x1, y), (x2, y))
    plt.text(0.5 * (x1 + x2), y - 0.05, label, va='top', ha='center')

def draw_gas(xmin: float, xmax: float, num_points: int = 25, arrow_length: float = 0.15):
    """
    """
    np.random.seed(27875377)
    x = np.random.uniform(xmin, xmax, num_points)
    y = np.random.uniform(-0.8, 0.8, num_points)
    theta = np.random.uniform(0., 2. * np.pi, num_points)
    plt.plot(x, y, 'o', color='black', ms=2)
    for _x, _y, _theta in zip(x, y, theta):
        _dx = arrow_length * np.sin(_theta)
        _dy = arrow_length * np.cos(_theta)
        draw_arrow((_x, _y), (_x + _dx, _y + _dy), lw=0.25)

margin_figure('shock cartoon', height=2.25, left=0.05, bottom=0.05, right=0.95)
setup_ca_drawing()
draw_shock('$\\beta c$', notes='Lab system')
save_cf()

margin_figure('shock cartoon front', height=2.25, left=0.05, bottom=0.05, right=0.95)
setup_ca_drawing()
draw_shock(notes='Shock front at rest')
draw_velocity('$\\mathcal{v}_1 = \\beta c$', 0.75, 0.25)
draw_velocity('$\\mathcal{v}_2$', -0.25, -0.75)
save_cf()

margin_figure('shock cartoon upstream', height=2.25, left=0.05, bottom=0.05, right=0.95)
setup_ca_drawing()
draw_shock('$\\beta c$', False, notes='Upstream gas at rest')
draw_velocity('$\\frac{3}{4}\\beta c$', -0.75, -0.25)
draw_gas(0.1, 0.9)
save_cf()

margin_figure('shock cartoon downstream', height=2.25, left=0.05, bottom=0.05, right=0.95)
setup_ca_drawing()
draw_shock('$\\frac{1}{4} \\beta c$', False, notes='Downstream gas at rest')
draw_velocity('$-\\frac{3}{4} \\beta c$', 0.75, 0.25)
draw_gas(-0.9, -0.1)
save_cf()


if __name__ == '__main__':
    plt.show()
