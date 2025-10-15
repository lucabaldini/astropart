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


from matplotlib.patches import Circle
import numpy as np

from plotting import plt, margin_figure, setup_ca, save_cf


def _draw_circle(radius, **kwargs):
    """Draw a circle.
    """
    circle = Circle((0., 0.), radius=radius, fill=False, **kwargs)
    plt.gca().add_artist(circle)
    return circle

def _draw_radius(radius, theta, label=None, **kwargs):
    """Draw a line.
    """
    theta = np.radians(theta)
    (x1, y1) = (0., 0.)
    (x2, y2) = (radius * np.cos(theta), radius * np.sin(theta))
    plt.plot((x1, x2), (y1, y2), color='k', **kwargs)
    if label is not None:
        (x2, y2) = (0.5 * radius * np.cos(theta), 0.5 * radius * np.sin(theta))
        plt.text(x2, y2, label, ha='center', va='center', backgroundcolor='white',
            bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none'))


margin_figure('gravitational_energy', height=1.9, left=0.05, bottom=0.05, right=0.8)
_draw_circle(1.)
_draw_radius(1., 45., '$R$')
_draw_circle(0.7, ls='dashed')
_draw_radius(0.7, -45., '$r$')
_draw_circle(0.75, ls='dashed')
plt.text(0.6, -0.6, '$dr$', ha='center', va='center')
plt.gca().set_aspect(1)
setup_ca(xmin=-1.05, xmax=1.05, ymin=-1.05, ymax=1.05, grids=False)
plt.axis('off')
save_cf()


if __name__ == '__main__':
    plt.show()
