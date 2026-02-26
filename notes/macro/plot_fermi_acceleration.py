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
    save_cf, draw_line, draw_arrow, draw_rectangle, draw_arc, _srcp


def draw_cloud(label: str, head_on: bool, beta: float = 0.3):
    """Draw the basic setup for the sketch of the cloud.
    """
    l = 0.9
    draw_line((0., -l), (0., l))
    # See this interesting post as to why we need to set the alpha value explicitely
    # https://stackoverflow.com/questions/5195466
    kwargs = dict(edgecolor='none', facecolor='none', hatch='//', alpha=0.2)
    draw_rectangle((-1., -l), 1., 2. * l, **kwargs)
    plt.text(0., 1.05, label, ha='center')
    draw_arrow((-1., 0.), (0.75, 0.))
    draw_line((0.7, 0.), (1.,0.))
    plt.text(0.7, -0.075, '$x$', va='top', ha='center')
    y = 0.75
    x = 0.3
    text = '$\\beta c$'
    #if speed < 0:
    #    x -= 0.325
    #    text = '$-\\beta c$'
    plt.text(x , y, text, va='center', ha='left')
    draw_arrow((0., y), (beta, y))
    draw_particle()
    # if beta > 0:
    #     draw_arc((0., 0.), 0.2, 0., np.degrees(np.arctan(1.25 * beta / y)))
    #     plt.text(0.3, 0.025, '$\\theta$')
    #     plt.text(0.95, 0.5, '$v_x = -c\\cos\\theta$', ha='right')
    # else:
    #     draw_arc((0., 0.), 0.2, -np.degrees(np.arctan(1.25 * beta / y)), 180.)
    #     plt.text(-0.25, 0.2, '$\\theta$')
    #     plt.text(0.95, 0.5, '$v_x = c\\cos\\theta$', ha='right')

def draw_velocity(label: str, x1: float, x2: float, y: float = 0.5):
    """Draw an annotated arrow representing the gas velocity.
    """
    draw_arrow((x1, y), (x2, y))
    plt.text(0.5 * (x1 + x2), y - 0.05, label, va='top', ha='center')

def draw_particle(y: float = 0.5):
    """Draw the particle.
    """
    plt.plot(1., y, 'o', color='black', ms=4)
    draw_line((1., y), (0., 0.))
    draw_arrow((0., 0.), (1., -y))


margin_figure('fermi_accel_headon', height=2., left=0.05, bottom=0.05, right=0.95)
setup_ca_drawing()
draw_cloud('head-on $(\\cos\\theta > 0)$', head_on=True)
save_cf()

margin_figure('fermi_accel_follow', height=2., left=0.05, bottom=0.05, right=0.95)
setup_ca_drawing()
draw_cloud('following $(\\cos\\theta < 0)$', head_on=False)
save_cf()

margin_figure('fermi_accel_average', height=2.)
p_even = lambda x, beta=1.e-4: np.abs(x)
p_odd = lambda x, beta=1.e-4: np.sign(x) * beta * (1. - x**2.)
p = lambda x, beta=1.e-4: p_even(x, beta) + p_odd(x, beta)
x = np.linspace(-1., 1., 1000)
plt.plot(x, p_even(x), color='k')
plt.plot(x, 1.e3 * p_odd(x), color='k', ls='dashed')
plt.text(-0.6, 0.4, '$p_\\mathrm{even}$')
plt.text(-0.1, -0.2, '$p_\\mathrm{odd} \\times 10^3$')
setup_ca(xlabel='$x = \\cos\\theta$', ylabel='$p(x)$', ymin=-0.25)
save_cf()

margin_figure('fermi_accel_efficiency', height=2.)
x = np.logspace(10, 16, 100)
t0 = 80.
t_esc_max = t0 * (x / 1e10)**-0.3
t_esc_min = t0 * (x / 1e10)**-0.6
xi = 3.e-8
lambdac = 0.326 # years, i.e., 0.1 pc
t_acc = 1.e-6 * lambdac * np.log(x / 1.e9) / xi
plt.plot(x, t_esc_max, color='black')
plt.plot(x, t_esc_min, color='black')
plt.fill_between(x, t_esc_max, t_esc_min, color='gray', alpha=0.2)
plt.plot(x, t_acc, color='black')
plt.text(1.e11, 3.e-1, 'escape')
plt.text(2.e12, 3.e1, 'acceleration')
setup_ca(xlabel='Energy [eV]', ylabel='Characteristic time [Myr]', logx=True, logy=True,
    xticks=(1.e10, 1.e12, 1.e14, 1.e16))
save_cf()


if __name__ == '__main__':
    plt.show()
