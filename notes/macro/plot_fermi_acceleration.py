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


def draw_cloud(head_on: bool, beta: float = 0.3, hspan: float = 1., vspan: float = 0.9):
    """Draw the basic setup for the sketch of the cloud.
    """
    if head_on:
        label = "head-on $(\\cos\\theta > 0)$"
        xrect = -hspan
    else:
        label = "following $(\\cos\\theta < 0)$"
        xrect = 0.
    draw_line((0., -vspan), (0., vspan))
    # See this interesting post as to why we need to set the alpha value explicitly
    # https://stackoverflow.com/questions/5195466
    kwargs = dict(edgecolor="none", facecolor="none", hatch="//", alpha=0.2)
    draw_rectangle((xrect, -vspan), hspan, 2. * vspan, **kwargs)
    plt.text(0., vspan + 0.15, label, ha="center")
    draw_arrow((-hspan, 0.), (hspan, 0.))
    plt.text(hspan - 0.1, -0.05, "$x$", va="top", ha="center")
    # Draw the velocity of the cloud.
    y = 0.8 * vspan
    draw_arrow((0., y), (beta, y))
    plt.text(beta , y, "$\\beta c$", va='center', ha='left')
    # Draw the test particle trajectory.
    x = 0.8 * hspan
    if not head_on:
        x = -x
    y = 0.6 * vspan
    plt.plot(x, y, 'o', color='black', ms=4)
    draw_line((x, y), (0., 0.))
    draw_arrow((0., 0.), (x, -y))
    # Draw the angle arc.
    theta = np.degrees(np.arctan(1.25 * beta / y))
    if head_on:
        draw_arc((0., 0.), 0.2, 0., theta)
        plt.text(0.3, 0.05, '$\\theta$')
    else:
        draw_arc((0., 0.), 0.2, 0., 180. - theta)

        plt.text(-0.25, 0.2, '$\\theta$')


margin_figure('fermi_accel_headon', height=2., left=0.05, bottom=0.05, right=0.95)
setup_ca_drawing()
draw_cloud(head_on=True)
save_cf()

margin_figure('fermi_accel_follow', height=2., left=0.05, bottom=0.05, right=0.95)
setup_ca_drawing()
draw_cloud(head_on=False)
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
