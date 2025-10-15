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

from astropy.cosmology import Planck18 as cosmo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from astropart.plot import standard_figure, setup_ca, save_cf


def _parse_data(text):
    """
    """
    x = []
    y = []
    for line in text.split('\n'):
        if len(line) == 0:
            continue
        _x, _y = line.split(',')
        x.append(float(_x))
        y.append(float(_y))
    return np.array(x), np.array(y)


SN_DATA = _parse_data("""
0.1778140293637847, 0.5985981308411213
0.19575856443719408, 0.6492990654205607
0.22022838499184344, 0.7032710280373831
0.2561174551386623, 0.7735981308411214
0.29363784665579123, 0.8324766355140185
0.3393148450244698, 0.8913551401869158
0.3588907014681893, 0.9060747663551401
0.37520391517128876, 0.9093457943925232
0.3817292006525285, 0.894626168224299
0.3817292006525285, 0.8700934579439251
0.3719412724306688, 0.8275700934579437
0.35073409461663946, 0.7719626168224298
0.32463295269168024, 0.7179906542056073
0.27895595432300163, 0.636214953271028
0.23980424143556278, 0.5789719626168223
0.22185970636215335, 0.5560747663551401
0.20391517128874387, 0.5380841121495326
0.1827079934747145, 0.5233644859813082
0.16802610114192496, 0.5249999999999999
0.16476345840130507, 0.5429906542056074
0.16965742251223492, 0.5757009345794392
""")

CMB_DATA = _parse_data("""
0.9902120717781402, 0.18317757009345792
0.8548123980424143, 0.27967289719626165
0.7357259380097878, 0.3679906542056073
0.6460032626427405, 0.4350467289719626
0.5448613376835236, 0.5102803738317756
0.3980424143556281, 0.6214953271028035
0.1827079934747145, 0.7899532710280373
0.09624796084828713, 0.8668224299065419
0.0636215334420881, 0.8995327102803736
0.12234910277324634, 0.850467289719626
0.19902120717781407, 0.7915887850467288
0.38009787928221866, 0.6591121495327101
0.47797716150081565, 0.5871495327102803
0.5921696574225124, 0.5070093457943923
0.7520391517128875, 0.3974299065420559
0.9396411092985317, 0.2665887850467288
0.9885807504078303, 0.23224299065420562
""")

BAO_DATA = _parse_data("""
0.18923327895595432, 1.4932242990654203
0.18923327895595432, 1.4932242990654203
0.20554649265905384, 1.3264018691588784
0.21859706362153342, 1.1742990654205605
0.24306688417618272, 0.9665887850467288
0.2463295269168026, 0.8749999999999999
0.2626427406199021, 0.6901869158878503
0.28384991843393154, 0.44813084112149526
0.298531810766721, 0.25677570093457924
0.3181076672104404, 0.00981308411214954
0.38009787928221866, 0.017990654205607415
0.3572593800978792, 0.2551401869158878
0.3572593800978792, 0.2551401869158878
0.3393148450244698, 0.4497663551401867
0.3164763458401305, 0.6591121495327101
0.28711256117455136, 0.9453271028037381
0.26101141924959215, 1.1873831775700934
0.2446982055464927, 1.3345794392523362
0.2446982055464927, 1.3345794392523362
0.2267536704730832, 1.4866822429906539
""")

standard_figure('Cosmo constraints')
#plt.plot(*SN_DATA, 'o')
ellipse = Ellipse(
    xy=(0.272, 0.715),
    width=0.085,
    height=0.435,
    angle=-28.,
    facecolor='none',
    edgecolor='black'
)
plt.gca().add_patch(ellipse)
plt.text(0.385, 0.9, 'SNe')

#plt.plot(*CMB_DATA)
x = np.linspace(0.064, 1.)
y = 0.952 - 0.9 * x + 0.16 * x**2.
plt.plot(x, y, color='black')
y = 0.947 - 0.82 * x + 0.13 * x**2.
plt.plot(x, y, color='black')
plt.text(0.5, 0.6, 'CMB')

#plt.plot(*BAO_DATA)
x = np.linspace(0.2, 0.5)
y = 1.4 - 12. * (x - 0.2)
plt.plot(x, y, color='black')
y = 1.3 - 9.8 * (x - 0.25)
plt.plot(x, y, color='black')
plt.text(0.36, 0.3, 'BAO')

om = np.linspace(0.0001, 1., 100)
_line_value = lambda x: 1 - x
plt.plot(om, _line_value(om), color='black', ls='dashed')
plt.text(0.05, 0.925, '$k = 0$', rotation=-22)
# x = 0.3
# y = _line_value(x)
# dy = 0.075
#plt.text(x, y + dy, 'positive curvature', **kwargs)
#plt.text(x, y - dy, 'negative curvature', **kwargs)

_line_value = lambda x: 0.5 * x
plt.plot(om, _line_value(om), color='black', ls='dotted')
plt.text(0.12, 0.105, '$q_0 = 0$', rotation=8)
# kwargs = dict(rotation=17, ha='center', va='center')
# x = 0.2
# y = _line_value(x)
# dy = 0.07
# plt.text(x, y + dy, 'accelerating', **kwargs)
# plt.text(x, y - dy, 'decelerating', **kwargs)

x0 = 0.278
y0 = 1. - x0
plt.plot(x0, y0, 'o', color='black')
plt.annotate('Benchmark model', (x0, y0), (0.05, 0.3),
     arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleB=90, angleA=0'))

setup_ca(xlabel='$\\Omega_{m,0}$', ylabel='$\\Omega_{\\Lambda,0}$', xmin=0., xmax=0.6,
    ymin=0., ymax=1.25, grids=True)

save_cf()

plt.show()
