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


import crdb
import numpy as np
from scipy.optimize import curve_fit

from astropart import logger
from astropart.cr import load_crdb, DataSet, EnergyType
from astropart.plot import plt, setup_ca, full_figure, margin_figure, standard_figure,\
    save_cf, draw_line


#
# These are extracted by hand (with webplotdigitizer) from
# https://github.com/crdb-project/tutorial?tab=readme-ov-file
#
GCR_ABUNDANCE = np.array([
    2989303651.1231494,
    185664828.89858332,
    649686.1118251893,
    328311.9154894636,
    959587.7397962044,
    5827366.2277948875,
    1285643.3896688027,
    6118499.813572696,
    76051.44879392906,
    913928.1454401532,
    130018.90814439915,
    1166205.9269927915,
    182900.53832109208,
    1007528.47470708,
    23602.834945627685,
    182900.53832109208,
    22479.752787972076,
    56763.82614559483,
    30118.085480524183,
    88028.9267910308,
    14495.653026436868,
    49040.370873850436,
    26020.129055198686,
    56763.82614559483,
    44484.47495984261,
    618772.4151464485,
    2386.987599674544,
    34861.436355210084,
    293.37743694574164,
    356.5473917838292,
    29.669669157412358,
    46.01150612139718,
    6.872797538565672,
    26.91332938926638,
    6.234308276233213,
    12.953223234102516,
    4.431796442766143,
    20.087763937159945,
    5.6551352582432175,
    5.937664232986228,
    5.129767950510835,
    5.6551352582432175
])



db = load_crdb()

standard_figure('cr_intensity', height=6., bottom=0.1, top=0.9, right=0.95)

mp = 0.938

ams_H = db['AMS02 (2011/05-2018/05) H'].rigidity_to_kinetic_energy(1., mp)
ams_He = db['AMS02 (2011/05-2018/05) He'].rigidity_to_kinetic_energy(2., 4. * mp)
ams_all = ams_H + ams_He
calet_H = db['CALET (2015/10-2021/12) H']
calet_He = db['CALET (2015/10-2022/04) He']
calet_all = calet_H + calet_He
hawk_all = db['HAWC (2018-2019) QGSJet-II-04 AllParticles']
icecubetop_all = db['IceCube+IceTop (2010/06-2013/05) SIBYLL2.1 AllParticles']
icetop_all = db['IceTop (2016/05-2017/04) SIBYLL2.1 AllParticles']
icecube_all = icecubetop_all | icetop_all
auger_all = db['Auger SD750+SD1500 (2014/01-2018/08) AllParticles']

def _energy_range(dataset, label, ylogoff=10.):
    """
    """
    emin, emax = 1.e9 * dataset.min_energy(), 1.e9 * dataset.max_energy()
    emean = np.sqrt(emin * emax)
    y0 = ylogoff * dataset.value.max()
    draw_line((emin, y0), (emax, y0), ls='dashed')
    plt.text(emean, 0.2 * ylogoff * y0, label, ha='center', va='bottom', size='small')

# Draw all the relevant datasets.
scale = 1.e9
kwargs = dict(fillstyle='none', ls='none', ms=3.)
ams_all.plot(scale, label='AMS-02 (H + He)', marker='o', color='k', **kwargs)
_energy_range(ams_all, 'AMS-02 (H + He)')
calet_all.plot(scale, label='CALET (H + He)', marker='s', color='k', **kwargs)
_energy_range(calet_all, 'CALET (H + He)')
hawk_all.plot(scale, label='HAWK', marker='^', color='gray', **kwargs)
_energy_range(hawk_all, 'HAWK')
icecube_all.plot(scale, label='Icecube', marker='s', color='gray', **kwargs)
_energy_range(icecube_all, 'Icecube')
auger_all.plot(scale, label='Auger', marker='o', color='gray', **kwargs)
_energy_range(auger_all, 'Auger')

# Broadband fit to the all-particle CR spectrum.
energy = np.array([])
intensity = np.array([])
for dataset in (ams_all, calet_all, hawk_all, icecube_all, auger_all):
    energy = np.append(energy, dataset.energy)
    intensity = np.append(intensity, dataset.value)

emin = 1.e10
emax = 1.e18
enorm = 1.e9
epivot = 1.e14

def power_law(x, norm, index):
    return norm * (x / epivot)**(-index)

energy *= enorm
mask = np.logical_and(energy >= emin, energy <= emax)
sigma = 0.01 * intensity
p0 = (1.e-9, 2.7)
logger.info(f'Fitting all particle CR spectrum between {emin} and {emax} eV...')
popt, pcov = curve_fit(power_law, energy[mask], intensity[mask], p0, sigma[mask])
fit_norm, fit_index = popt
fit_norm *= (epivot / enorm)**fit_index
logger.info(f'Index: {fit_index}')
logger.info(f'Normalization: {fit_norm:3e}')
logger.info(f'Intensity @ {enorm} eV: {power_law(enorm, *popt)}')
x = np.geomspace(1e9, 1e22, 100)
plt.plot(x, power_law(x, *popt), color='k', ls='dashed')

x1, x2, y1, y2 = 1.e12, 1.e21, 1.e-30, 1.e-10  # subregion of the original image
axins = plt.gca().inset_axes([0.1, 0.06, 0.55, 0.275])
index = 2.75

x = auger_all.energy * 1.e9
y = auger_all.value * x**index
dy = auger_all.stat_err * x**index
axins.errorbar(x, y, dy, marker='o', color='gray', **kwargs)
x = icecube_all.energy * 1.e9
y = icecube_all.value * x**index
dy = icecube_all.stat_err * x**index
axins.errorbar(x, y, dy, marker='s', color='gray', **kwargs)
axins.set_xscale('log')
axins.set_yscale('log')
axins.set_xlim(2.e14, 2.e20)
axins.set_xticks([1.e15, 1.e16, 1.e17, 1.e18, 1.e19, 1.e20])
axins.set_ylim(1.e27, 2.e30)
axins.set_yticks([1.e28, 1.e29, 1.e30], [])
axins.grid(True, which='both')
axins.set_ylabel('Intensity $\\times E^{%.2f}$' % index, labelpad=-2.)
knee_energy = 2.e15
ankle_energy = 6.e18
kwargs = dict(ha='center', arrowprops=dict(arrowstyle='->'))
axins.annotate('Knee', (knee_energy, 5.e29), xytext=(knee_energy, 1.e28), **kwargs)
axins.annotate('Ankle', (ankle_energy, 3.e28), xytext=(ankle_energy, 5.e29), **kwargs)
#plt.gca().indicate_inset_zoom(axins, edgecolor="black")

plt.axvline(2.e13, ls='dashed', color='black')
plt.text(1.e11, 1.e-13, 'Space', ha='center')
plt.text(1.e18, 1.e-1, 'Ground', ha='center')

setup_ca(xmin=5.e8, xmax=1.e21, xlabel='Kinetic energy [eV]', ymin=1.e-30, ymax=1.e6,
    ylabel='Intensity [m$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$]',
    logx=True, logy=True)

# Add the top axis.
proton_mass = 1.e9
xmin, xmax = plt.gca().get_xlim()
ax2 = plt.gca().twiny()
ax2.set_xscale('log')
ax2.set_xlabel('Center of mass energy [eV]')
Ecm = np.logspace(9, 15, 7)
E = Ecm**2. / 2. / proton_mass
labels = [f'$10^{{{int(x)}}}$' for x in np.log10(Ecm)]
ax2.set_xticks(E)
ax2.set_xticklabels(labels)
# For some reason that I don't fully understand, this has to be put here.
ax2.set_xlim(xmin, xmax)

save_cf()


margin_figure('cr_integral_flux', height=2.5)
x = np.geomspace(1e9, 1e22, 100)
fov = np.pi
y = fov * fit_norm / (index - 1) * (x / enorm)**(1. - fit_index)
plt.plot(x, y, color='k')
fmt = dict(ls='dashed', color='k')
y0 = 1.
plt.axhline(y0, **fmt)
plt.text(1.e14, y0 * 0.5, '1 m$^{-2}$ s$^{-1}$', size='small', va='top')
y0 /= (3600. * 24 * 365.25)
plt.axhline(y0, **fmt)
plt.text(1.e9, y0 * 0.5, '1 m$^{-2}$ yr$^{-1}$', size='small', va='top')
y0 /= 1.e6
plt.axhline(y0, **fmt)
plt.text(1.e12, y0 * 0.5, '1 km$^{-2}$ yr$^{-1}$', size='small', va='top')
setup_ca(xmin=5.e8, xmax=1.e21, xticks=np.geomspace(1e9, 1e21, 5), logx=True, logy=True,
    xlabel='Kinetic energy [eV]', ylabel='Integral flux [m$^{-2}$ s$^{-1}$]')
save_cf()


standard_figure('cr_composition', height=3., bottom=0.15, top=0.99, right=0.95)
ss_composition = crdb.solar_system_composition()
ss_abundancies = {key: sum(item[1] for item in val) for key, val in ss_composition.items()}
zmax = len(GCR_ABUNDANCE)
symbols = list(ss_abundancies.keys())[:zmax]
z = np.arange(1, zmax + 1)
y = np.array(list(ss_abundancies.values()))[:zmax]
plt.plot(z, y, color='k', ls='dashed', label='Solar system')
plt.plot(z, GCR_ABUNDANCE, 'o', ms=4., color='k', label='Cosmic rays')
vmul = 4.
for i, symbol, abundance in zip(z, symbols, GCR_ABUNDANCE):
    fmt = dict(ha='center', size='small')
    if i % 2 == 0:
        fmt['va'] = 'bottom'
        y = abundance * vmul
    else:
        fmt['va'] = 'top'
        y = abundance / vmul
    plt.text(i, y, symbol, **fmt)
setup_ca(logy=True, xmin=0., xmax=44., xlabel='Atomic number', legend=True,
    ylabel='Relative abundance (Si = $10^6$)')
save_cf()


margin_figure('cr_bc_ratio', height=2.5)
ams_bc = db['AMS02 (2011/05-2018/05) B/C']
ams_bc.plot(marker='o', color='k', ls='none', ms=2.)
setup_ca(logx=True, logy=True, xlabel='Rigidity [GV]', ylabel='B/C ratio',
    xmin=2., xmax=3.e3, xticks=(10., 100., 1000.))
save_cf()


standard_figure('cr_others', height=3., bottom=0.15, top=0.99, right=0.95)
ams_p = db['AMS02 (2011/05-2013/11) H'].rigidity_to_kinetic_energy(1., mp)
ams_ele = db['AMS02 (2011/05-2017/11) e-']
ams_pos = db['AMS02 (2011/05-2017/11) e+']
ams_pbar = db['AMS02 (2011/05-2015/05) 1H-bar'].rigidity_to_kinetic_energy(1., mp)
fmt = dict(color='k', ls='none', ms=3.)
ams_p.plot(marker='s', label='$p$', **fmt)
ams_ele.plot(marker='o', label='$e^-$', **fmt)
ams_pos.plot(marker='o', label='$e^+$', fillstyle='none', **fmt)
ams_pbar.plot(marker='s', label='$\\overline{p}$', fillstyle='none', **fmt)
setup_ca(logx=True, logy=True, xlabel='Kinetic energy [GeV]', legend=True,
    ylabel='Intensity [m$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$]')
save_cf()


plt.show()
