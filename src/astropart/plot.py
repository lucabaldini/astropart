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

"""Basic plotting style.
"""

from pathlib import Path

from loguru import logger
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc, Ellipse
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

from astropart import ASTROPART_FIGURES


STANDARD_FIGURE_WIDTH = 4.15
DEFAULT_FIGURE_HEIGHT = 3.5
STANDARD_MARGINS = (0.175, 0.15, 0.975, 0.975)
MARGIN_FIGURE_WIDTH = 1.95
FULL_FIGURE_WIDTH = 6.75
THIN_LW = 0.75


def _default_margins(width, height):
    """Calculate the default marging for a given size.

    This is setup in such a way that the physical size of the margins, which is
    necessary to accomodate the axis labels and stuff, is approximately
    independent from the overall size.
    """
    left, bottom, right, top = STANDARD_MARGINS
    hor_scale = STANDARD_FIGURE_WIDTH / width
    ver_scale = DEFAULT_FIGURE_HEIGHT / height
    left *= hor_scale
    right = 1. - (1. - right) * hor_scale
    bottom *= ver_scale
    top = 1. - (1. - top) * ver_scale
    return left, bottom, right, top

def _figure(name, width, height, **kwargs):
    """Basic figure factory.
    """
    default_left, default_bottom, default_right, default_top = _default_margins(width, height)
    left = kwargs.get('left', default_left)
    bottom = kwargs.get('bottom', default_bottom)
    right = kwargs.get('right', default_right)
    top = kwargs.get('top', default_top)
    fig = plt.figure(name, (width, height))
    plt.subplots_adjust(left, bottom, right, top)
    return fig

def subplot_vertical_stack(height_ratios):
    """Create a vertical stack of subplots in the current figure, with all the
    x-axes tied together and the labels on the y-axes horizontally aligned.
    """
    fig = plt.gcf()
    num_rows = len(height_ratios)
    kwargs = dict(height_ratios=height_ratios, hspace=0.05)
    axes = fig.subplots(num_rows, 1, sharex=True, gridspec_kw=kwargs)
    fig.align_ylabels((axes))
    plt.sca(axes[0])
    return axes

def standard_figure(name : str, height : float = DEFAULT_FIGURE_HEIGHT, **kwargs):
    """Standard figure.
    """
    return _figure(name, STANDARD_FIGURE_WIDTH, height, **kwargs)

def margin_figure(name : str, height : float = DEFAULT_FIGURE_HEIGHT, **kwargs):
    """Marging figure.
    """
    return _figure(name, MARGIN_FIGURE_WIDTH, height, **kwargs)

def full_figure(name : str, height : float = DEFAULT_FIGURE_HEIGHT, **kwargs):
    """
    """
    return _figure(name, FULL_FIGURE_WIDTH, height, **kwargs)

def residual_figure(name : str, height : float = DEFAULT_FIGURE_HEIGHT, **kwargs):
    """
    """
    fig = _figure(name, STANDARD_FIGURE_WIDTH, height, **kwargs)
    ax1, ax2 = subplot_vertical_stack((2.5, 1.))
    return fig, ax1, ax2

def setup_ca(**kwargs):
    """Setup the axes for the current plot.
    """
    # Cache the settings...
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    xmin = kwargs.get('xmin', None)
    xmax = kwargs.get('xmax', None)
    ymin = kwargs.get('ymin', None)
    ymax = kwargs.get('ymax', None)
    bb = (xmin, xmax, ymin, ymax)
    logx = kwargs.get('logx', False)
    logy = kwargs.get('logy', False)
    grids = kwargs.get('grids', True)
    xticks = kwargs.get('xticks', None)
    yticks = kwargs.get('yticks', None)
    legend = kwargs.get('legend', False)
    # ... and apply the style to the plot.
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    if xticks is not None:
        plt.gca().set_xticks(xticks)
    if yticks is not None:
        plt.gca().set_yticks(yticks)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if bb != (None, None, None, None):
        plt.axis(bb)
    if grids:
        plt.grid(True, which='both')
    if legend:
        plt.legend()

def setup_ca_drawing(xside: float = 1.05, yside: float = 1.05, equal_aspect: bool = True) -> None:
    """Setup the current figure for hand-free drawing.

    This means: set an 'equal' aspect ratio, set the margins, and remove the axes.
    """
    setup_ca(xmin=-xside, xmax=xside, ymin=-yside, ymax=yside, grids=False)
    if equal_aspect:
        plt.gca().set_aspect('equal')
    plt.axis('off')

def draw_line(p0: tuple[float, float], p1: tuple[float, float], **kwargs):
    """Draw a line.

    Arguments
    ---------
    p0 : tuple
        The starting point (x0, y0)

    p1 : tuple
        The ending point (x1, y1)

    kwargs: dict
        Keyword arguments to be passed to plt.plot()
    """
    kwargs.setdefault('color', 'black')
    x0, y0 = p0
    x1, y1 = p1
    plt.plot((x0, x1), (y0, y1), **kwargs)

def _rotate(x, y, theta, x0=0., y0=0.):
    """Small utility function implementing a two-dimensional rotation
    around a given point.
    """
    # Cache the necessary things, so that we do not calculate them twice...
    dx, dy = (x - x0), (y - y0)
    _sin, _cos = np.sin(theta), np.cos(theta)
    # ...and perform the actual rotation.
    return dx * _cos - dy * _sin, dx * _sin + dy *_cos

def draw_wavy_line(p0: tuple[float, float], p1: tuple[float, float],
    amplitude: float = 0.02, period: float = 0.075, num_points: int = 250, **kwargs):
    """Draw a wavy line.
    """
    kwargs.setdefault('color', 'black')
    # Cache a few useful things.
    x0, y0 = p0
    x1, y1 = p1
    # Create a series of (x, y) points along the line connecting p0 and p1.
    x = np.linspace(x0, x1, num_points)
    y = y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    # Rotate the points around (x0, y0) of an angle theta so that the rotate
    # y coordinate is identically 0 and the rotated x coordinate goes from 0 to
    # the distance between the original points.
    theta = np.arctan2(y1 - y0, x1 - x0)
    xrot, yrot = _rotate(x, y, -theta, x0, y0)
    # Add the wiggle on the y coordinate. Note that tweak the period parameter a
    # little bit so that the wiggle makes an integer number of cycles.
    length = xrot.max()
    omega = 2. * np.pi * round(length / period) / length
    yrot -= amplitude * np.sin(omega * xrot)
    # Rotate back into the original reference, and add p0.
    x, y = _rotate(xrot, yrot, theta)
    x += x0
    y += y0
    plt.plot(x, y, **kwargs)

def draw_arrow(p0, p1, **kwargs):
    """
    """
    arrowprops = dict(arrowstyle='->')
    arrowprops.update(**kwargs)
    plt.gca().annotate("", xy=p1, xytext=p0, arrowprops=arrowprops)

def draw_rectangle(p0, width, height, **kwargs):
    """
    """
    rect = Rectangle(p0, width, height, **kwargs)
    plt.gca().add_patch(rect)

def draw_circle(p0, radius, **kwargs):
    """
    """
    kwargs.setdefault('edgecolor', 'black')
    kwargs.setdefault('facecolor', 'none')
    circ = Circle(p0, radius, **kwargs)
    plt.gca().add_patch(circ)

def draw_ellipse(p0, width, height, angle=0., **kwargs):
    """
    """
    kwargs.setdefault('edgecolor', 'black')
    kwargs.setdefault('facecolor', 'white')
    ellipse = Ellipse(p0, width, height, angle, **kwargs)
    plt.gca().add_patch(ellipse)

def draw_arc(p0, radius, theta1, theta2, **kwargs):
    """
    """
    #kwargs.setdefault('edgecolor', 'black')
    #kwargs.setdefault('facecolor', 'white')
    arc = Arc(p0, 2. * radius, 2. * radius, 0., theta1, theta2, **kwargs)
    plt.gca().add_patch(arc)

def add_yaxis(x : float, ymin : float, ymax : float, **kwargs):
    """Add an orphan y axis for the purpose of illustrating the order of magnitude
    for a given quantity.
    """
    bottom = kwargs.get('bottom', 0.05)
    top = kwargs.get('top', 0.95)
    width = kwargs.get('width', 0.1)
    height = top - bottom
    logy = kwargs.get('logy', False)
    title = kwargs.get('title', None)
    ax = plt.gcf().add_axes((x, bottom, width, height))
    ax.get_xaxis().set_visible(False)
    for spine in ('top', 'right', 'bottom'):
        ax.spines[spine].set_visible(False)
    ax.set_xlim(0., 1.)
    ax.set_ylim(ymin, ymax)
    if logy:
        ax.set_yscale('log')
    if title is not None:
        ax.set_title(title, loc='left', ha='center')
    # Some detailed formatting for the minor ticks.
    ax.tick_params(which='major', length=4)
    if logy:
        ax.yaxis.set_minor_locator(LogLocator(numticks=np.log10(ymax / ymin) + 2))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='minor', length=2)
    return ax

def save_cf(file_name : str = None, file_types : str = ('pgf', 'pdf'),
    folder_path : str = ASTROPART_FIGURES):
    """Save the current matplotlib figure.
    """
    if file_name is None:
        file_name = plt.gcf().get_label().lower().replace(' ', '_')
    if file_name == '':
        raise RuntimeError('Cannot determine file name for figure name.')
    for file_type in file_types:
        file_path = folder_path / f'{file_name}.{file_type}'
        logger.info(f'Saving current figure to {file_path}...')
        try:
            plt.savefig(file_path, transparent=False)
        except IOError as e:
            logger.error(e)

def _srcp(key, value):
    """Set a given RC parameter.
    """
    matplotlib.rcParams[key] = value

def _setup_style():
    """Setup the global style.
    """
    # Figure
    _srcp('figure.figsize', (8., 6.))
    _srcp('figure.dpi', 100)
    _srcp('figure.facecolor', 'white')
    _srcp('figure.edgecolor', 'white')
    _srcp('figure.subplot.left', 0.125)
    _srcp('figure.subplot.right', 0.95)
    _srcp('figure.subplot.bottom', 0.10)
    _srcp('figure.subplot.top', 0.95)
    # Axes
    _srcp('axes.linewidth', 1.)
    _srcp('axes.xmargin', 0.)
    _srcp('axes.ymargin', 0.05)
    # Fonts.
    _srcp('font.size', 9.)
    _srcp('font.family', 'serif')
    # Grids
    _srcp('grid.color', '#c0c0c0')
    _srcp('grid.linestyle', '--')
    _srcp('grid.linewidth', 0.8)
    _srcp('grid.alpha', 1.0)
    #
    _srcp('lines.linewidth', 1.0)


_setup_style()
