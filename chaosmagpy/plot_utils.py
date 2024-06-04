# Copyright (C) 2024 Clemens Kloss
#
# This file is part of ChaosMagPy.
#
# ChaosMagPy is released under the MIT license. See LICENSE in the root of the
# repository for full licensing details.

"""
`chaosmagpy.plot_utils` provides functions for plotting model outputs.

.. autosummary::
    :toctree: functions

    plot_timeseries
    plot_maps
    plot_power_spectrum
    nio_colormap

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import shapefile
from matplotlib.colors import LinearSegmentedColormap
from . import data_utils
from . import config_utils


def plot_timeseries(time, *args, **kwargs):
    """
    Creates line plots for the timeseries of the input arguments.

    Parameters
    ----------
    time : ndarray, shape (N,)
        Array containing time in modified Julian dates.
    *args : ndarray, shape (N, k)
        Array containing `k` columns of values to plot against time. Several
        arrays can be provided as separated arguments.

    Other Parameters
    ----------------
    figsize : 2-tuple of floats
        Figure dimension (width, height) in inches.
    titles : list of strings
        Subplot titles (defaults to empty strings).
    ylabel : string
        Label of the vertical axis (defaults to an empty string).
    layout : 2-tuple of int
        Layout of the subplots (defaults to vertically stacked subplots).
    **kwargs : keywords
        Other options to pass to matplotlib plotting method.

    Notes
    -----
    For more customization get access to the figure and axes handles
    through matplotlib by using ``fig = plt.gcf()`` and ``axes = fig.axes``
    right after the call to this plotting function.

    """

    n = len(args)  # number of subplots

    defaults = {
        'figsize': (config_utils.basicConfig['plots.figure_width'],
                    0.8*n*config_utils.basicConfig['plots.figure_width']),
        'titles': n*[''],
        'ylabel': '',
        'layout': (n, 1)
    }

    kwargs = defaultkeys(defaults, kwargs)

    # remove keywords that are not intended for plot
    figsize = kwargs.pop('figsize')
    layout = kwargs.pop('layout')
    titles = kwargs.pop('titles')
    ylabel = kwargs.pop('ylabel')

    if layout[0]*layout[1] != n:
        raise ValueError('Plot layout is not compatible with the number of '
                         'produced subplots.')

    date_time = data_utils.timestamp(np.ravel(time))

    fig, axes = plt.subplots(layout[0], layout[1], sharex='all',
                             figsize=figsize, squeeze=False)

    for ax, component, title in zip(axes.flat, args, titles):
        ax.plot(date_time, component, **kwargs)
        ax.set_title(title)
        ax.grid(True)
        ax.set(ylabel=ylabel, xlabel='time')
        fig.autofmt_xdate()
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d-%h')

    fig.tight_layout(rect=(0, 0.02, 1, 1))


def plot_maps(theta_grid, phi_grid, *args, **kwargs):
    """
    Plots global maps of the input arguments.

    Parameters
    ----------
    theta_grid : ndarray
        Array containing the colatitude in degrees.
    phi_grid : ndarray
        Array containing the longitude in degrees.
    *args : ndarray
        Array of values to plot on the global map. Several
        arrays can be provided as separated arguments.

    Other Parameters
    ----------------
    figsize : 2-tuple of floats
        Figure dimension (width, height) in inches.
    titles : list of strings
        Subplot titles (defaults to empty strings).
    label : string
        Colorbar label (defaults to an empty string).
    layout : 2-tuple of int
        Layout of the subplots (defaults to vertically stacked subplots).
    cmap : str
        Colormap code (defaults to ``'PuOr_r'`` colormap).
    limiter : function, lambda expression
        Function to compute symmetric colorbar limits (defaults to maximum of
        the absolute values in the input, or use ``'vmin'``, ``'vmax'``
        keywords instead).
    projection : str
        Projection of the target frame (defaults to `'mollweide'`).
    **kwargs : keywords
        Other options to pass to matplotlib :func:`pcolormesh` method.

    Notes
    -----
    For more customization get access to the figure and axes handles
    through matplotlib by using ``fig = plt.gcf()`` and ``axes = fig.axes``
    right after the call to this plotting function.

    """

    n = len(args)  # number of plots

    defaults = {
        'figsize': (config_utils.basicConfig['plots.figure_width'],
                    0.4*n*config_utils.basicConfig['plots.figure_width']),
        'titles': n*[''],
        'label': '',
        'layout': (n, 1),
        'cmap': 'PuOr_r',
        'limiter': lambda x: np.amax(np.abs(x)),  # maximum value
        'projection': 'mollweide',
        'shading': 'auto'
    }

    kwargs = defaultkeys(defaults, kwargs)

    # remove keywords that are not intended for pcolormesh
    figsize = kwargs.pop('figsize')
    titles = kwargs.pop('titles')
    label = kwargs.pop('label')
    limiter = kwargs.pop('limiter')
    projection = kwargs.pop('projection')
    layout = kwargs.pop('layout')

    if layout[0]*layout[1] != n:
        raise ValueError('Plot layout is not compatible with the number of '
                         'produced subplots.')

    # load shapefile with the coastline
    shp = config_utils.basicConfig['file.shp_coastline']

    # create axis handle
    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize,
                             subplot_kw=dict(projection=projection),
                             squeeze=False)

    # make subplots
    for ax, component, title in zip(axes.flat, args, titles):

        # evaluate colorbar limits depending on vmax/vmin in kwargs
        kwargs.setdefault('vmax', limiter(component))
        kwargs.setdefault('vmin', -limiter(component))

        with shapefile.Reader(shp) as sf:

            for rec in sf.shapeRecords():
                lon = np.radians([point[0] for point in rec.shape.points[:]])
                lat = np.radians([point[1] for point in rec.shape.points[:]])

                ax.plot(lon, lat, color='k', linewidth=0.8)

        # produce colormesh and evaluate keywords (defaults and input)
        pc = ax.pcolormesh(np.radians(phi_grid), np.radians(90. - theta_grid),
                           component, **kwargs)

        plt.colorbar(pc, ax=ax, extend='both', label=label)

        ax.xaxis.set_ticks(np.radians(np.linspace(-180., 180., num=13)))
        ax.yaxis.set_ticks(np.radians(np.linspace(-60., 60., num=5)))
        ax.xaxis.set_major_formatter('')
        ax.grid(True)

        ax.set_title(title)

    fig.tight_layout()


def plot_power_spectrum(spectrum, **kwargs):
    """
    Plot the spherical harmonic spectrum.

    Parameters
    ----------
    spectrum : ndarray, shape (N,)
        Spherical harmonics spectrum of degree `N`.

    Other Parameters
    ----------------
    figsize : 2-tuple of floats
        Figure dimension (width, height) in inches.
    titles : list of strings
        Subplot titles (defaults to empty strings).
    ylabel : string
        Label of the vertical axis (defaults to an empty string).
    **kwargs
        Keywords passed to :func:`matplotlib.pyplot.semilogy`

    Notes
    -----
    For more customization get access to the figure and axes handles
    through matplotlib by using ``fig = plt.gcf()`` and ``axes = fig.axes``
    right after the call to this plotting function.

    """

    defaults = {
        'figsize': (config_utils.basicConfig['plots.figure_width'],
                    0.8*config_utils.basicConfig['plots.figure_width']),
        'titles': '',
        'ylabel': ''
    }

    kwargs = defaultkeys(defaults, kwargs)

    figsize = kwargs.pop('figsize')
    titles = kwargs.pop('titles')
    ylabel = kwargs.pop('ylabel')

    degrees = np.arange(1, spectrum.shape[0] + 1, step=1.0)
    spectrum[spectrum == 0] = np.nan  # remove non-positive values for log

    # create axis handle
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=figsize)

    ax.semilogy(degrees, spectrum, **kwargs)
    ax.set_title(titles)
    ax.grid(True, which='minor', linestyle=':')
    ax.grid(True, which='major', linestyle='-', axis='both')
    ax.set(ylabel=ylabel, xlabel='degree')

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()


def fmt(x, pos):
    # format=ticker.FuncFormatter(fmt)
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)

    if a == '0.0':
        return r'${}$'.format(a)
    else:
        return r'${}$e${}$'.format(a, b)


def defaultkeys(defaults, keywords):
    """
    Return dictionary of default keywords. Overwrite any keywords in
    ``defaults`` using keywords from ``keywords``, except if they are None.

    Parameters
    ----------
    defaults, keywords : dict
        Dictionary of default and replacing keywords.

    Returns
    -------
    keywords : dict

    """

    # overwrite value with the one in kwargs, if not then use the default
    for key, value in defaults.items():
        if keywords.setdefault(key, value) is None:
            keywords[key] = value

    return keywords


def nio_colormap():
    """
    Define custom-built colormap 'nio' and register. Can be called in plots as
    ``cmap='nio'`` after importing this module.

    .. plot::
        :include-source: false

        import numpy as np
        import chaosmagpy as cp
        import matplotlib.pyplot as plt

        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        figh = 0.35 + 0.15 + 0.22
        fig, ax = plt.subplots(1, 1, figsize=(6.4, figh))
        fig.subplots_adjust(top=0.792, bottom=0.208, left=0.023, right=0.977)
        ax.imshow(gradient, aspect='auto', cmap='nio')
        # ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()

    """

    cdict = {'red':   ((0.0000, 0.0000, 0.0000),
                       (0.1667, 0.0000, 0.0000),
                       (0.3333, 0.0000, 0.0000),
                       (0.5020, 1.0000, 1.0000),
                       (0.6667, 1.0000, 1.0000),
                       (0.8333, 1.0000, 1.0000),
                       (1.0000, 1.0000, 1.0000)),

             'green': ((0.0000, 0.0000, 0.0000),
                       (0.1667, 0.0000, 0.0000),
                       (0.3333, 1.0000, 1.0000),
                       (0.5020, 1.0000, 1.0000),
                       (0.6667, 1.0000, 1.0000),
                       (0.8333, 0.0000, 0.0000),
                       (1.0000, 0.0000, 0.0000)),

             'blue':  ((0.0000, 0.0000, 0.0000),
                       (0.1667, 1.0000, 1.0000),
                       (0.3333, 1.0000, 1.0000),
                       (0.5020, 1.0000, 1.0000),  # >0.5 for white (intpl bug?)
                       (0.6667, 0.0000, 0.0000),
                       (0.8333, 0.0000, 0.0000),
                       (1.0000, 1.0000, 1.0000))}

    return LinearSegmentedColormap('nio', cdict)


# register cmap name for convenient use
plt.colormaps.register(cmap=nio_colormap())


if __name__ == '__main__':
    pass
