import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import cartopy.crs as ccrs
from chaosmagpy.config_utils import basicConfig
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap


def plot_timeseries(time, *args, **kwargs):
    """
    Returns a line plot showing the timeseries of the input arguments.

    Parameters
    ----------
    time : ndarray, shape (N,)
        Array containing time in modified Julian dates.
    *args : ndarray, shape (N, k)
        Array containing `k` columns of values to plot against time. Several
        arrays can be provided as separated arguments.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Matplotlib figure.
    axes : :class:`matplotlib.axes.Axes`, ndarray
        Array of which singleton dimenions have been squeezed out. Only the
        axes instance is returned in case of a single axis.

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

    """

    n = len(args)  # number of subplots

    defaults = dict(figsize=(basicConfig['plots.figure_width'],
                             n*0.8*basicConfig['plots.figure_width']),
                    titles=n*[''],
                    ylabel='',
                    layout=(n, 1))

    kwargs = defaultkeys(defaults, kwargs)

    # remove keywords that are not intended for plot
    figsize = kwargs.pop('figsize')
    layout = kwargs.pop('layout')
    titles = kwargs.pop('titles')
    ylabel = kwargs.pop('ylabel')

    if layout[0]*layout[1] != n:
        raise ValueError('Plot layout is not compatible with the number of '
                         'produced subplots.')

    date_time = np.array(  # generate list of datetime objects
        [timedelta(days=dt) + datetime(2000, 1, 1) for dt in np.ravel(time)])

    fig, axes = plt.subplots(layout[0], layout[1], sharex='all',
                             figsize=figsize, squeeze=False)

    for ax, component, title in zip(axes.flat, args, titles):
        ax.plot(date_time, component, **kwargs)
        ax.set_title(title)
        ax.grid()
        ax.set(ylabel=ylabel, xlabel='time')
        fig.autofmt_xdate()
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d-%h')

    fig.tight_layout(rect=(0, 0.02, 1, 1))

    if axes.shape == (1, 1):
        return fig, axes[0, 0]
    else:
        return fig, np.squeeze(axes)


def plot_maps(theta_grid, phi_grid, *args, **kwargs):
    """
    Returns a global map showing the input arguments.

    Parameters
    ----------
    theta_grid : ndarray
        Array containing the colatitude in degrees.
    phi_grid : ndarray
        Array containing the longitude in degrees.
    *args : ndarray
        Array of values to plot on the global map. Several
        arrays can be provided as separated arguments.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Matplotlib figure.
    axes : :class:`matplotlib.axes.Axes`, ndarray
        Array of which singleton dimenions have been squeezed out. Only the
        axes instance is returned in case of a single axis.

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
        Colormap code (defaults to ``'PuOr'`` colormap).
    limiter : function, lambda expression
        Function to compute symmetric colorbar limits (defaults to maximum of
        the absolute values in the input, or use ``'vmin'``, ``'vmax'``
        keywords instead).
    projection : :mod:`cartopy.crs`
        Projection of the target frame (defaults to
        :func:`cartopy.crs.Mollweide()`).
    transform : :mod:`cartopy.crs`
        Projection of input frame (defaults to
        :func:`cartopy.crs.PlateCarree()`)
    **kwargs : keywords
        Other options to pass to matplotlib :func:`pcolormesh` method.

    """

    n = len(args)  # number of plots

    defaults = dict(figsize=(basicConfig['plots.figure_width'],
                             n*0.4*basicConfig['plots.figure_width']),
                    titles=n*[''],
                    label='',
                    layout=(n, 1),
                    cmap='PuOr',
                    limiter=lambda x: np.amax(np.abs(x)),  # maximum value
                    projection=ccrs.Mollweide(),
                    transform=ccrs.PlateCarree())

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
    # create axis handle
    fig, axes = plt.subplots(layout[0], layout[1], sharex=True, sharey=True,
                             subplot_kw=dict(projection=projection),
                             figsize=figsize, squeeze=False)

    # make subplots
    for ax, component, title in zip(axes.flat, args, titles):
        # evaluate colorbar limits depending on vmax/vmin in kwargs
        kwargs.setdefault('vmax', limiter(component))
        kwargs.setdefault('vmin', -limiter(component))

        # produce colormesh and evaluate keywords (defaults and input)
        pc = ax.pcolormesh(phi_grid, 90. - theta_grid, component, **kwargs)

        ax.gridlines(linewidth=0.5, linestyle='dashed',
                     ylocs=np.linspace(-90, 90, num=7),  # parallels
                     xlocs=np.linspace(-180, 180, num=13))  # meridians

        ax.coastlines(linewidth=0.5)

        clb = plt.colorbar(pc, ax=ax, format=ticker.FuncFormatter(fmt),
                           extend='both')

        clb.set_label(label)
        ax.set_global()
        ax.set_title(title)

    fig.tight_layout()

    if axes.shape == (1, 1):
        return fig, axes[0, 0]
    else:
        return fig, np.squeeze(axes)


def plot_power_spectrum(spectrum, **kwargs):
    """
    Plot spherical harmonic spectrum.

    Parameters
    ----------
    spectrum : ndarray, shape (N,)
        Spherical harmonics spectrum of degree `N`.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Matplotlib figure.
    axes : :class:`matplotlib.axes.Axes`
        A single axes instance.

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

    """

    defaults = dict(figsize=(basicConfig['plots.figure_width'],
                             0.8*basicConfig['plots.figure_width']),
                    titles='',
                    ylabel='')

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
    ax.grid(b=True, which='minor', linestyle=':')
    ax.grid(b=True, which='major', linestyle='-', axis='both')
    ax.set(ylabel=ylabel, xlabel='degree')

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlim((0, degrees[-1]))
    plt.tight_layout()

    return fig, ax


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

    """

    cdict = {'red':   ((0.0000, 0.2500, 0.2500),
                       (0.2222, 0.0000, 0.0000),
                       (0.3968, 0.1000, 0.1000),
                       (0.4921, 1.0000, 1.0000),
                       (1.0000, 1.0000, 1.0000)),

             'green': ((0.0000, 0.0000, 0.0000),
                       (0.2222, 0.0000, 0.0000),
                       (0.3968, 1.0000, 1.0000),
                       (0.5873, 1.0000, 1.0000),
                       (0.8254, 0.0000, 0.0000),
                       (1.0000, 0.0000, 0.0000)),

             'blue':  ((0.0000, 0.2510, 0.2510),
                       (0.2222, 1.0000, 1.0000),
                       (0.4921, 1.0000, 1.0000),
                       (0.5873, 0.0000, 0.0000),
                       (0.8254, 0.0000, 0.0000),
                       (1.0000, 1.0000, 1.0000))}

    return LinearSegmentedColormap('nio', cdict)


# register cmap name for convenient use
plt.register_cmap(cmap=nio_colormap())


if __name__ == '__main__':
    pass
