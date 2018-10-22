import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import cartopy.crs as ccrs
from datetime import date, timedelta
from matplotlib.colors import LinearSegmentedColormap

plt.rc('font', **{'family': 'serif', 'sans-serif': ['Helvetica'], 'size': 8})

DEFAULT_WIDTH = 16 / 2.54  # default figure width: 25cm


def plot_timeseries(time, X, Y, Z, **kwargs):

    defaults = dict(figsize=(DEFAULT_WIDTH, 0.8*DEFAULT_WIDTH),
                    titles=['', '', ''],
                    label='')

    # overwrite value with the one in kwargs, if not then use the default
    kwds = defaults
    for key, value in kwargs.items():
        if value is not None:
            kwds[key] = value

    # remove keywords that are not intended for pcolormesh
    figsize = kwds.pop('figsize')
    titles = kwds.pop('titles')
    label = kwds.pop('label')

    date_time = np.array(  # generate list of datetime objects
        [timedelta(days=dt) + date(2000, 1, 1) for dt in time])

    fig, axes = plt.subplots(3, 1, sharex='col', figsize=figsize)
    for ax, component, title in zip(axes, [X, Y, Z], titles):
        ax.plot(date_time, component, **kwds)
        ax.set_title(title)
        ax.grid()
        ax.set(ylabel=label)
        fig.autofmt_xdate()
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    plt.show()


def plot_maps(theta_grid, phi_grid, X, Y, Z, **kwargs):

    defaults = dict(figsize=(DEFAULT_WIDTH, 1.2*DEFAULT_WIDTH),
                    titles=['', '', ''],
                    label='',
                    cmap='PuOr',
                    limiter=lambda x: np.amax(np.abs(x)),  # maximum value
                    projection=ccrs.Mollweide(),
                    transform=ccrs.PlateCarree())

    # overwrite value with the one in kwargs, if not then use the default
    kwds = defaults
    for key, value in kwargs.items():
        if value is not None:
            kwds[key] = value

    # remove keywords that are not intended for pcolormesh
    figsize = kwds.pop('figsize')
    titles = kwds.pop('titles')
    label = kwds.pop('label')
    limiter = kwds.pop('limiter')
    projection = kwds.pop('projection')

    # create axis handle
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=figsize,
                             subplot_kw=dict(projection=projection))
    # make subplots
    for ax, component, title in zip(axes, [X, Y, Z], titles):
        # evaluate colorbar limits depending on vmax/vmin in kwargs
        kwds.setdefault('vmax', limiter(component))
        kwds.setdefault('vmin', -limiter(component))

        # produce colormesh and evaluate keywords (defaults and input)
        pc = ax.pcolormesh(phi_grid, 90. - theta_grid, component, **kwds)

        ax.gridlines(linewidth=0.5, linestyle='dashed',
                     ylocs=np.linspace(-90, 90, num=7),  # parallels
                     xlocs=np.linspace(-180, 180, num=13))  # meridians

        ax.coastlines(linewidth=0.5)

        clb = plt.colorbar(pc, ax=ax, format=ticker.FuncFormatter(fmt),
                           extend='both')

        clb.set_label(label)
        ax.set_global()
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def fmt(x, pos):
    # format=ticker.FuncFormatter(fmt)
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)

    if a == '0.0':
        return r'${}$'.format(a)
    else:
        return r'${}$e${}$'.format(a, b)


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
