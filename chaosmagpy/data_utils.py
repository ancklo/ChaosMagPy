import pandas as pd
import numpy as np
import hdf5storage as hdf
import warnings
import h5py
import os
import calendar
from datetime import timedelta, datetime

ROOT = os.path.abspath(os.path.dirname(__file__))


def load_matfile(filepath, variable_name, struct=False):
    """
    Load variable from matfile. Can handle mat-files v7.3 and before.

    Parameters
    ----------
    filepath : str
        Filepath to mat-file.
    variable_name : str
        Name of variable or struct to be loaded from mat-file.
    struct : {False, True}, optional
        If struct is to be loaded from mat-file, use ``True`` (only required
        before v7.3 mat-files).

    Returns
    -------
    variable : ndarray, dict
        Array or dictionary (if ``struct=True``) containing the values.

    """

    mat_contents = hdf.loadmat(str(filepath),
                               variable_names=[str(variable_name)])

    variable = mat_contents[str(variable_name)]

    if struct is True and '__header__' in mat_contents:
        return variable[0, 0]  # version 5 only seems to have header
    else:
        return variable


def load_RC_datfile(filepath=None, parse_dates=False):
    """
    Load RC-index data file into pandas data frame.

    Parameters
    ----------
    filepath : str, optional
        Filepath to RC index ``*.dat``. If ``None``, the RC
        index will be fetched from `spacecenter.dk <http://www.spacecenter.dk/\
        files/magnetic-models/RC/current/>`_.
    parse_dates : bool, optional
        Replace index with datetime object for time-series manipulations.
        Default is ``False``.

    Returns
    -------
    df : dataframe
        Pandas dataframe with names {'time', 'RC', 'RC_e', 'RC_i', 'flag'},
        where ``'time'`` is given in modified Julian dates.

    """

    if filepath is None:
        from lxml import html
        import requests

        link = "http://www.spacecenter.dk/files/magnetic-models/RC/current/"

        page = requests.get(link)
        print(f'Accessing {page.url}.')

        tree = html.fromstring(page.content)
        file = tree.xpath('//tr[5]//td[2]//a/text()')[0]  # get name from list
        date = tree.xpath('//tr[5]//td[3]/text()')[0]

        print(f'Downloading RC-index file "{file}" '
              f'(last modified on {date.strip()}).')

        filepath = link + file

    column_names = ['time', 'RC', 'RC_e', 'RC_i', 'flag']
    column_types = {'time': 'float64', 'RC': 'float64', 'RC_e': 'float64',
                    'RC_i': 'float64', 'flag': 'category'}

    df = pd.read_csv(filepath,  delim_whitespace=True, comment='#',
                     dtype=column_types, names=column_names)

    # set datetime as index
    if parse_dates is True:
        df.index = pd.to_datetime(
            df['time'].values, unit='D', origin=pd.Timestamp('2000-1-1'))
        df.drop(['time'], axis=1, inplace=True)  # delete redundant time column

    return df


def save_RC_h5file(filepath, read_from=None):
    """
    Return h5-file of the RC index.

    Parameters
    ----------
    filepath : str
        Filepath and name of ``*.h5`` output file.
    read_from : str, optional
        Filepath of RC index ``*.dat``. If ``None``, the RC
        index will be fetched from `spacecenter.dk <http://www.spacecenter.dk/\
        files/magnetic-models/RC/current/>`_.

    Notes
    -----
    Saves an h5-file of the RC index with keywords
    ['time', 'RC', 'RC_e', 'RC_i', 'flag']. Time is given in modified Julian
    dates 2000.

    """

    try:
        df_rc = load_RC_datfile(read_from, parse_dates=False)

        with h5py.File(filepath, 'w') as f:

            for column in df_rc.columns:
                variable = df_rc[column].values
                if column == 'flag':
                    dset = f.create_dataset(column, variable.shape, dtype="S1")
                    dset[:] = variable.astype('bytes')

                else:
                    f.create_dataset(column, data=variable)  # just save floats

            print(f'Successfully saved to {f.filename}.')

    except Exception as err:
        warnings.warn(f"Can't save new RC index. Raised exception: '{err}'.")


def load_shcfile(filepath, leap_year=None):
    """
    Load shc-file and return coefficient arrays.

    Parameters
    ----------
    filepath : str
        File path to spherical harmonic coefficient shc-file.
    leap_year : {True, False}, optional
        Take leap year in time conversion into account (default). Otherwise,
        use conversion factor of 365.25 days per year.

    Returns
    -------
    time : ndarray, shape (N,)
        Array containing `N` times for each model snapshot in modified
        Julian dates with origin January 1, 2000 0:00 UTC.
    coeffs : ndarray, shape (nmax(nmax+2), N)
        Coefficients of model snapshots. Each column is a snapshot up to
        spherical degree and order `nmax`.
    parameters : dict, {'SHC', 'nmin', 'nmax', 'N', 'order', 'step'}
        Dictionary containing parameters of the model snapshots and the
        following keys: ``'SHC'`` shc-file name, `nmin` minimum degree,
        ``'nmax'`` maximum degree, ``'N'`` number of snapshot models,
        ``'order'`` piecewise polynomial order and ``'step'`` number of
        snapshots until next break point. Extract break points of the
        piecewise polynomial with ``breaks = time[::step]``.

    """
    leap_year = True if leap_year is None else leap_year

    with open(filepath, 'r') as f:

        data = np.array([])
        for line in f.readlines():

            if line[0] == '#':
                continue

            read_line = np.fromstring(line, sep=' ')
            if read_line.size == 5:
                name = os.path.split(filepath)[1]  # file name string
                values = [name] + read_line.astype(np.int).tolist()

            else:
                data = np.append(data, read_line)

        # unpack parameter line
        keys = ['SHC', 'nmin', 'nmax', 'N', 'order', 'step']
        parameters = dict(zip(keys, values))

        time = data[:parameters['N']]
        coeffs = data[parameters['N']:].reshape((-1, parameters['N']+2))
        coeffs = np.squeeze(coeffs[:, 2:])  # discard columns with n and m

        mjd = np.array([dyear_to_mjd(t, leap_year=leap_year) for t in time])

    return mjd, coeffs, parameters


def mjd2000(*args, **kwargs):
    """
    Computes the modified Julian date as floating point number. It assigns 0 to
    0h00 January 1, 2000. Leap seconds are not accounted for.

    Parameters
    ----------
    time : :class:`datetime.datetime`
        Datetime class instance, `OR ...`
    year : int
    month : int
        Month of the year `[1, 12]`.
    day : int
        Day of the corresponding month.
    hour : int, optional
        Hour of the day `[0, 23]` (default is 0).
    minute : int, optional
        Minutes of the hour `[0, 59]` (default is 0).
    second : int, optional
        Seconds of the minute `[0, 59]` (default is 0).

    Returns
    -------
    time : float
        Modified Julian date (units of days).

    """

    if isinstance(args[0], datetime):
        time = args[0]
    else:
        time = datetime(*args, **kwargs)

    delta = (time - datetime(2000, 1, 1))  # starting 0h00 January 1, 2000

    return delta.days + delta.seconds/86400


def dyear_to_mjd(time, leap_year=None):
    """
    Convert time from decimal years to modified Julian date 2000. Leap years
    are accounted for by default.

    Parameters
    ----------
    time : float
        Time in decimal years.
    leap_year : {True, False}, optional
        Take leap years into account by using a conversion factor of 365 or 366
        days in a year (leap year, used by default). If ``False`` a conversion
        factor of 365.25 days in a year is used.

    Returns
    -------
    time : float
        Time in modified Julian date 2000.
    """

    leap_year = True if leap_year is None else leap_year

    # remainder is zero = leap year
    if leap_year is True:
        year = int(time)
        days = 366 if calendar.isleap(year) else 365
        day = (time - year) * days
        date = timedelta(days=day) + datetime(year, 1, 1)

        delta = date - datetime(2000, 1, 1)

        mjd = delta.days + (delta.seconds + delta.microseconds/1e6)/86400

    elif leap_year is False:
        days = 365.25

        mjd = (time - 2000.0) * days

    else:
        raise ValueError('Unknown leap year option: use either True or False')

    return mjd


def mjd_to_dyear(time, leap_year=None):
    """
    Convert time in modified Julian date 2000 to decimal years. Leap years are
    accounted for by default.

    Parameters
    ----------
    time : float
        Time in modified Julian date 2000.
    leap_year : {True, False}, optional
        Take leap years into account by using a conversion factor of 365 or 366
        days in a year (leap year, used by default). If ``False`` a conversion
        factor of 365.25 days in a year is used.

    Returns
    -------
    time : float
        Time in decimal years.
    """

    leap_year = True if leap_year is None else leap_year

    # remainder is zero = leap year
    if leap_year is True:
        date = timedelta(days=time) + datetime(2000, 1, 1)
        days = 366 if calendar.isleap(date.year) else 365

        delta = date - datetime(date.year, 1, 1)

        dyear = (date.year + delta.days/days
                 + (delta.seconds + delta.microseconds/1e6)/86400/days)

    elif leap_year is False:
        days = 365.25

        dyear = time/days + 2000.0

    else:
        raise ValueError('Unknown leap year option: use either True or False')

    return dyear


def memory_usage(pandas_obj):
    """
    Compute memory usage of pandas object. For full report, use:
    ``df.info(memory_usage='deep')``.

    """

    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def gauss_units(deriv):
    """
    Return string of the magnetic field units given the derivative with time.
    String is meant to be parsed to plot labels.
    """

    deriv = 0 if deriv is None else deriv

    if deriv == 0:
        units = 'nT'
    else:
        units = '$\\mathrm{{nT}}\\cdot \\mathrm{{yr}}^{{{:}}}$'.format(-deriv)

    return units


def rsme(x, y):
    """
    Compute RSME (root square mean error) of inputs x and y.

    Parameters
    ----------
    x, y : ndarray
    """

    x = np.array(x)
    y = np.array(y)

    return np.mean(np.abs(x-y)**2)**0.5
