"""
The module provides functions for loading and writing data and models. It also
offers functions to do simple time conversions.

.. autosummary::
    :toctree: functions

    load_matfile
    load_RC_datfile
    save_RC_h5file
    load_shcfile
    save_shcfile
    mjd2000
    dyear_to_mjd
    mjd_to_dyear
    memory_usage
    gauss_units

"""


import pandas as pd
import numpy as np
import hdf5storage as hdf
import warnings
import h5py
import os
import calendar
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.dirname(__file__))


def load_matfile(filepath, variable_names=None):
    """
    Load mat-file and return dictionary.

    Function loads mat-file by traversing the structure converting data into
    low-level numpy arrays of different types. There is no guarantee that any
    kind of data is read in correctly. The data dtype can also vary depending
    on the mat-file (v7.3 returns floats instead of integers). But it should
    work identically for v7.3 and prior mat-files. Arrays are squeezed if
    possible.

    Parameters
    ----------
    filepath : str
        Filepath and name of mat-file.
    variable_names : list of strings
        Top-level variables to be loaded.

    Returns
    -------
    data : dict
        Dictionary containing the data as dictionaries or numpy arrays.

    """

    # define a recursively called function to traverse structure
    def traverse_struct(struct):

        # for dictionaries, iterate through keys
        if isinstance(struct, dict):
            out = dict()
            for key, value in struct.items():
                out[key] = traverse_struct(value)
            return out

        # for ndarray, iterate through dtype names
        elif isinstance(struct, np.ndarray):

            # collect dtype names if available
            names = struct.dtype.names

            # if no fields in array
            if names is None:
                if struct.dtype == np.dtype('O') and struct.shape == (1, 1):
                    return traverse_struct(struct[0, 0])
                else:
                    return struct.squeeze()

            else:  # if there are fields, iterate through fields
                out = dict()
                for name in names:
                    out[name] = traverse_struct(struct[name])
                return out

        else:
            return struct

    output = hdf.loadmat(filepath, variable_names=variable_names)

    # loadmat returns dictionary, go through keys and call traverse_struct
    for key, value in output.items():
        if key.startswith('__') and key.endswith('__'):
            pass
        else:
            output[key] = traverse_struct(value)

    return output


def load_RC_datfile(filepath=None, parse_dates=None):
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

    parse_dates = False if parse_dates is None else parse_dates

    # set datetime as index
    if parse_dates:
        df.index = pd.to_datetime(
            df['time'].values, unit='D', origin=pd.Timestamp('2000-1-1'))
        df.drop(['time'], axis=1, inplace=True)  # delete redundant time column

    return df


def save_RC_h5file(filepath, read_from=None):
    """
    Return hdf-file of the RC index.

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

        mjd = dyear_to_mjd(time, leap_year=leap_year)

    return mjd, coeffs, parameters


def save_shcfile(time, coeffs, order=None, filepath=None, nmin=None, nmax=None,
                 leap_year=None, header=None):
    """
    Save Gauss coefficients as shc-file.

    Parameters
    ----------
    time : float, list, ndarray, shape (n,)
        Time of model coeffcients in modified Julian date.
    coeffs : ndarray, shape (N,) or (n, N)
        Gauss coefficients as vector or array. The first dimension of the array
        must be equal to the length `n` of the given ``time``.
    order : int, optional (defaults to 1)
        Order of the piecewise polynomial with which the coefficients are
        parameterized in time (breaks are given by ``time[::order]``).
    filepath : str, optional
        Filepath and name of the output file. Defaults to the current working
        directory and filename `model.shc`.
    nmin : int, optional
        Minimum spherical harmonic degree (defaults to 1). This will remove
        first values from coeffs if greater than 1.
    nmax : int, optional
        Maximum spherical harmonic degree (defaults to degree compatible with
        number of coeffcients, otherwise coeffcients are truncated).
    leap_year : {True, False}, optional
        Take leap years for decimal year conversion into account
        (defaults to ``True``).
    header : str, optional
        Optional header at beginning of file. Defaults to empty string.

    """

    time = np.array(time, dtype=np.float)

    order = 1 if order is None else int(order)

    nmin = 1 if nmin is None else int(nmin)

    if nmax is None:
        nmax = int(np.sqrt(coeffs.shape[-1] + 1) - 1)
    else:
        nmax = int(nmax)

    assert (nmin <= nmax), \
        '``nmin`` must be smaller than or equal to ``nmax``.'

    if filepath is None:
        filepath = 'model.shc'

    header = '' if header is None else header

    if coeffs.ndim == 1:
        coeffs = coeffs.reshape((1, -1))

    coeffs = coeffs[:, (nmin**2-1):((nmax+1)**2-1)]

    # compute all possible degree and orders
    deg = np.array([], dtype=np.int)
    ord = np.array([], dtype=np.int)
    for n in range(nmin, nmax+1):
        deg = np.append(deg, np.repeat(n, 2*n+1))
        ord = np.append(ord, [0])
        for m in range(1, n+1):
            ord = np.append(ord, [m, -m])

    comment = (header +
               f"# Created on {datetime.utcnow()} UTC.\n"
               f"# Leap years are accounted for in "
               f"decimal years format ({leap_year}).\n"
               f"{nmin} {nmax} {time.size} {order} {order-1}\n")

    with open(filepath, 'w') as f:
        # write comment line
        f.write(comment)

        # write header lines to 8 significants
        f.write('  ')  # to represent two missing values
        for t in time:
            f.write(' {:9.4f}'.format(
                mjd_to_dyear(t, leap_year=leap_year)))
        f.write('\n')

        # write coefficient table to 8 significants
        for row, (n, m) in enumerate(zip(deg, ord)):

            f.write('{:} {:}'.format(n, m))

            for value in coeffs[:, row]:
                f.write(' {:.8e}'.format(value))

            f.write('\n')

    print('Coefficients saved to {}.'.format(
        os.path.join(os.getcwd(), filepath)))


@np.vectorize
def mjd2000(*args, **kwargs):
    """
    Computes the modified Julian date as floating point number.

    It assigns 0 to 0h00 January 1, 2000. Leap seconds are not accounted for.

    Parameters
    ----------
    time : :class:`datetime.datetime`, ndarray, shape (...)
        Datetime class instance, `OR ...`
    year : int, ndarray, shape (...)
    month : int, ndarray, shape (...)
        Month of the year `[1, 12]`.
    day : int, ndarray, shape (...)
        Day of the corresponding month.
    hour : int , ndarray, shape (...), optional
        Hour of the day `[0, 23]` (default is 0).
    minute : int, ndarray, shape (...), optional
        Minutes of the hour `[0, 59]` (default is 0).
    second : int, ndarray, shape (...), optional
        Seconds of the minute `[0, 59]` (default is 0).

    Returns
    -------
    time : ndarray, shape (...)
        Modified Julian date (units of days).

    """

    if isinstance(args[0], datetime):
        time = args[0]
    else:
        time = datetime(*args, **kwargs)

    delta = (time - datetime(2000, 1, 1))  # starting 0h00 January 1, 2000

    return delta.days + delta.seconds/86400


@np.vectorize
def dyear_to_mjd(time, leap_year=None):
    """
    Convert time from decimal years to modified Julian date 2000.

    Leap years are accounted for by default.

    Parameters
    ----------
    time : float, ndarray, shape (...)
        Time in decimal years.
    leap_year : {True, False}, optional
        Take leap years into account by using a conversion factor of 365 or 366
        days in a year (leap year, used by default). If ``False`` a conversion
        factor of 365.25 days in a year is used.

    Returns
    -------
    time : ndarray, shape (...)
        Time in modified Julian date 2000.

    """

    leap_year = True if leap_year is None else leap_year

    # remainder is zero = leap year
    if leap_year:
        year = int(time)
        days = 366 if calendar.isleap(year) else 365
        day = (time - year) * days
        date = timedelta(days=day) + datetime(year, 1, 1)

        delta = date - datetime(2000, 1, 1)

        mjd = delta.days + (delta.seconds + delta.microseconds/1e6)/86400

    elif not leap_year:
        days = 365.25

        mjd = (time - 2000.0) * days

    else:
        raise ValueError('Unknown leap year option: use either True or False')

    return mjd


@np.vectorize
def mjd_to_dyear(time, leap_year=None):
    """
    Convert time in modified Julian date 2000 to decimal years.

    Leap years are accounted for by default.

    Parameters
    ----------
    time : float, ndarray, shape (...)
        Time in modified Julian date 2000.
    leap_year : {True, False}, optional
        Take leap years into account by using a conversion factor of 365 or 366
        days in a year (leap year, used by default). If ``False`` a conversion
        factor of 365.25 days in a year is used.

    Returns
    -------
    time : ndarray, shape (...)
        Time in decimal years.

    """

    leap_year = True if leap_year is None else leap_year

    # remainder is zero = leap year
    if leap_year:
        date = timedelta(days=time) + datetime(2000, 1, 1)
        days = 366 if calendar.isleap(date.year) else 365

        delta = date - datetime(date.year, 1, 1)

        dyear = (date.year + delta.days/days
                 + (delta.seconds + delta.microseconds/1e6)/86400/days)

    elif not leap_year:
        days = 365.25

        dyear = time/days + 2000.0

    else:
        print(leap_year)
        raise ValueError('Unknown leap year option: use either True or False')

    return dyear


def memory_usage(pandas_obj):
    """
    Compute memory usage of pandas object.

    For full report, use: ``df.info(memory_usage='deep')``.

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
