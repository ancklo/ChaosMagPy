# Copyright (C) 2024 Clemens Kloss
#
# This file is part of ChaosMagPy.
#
# ChaosMagPy is released under the MIT license. See LICENSE in the root of the
# repository for full licensing details.

"""
`chaosmagpy.chaos` provides classes and functions to read the CHAOS model and
other geomagnetic field models.

.. autosummary::
    :toctree: classes
    :template: myclass.rst

    Base
    BaseModel
    CHAOS

.. autosummary::
    :toctree: functions

    load_CHAOS_matfile
    load_CHAOS_shcfile
    load_CovObs_txtfile
    load_gufm1_txtfile
    load_CALS7K_txtfile
    load_IGRF_txtfile

"""

import numpy as np
import os
import warnings
import scipy.interpolate as sip
import hdf5storage as hdf
import h5py
import textwrap
import datetime
from timeit import default_timer as timer
from . import coordinate_utils as cu
from . import model_utils as mu
from . import data_utils as du
from . import plot_utils as pu
from . import config_utils


class Base(object):
    """
    Piecewise polynomial base class.

    """

    def __init__(self, name, breaks=None, order=None, coeffs=None, meta=None):

        self.name = str(name)

        # ensure breaks is None or has 2 elements
        if breaks is not None:
            breaks = np.asarray(breaks, dtype=float)
            if breaks.size == 1:
                breaks = np.append(breaks, breaks)

        self.breaks = breaks

        self.pieces = None if breaks is None else int(breaks.size - 1)

        if coeffs is None:
            self.coeffs = coeffs
            self.order = None if order is None else int(order)
            self.dim = None

        else:
            coeffs = np.asarray(coeffs, dtype=float)

            if order is None:
                self.order = coeffs.shape[0]
            else:
                self.order = min(int(order), coeffs.shape[0])

            self.coeffs = coeffs[-self.order:]
            self.dim = coeffs.shape[-1]

        self.meta = meta

    def synth_coeffs(self, time, *, dim=None, deriv=None, extrapolate=None):
        """
        Compute the coefficients from the piecewise polynomial representation.

        Parameters
        ----------
        time : ndarray, shape (...) or float
            Array containing the time in modified Julian date.
        dim : int, positive, optional
            Truncation value of the number of coefficients (no truncation by
            default).
        deriv : int, positive, optional
            Derivative in time (None defaults to 0).
        extrapolate : {'linear', 'quadratic', 'cubic', 'spline', 'constant', \
'off'} or int, optional
            Extrapolate to times outside of the piecewise polynomial bounds.
            Specify the polynomial degree as string or the order as an integer.
            Defaults to ``'linear'`` (equiv. to order-2 polynomials).

             +------------+---------------------------------------------------+
             | Value      | Description                                       |
             +============+===================================================+
             | 'constant' | Use degree zero polynomial only (extrapolate=1).  |
             +------------+---------------------------------------------------+
             | 'linear'   | Use degree-1 polynomials (extrapolate=2).         |
             +------------+---------------------------------------------------+
             | 'quadratic'| Use degree-2 polynomials (extrapolate=3).         |
             +------------+---------------------------------------------------+
             | 'cubic'    | Use degree-3 polynomials (extrapolate=4).         |
             +------------+---------------------------------------------------+
             | 'spline'   | Use all degree polynomials.                       |
             +------------+---------------------------------------------------+
             | 'off'      | Return NaN outside model bounds (extrapolate=0).  |
             +------------+---------------------------------------------------+

        Returns
        -------
        coeffs : ndarray, shape (..., ``dim``)
            Array containing the coefficients.

        """

        if (self.coeffs is None) or (self.coeffs.size == 0):
            raise ValueError(f'Coefficients of "{self.name}" are missing.')

        # handle optional argument: dim
        if dim is None:
            dim = self.dim
        elif dim > self.dim:
            warnings.warn(
                'Supplied dim = {0} is incompatible with number of '
                'coefficients. Using dim = {1} instead.'.format(
                    dim, self.dim))
            dim = self.dim

        if deriv is None:
            deriv = 0

        if extrapolate is None:
            extrapolate = 'linear'  # linear extrapolation

        # setting spline interpolation
        PP = sip.PPoly.construct_fast(
            self.coeffs[..., :dim].astype(float),
            self.breaks.astype(float), extrapolate=True)

        start = self.breaks[0]
        end = self.breaks[-1]

        if (np.amin(time) < start) or (np.amax(time) > end):

            if isinstance(extrapolate, (str, bool)):  # convert to integer
                dkey = {
                    'linear': 2,
                    'quadratic': 3,
                    'cubic': 4,
                    'constant': 1,
                    'spline': self.order,
                    'off': 0,
                    True: 2,
                    False: 0
                }
                try:
                    key = min(dkey[extrapolate], self.order)
                except KeyError:
                    string = '", "'.join([str(key) for key in dkey.keys()])
                    raise ValueError(
                        f'Unknown extrapolation method "{extrapolate}". Use '
                        f'one of {{"{string}"}}.')
            else:
                key = min(extrapolate, self.order)

            if key == 0:
                message = 'no'
            else:
                message = f'degree-{key - 1}'

            warnings.warn("Requested coefficients are "
                          "outside of the model time period from "
                          f"{start} to {end} Modified Julian Date 2000. "
                          f"Doing {message} extrapolation.")

            if key > 0:
                for x in [start, end]:  # left and right
                    bin = np.zeros((self.order, 1, dim))
                    for k in range(key):
                        bin[-1-k] = PP(x, nu=k)
                    PP.extend(bin, np.array([x]))

            else:  # no extrapolation
                PP.extrapolate = False

        PP = PP.derivative(nu=deriv)
        coeffs = PP(time) * 365.25**deriv

        return coeffs

    def to_ppdict(self):
        """
        Return a dictionary of the piecewise polynomial that is compatible with
        MATLAB's pp-form.

        Returns
        -------
        pp : dict
            Dictionary of the pp-form compatible with MATLAB. Elements are
            NumPy arrays.

        See Also
        --------
        Base.save_matfile

        """

        pp = dict(
            form='pp',
            order=np.array(self.order, float),
            pieces=np.array(self.pieces, float),
            dim=np.array(self.dim, float),
            breaks=self.breaks.copy().reshape((1, -1)),  # ensure 2d
            coefs=np.reshape(self.coeffs.copy(), (self.order, -1)).transpose()
        )

        return pp

    def save_matfile(self, filepath, path=None):
        """
        Save piecewise polynomial as MAT-file.

        Parameters
        ----------
        filepath : str
            Filepath and name of MAT-file.
        path : str
            Location in MAT-file. Defaults to ``'/pp'``.

        See Also
        --------
        Base.to_ppdict

        """

        if path is None:
            path = '/pp'

        pp = self.to_ppdict()

        hdf.write(pp, path=path, filename=filepath, matlab_compatible=True)


class BaseModel(Base):
    """
    Class for piecewise polynomial spherical harmonic models.

    Parameters
    ----------
    name : str
        User specified name of the model.
    breaks : ndarray, shape (m+1,)
        Break points (`m` pieces plus one) for the piecewise polynomial
        representation of the field model in modified Julian date format.
        If a single break point is given, it is appended to itself to have two
        points defining the model interval (single piece).
    order : int, positive
        Order `k` of the polynomial pieces (e.g. 1 = constant, 4 = cubic).
    coeffs : ndarray, shape (`k`, `m`, ``nmax`` * (``nmax`` + 2))
        Coefficients of the piecewise polynomial representation of the
        field model.
    source : {'internal', 'external'}
        Internal or external source (defaults to ``'internal'``)
    meta : dict, optional
        Dictionary containing additional information about the model if
        available.

    Attributes
    ----------
    breaks : ndarray, shape (m+1,)
        Break points (`m` pieces plus one) for the piecewise polynomial
        representation of the field model in modified Julian date format.
    pieces : int, positive
        Number `m` of intervals given by break points in ``breaks``.
    order : int, positive
        Order `k` of the polynomial pieces (e.g. 1 = constant, 4 = cubic).
    nmax : int, positive
        Maximum spherical harmonic degree of the field model.
    dim : int, ``nmax`` * (``nmax`` + 2)
        Dimension of the model.
    coeffs : ndarray, shape (`k`, `m`, ``nmax`` * (``nmax`` + 2))
        Coefficients of the time-dependent field.
    source : {'internal', 'external'}
        Internal or external source (defaults to ``'internal'``)
    meta : dict, optional
        Dictionary containing additional information about the model if
        available.

    """

    def __init__(self, name, breaks=None, order=None, coeffs=None,
                 source=None, meta=None):
        """
        Initialize spherical harmonic model as a piecewise polynomial.

        """

        super().__init__(name, breaks=breaks, order=order, coeffs=coeffs,
                         meta=meta)

        if self.dim is None:
            self.nmax = None
        else:
            self.nmax = int(np.sqrt(self.dim + 1) - 1)

        self.source = 'internal' if source is None else source

    def synth_coeffs(self, time, *, nmax=None, deriv=None, extrapolate=None):
        """
        Compute the coefficients from the piecewise polynomial representation.

        Parameters
        ----------
        time : ndarray, shape (...) or float
            Array containing the time in modified Julian dates.
        nmax : int, positive, optional
            Maximum degree of the harmonic expansion (default is given by the
            model coefficients, but can also be smaller, if specified).
        deriv : int, positive, optional
            Derivative in time (defaults to 0). For secular variation,
            choose ``deriv=1``.
        extrapolate : {'linear', 'quadratic', 'cubic', 'spline', 'constant', \
'off'} or int, optional
            Extrapolate to times outside of the model bounds. Specify
            polynomial degree as string or any order as integer. Defaults to
            ``'linear'`` (equiv. to order 2 polynomials).

             +------------+---------------------------------------------------+
             | Value      | Description                                       |
             +============+===================================================+
             | 'constant' | Use degree zero polynomial only (extrapolate=1).  |
             +------------+---------------------------------------------------+
             | 'linear'   | Use degree-1 polynomials (extrapolate=2).         |
             +------------+---------------------------------------------------+
             | 'quadratic'| Use degree-2 polynomials (extrapolate=3).         |
             +------------+---------------------------------------------------+
             | 'cubic'    | Use degree-3 polynomials (extrapolate=4).         |
             +------------+---------------------------------------------------+
             | 'spline'   | Use all degree polynomials.                       |
             +------------+---------------------------------------------------+
             | 'off'      | Return NaN outside model bounds (extrapolate=0).  |
             +------------+---------------------------------------------------+

        Returns
        -------
        coeffs : ndarray, shape (..., ``nmax`` * (``nmax`` + 2))
            Array containing the coefficients.

        """

        dim = None if nmax is None else int(nmax*(nmax+2))
        coeffs = super().synth_coeffs(time, dim=dim, deriv=deriv,
                                      extrapolate=extrapolate)

        return coeffs

    def synth_values(self, time, radius, theta, phi, *, nmax=None,
                     deriv=None, grid=None, extrapolate=None):
        """
        Compute magnetic components from the field model.

        Parameters
        ----------
        time : ndarray, shape (...) or float
            Array containing the time in modified Julian date.
        radius : ndarray, shape (...) or float
            Radius in kilometers.
        theta : ndarray, shape (...) or float
            Colatitude in degrees.
        phi : ndarray, shape (...) or float
            Longitude in degrees.
        nmax : int, positive, optional
            Maximum degree of the harmonic expansion (default is given by the
            model coefficients, but can also be smaller, if specified).
        deriv : int, positive, optional
            Derivative in time (defaults to 0). For secular variation,
            choose ``deriv=1``.
        grid : bool, optional
            If ``True``, field components are computed on a regular grid.
            Arrays ``theta`` and ``phi`` must have one dimension less than the
            output grid since the grid will be created as their outer product.
        extrapolate : {'linear', 'quadratic', 'cubic', 'spline', 'constant', \
'off'} or int, optional
            Extrapolate to times outside of the model bounds. Specify
            polynomial degree as string or any order as integer. Defaults to
            ``'linear'`` (equiv. to order 2 polynomials).

        Returns
        -------
        B_radius, B_theta, B_phi : ndarray, shape (...)
            Radial, colatitude and azimuthal field components.

        """

        if nmax is None:
            nmax = self.nmax
        elif nmax > self.nmax:
            warnings.warn(
                f'Supplied nmax = {nmax} is incompatible with number of '
                f'coefficients. Using nmax = {self.nmax} instead.'
            )
            nmax = self.nmax

        coeffs = self.synth_coeffs(time, nmax=nmax, deriv=deriv,
                                   extrapolate=extrapolate)

        return mu.synth_values(coeffs, radius, theta, phi, nmax=nmax,
                               source=self.source, grid=grid)

    def power_spectrum(self, time, radius=None, **kwargs):
        """
        Compute the spatial power spectrum.

        Parameters
        ----------
        time : ndarray, shape (...)
            Time in modified Julian date.
        radius : float, optional
            Radius in kilometers (defaults to Earth's surface defined in
            ``basicConfig['r_surf']``).

        Returns
        -------
        R_n : ndarray, shape (..., ``nmax``)
            Spatial power spectrum of the spherical harmonic expansion up to
            degree ``nmax``.

        Other Parameters
        ----------------
        nmax : int, positive, optional
            Maximum degree of the harmonic expansion (default is given by the
            model coefficients, but can also be smaller, if specified).
        deriv : int, positive, optional
            Derivative in time (default is 0). For secular variation, choose
            ``deriv=1``.
        **kwargs : keywords
            Other options to pass to :meth:`BaseModel.synth_coeffs` method.

        See Also
        --------
        chaosmagpy.model_utils.power_spectrum

        """

        if radius is None:
            radius = config_utils.basicConfig['params.r_surf']

        coeffs = self.synth_coeffs(time, **kwargs)
        spec = mu.power_spectrum(coeffs, radius, source=self.source)

        return spec

    def plot_power_spectrum(self, time, **kwargs):
        """
        Plot the spatial power spectrum.

        Parameters
        ----------
        time : float
            Time in modified Julian date.

        Other Parameters
        ----------------
        radius : float, optional
            Radius in kilometers (defaults to Earth's surface defined in
            basicConfig['r_surf']).
        nmax : int, positive, optional
            Maximum degree of the harmonic expansion (default is given by the
            model coefficients, but can also be smaller, if specified).
        deriv : int, positive, optional
            Derivative in time (default is 0). For secular variation, choose
            ``deriv=1``.

        Notes
        -----
        For more customization get access to the figure and axes handles
        through matplotlib by using ``fig = plt.gcf()`` and ``axes = fig.axes``
        right after the call to this plotting method.

        See Also
        --------
        chaosmagpy.model_utils.power_spectrum

        """

        defaults = dict(radius=None,
                        deriv=0,
                        nmax=self.nmax,
                        titles='spatial power spectrum')

        kwargs = pu.defaultkeys(defaults, kwargs)

        radius = kwargs.pop('radius')
        nmax = kwargs.pop('nmax')
        deriv = kwargs.pop('deriv')

        units = f'({du.gauss_units(deriv)})$^2$'
        kwargs.setdefault('ylabel', units)

        R_n = self.power_spectrum(time, radius, nmax=nmax, deriv=deriv)

        pu.plot_power_spectrum(R_n, **kwargs)

    def plot_maps(self, time, radius, **kwargs):
        """
        Plot global maps of the field components.

        Parameters
        ----------
        time : ndarray, shape (), (1,) or float
            Time in modified Julian date.
        radius : ndarray, shape (), (1,) or float
            Array containing the radius in kilometers.

        Other Parameters
        ----------------
        nmax : int, positive, optional
            Maximum degree of the harmonic expansion (default is given by the
            model coefficients, but can also be smaller, if specified).
        deriv : int, positive, optional
            Derivative in time (default is 0). For secular variation, choose
            ``deriv=1``.
        **kwargs : keywords
            Other options are passed to :func:`plot_utils.plot_maps`
            function.

        Notes
        -----
        For more customization get access to the figure and axes handles
        through matplotlib by using ``fig = plt.gcf()`` and ``axes = fig.axes``
        right after the call to this plotting method.

        See Also
        --------
        chaosmagpy.plot_utils.plot_maps

        """

        defaults = dict(deriv=0, nmax=self.nmax)

        kwargs = pu.defaultkeys(defaults, kwargs)

        # remove keywords that are not intended for pcolormesh
        nmax = kwargs.pop('nmax')
        deriv = kwargs.pop('deriv')
        titles = [f'$B_r$ ($n\\leq{nmax}$, deriv={deriv})',
                  f'$B_\\theta$ ($n\\leq{nmax}$, deriv={deriv})',
                  f'$B_\\phi$ ($n\\leq{nmax}$, deriv={deriv})']

        # add plot_maps options to dictionary
        kwargs.setdefault('label', du.gauss_units(deriv))
        kwargs.setdefault('titles', titles)

        # handle optional argument: nmax > coefficient nmax
        if nmax > self.nmax:
            warnings.warn(
                'Supplied nmax = {0} is incompatible with number of model '
                'coefficients. Using nmax = {1} instead.'.format(
                    nmax, self.nmax))
            nmax = self.nmax

        time = np.array(time, dtype=float)
        theta = np.linspace(1, 179, num=320)
        phi = np.linspace(-180, 180, num=640)

        B_radius, B_theta, B_phi = self.synth_values(
            time, radius, theta, phi, nmax=nmax, deriv=deriv,
            grid=True, extrapolate=None)

        pu.plot_maps(theta, phi, B_radius, B_theta, B_phi, **kwargs)

    def plot_timeseries(self, radius, theta, phi, **kwargs):
        """
        Plot the time series of the time-dependent field components at a
        specific location.

        Parameters
        ----------
        radius : ndarray, shape (), (1,) or float
            Radius of station in kilometers.
        theta : ndarray, shape (), (1,) or float
            Colatitude in degrees :math:`[0^\\circ, 180^\\circ]`.
        phi : ndarray, shape (), (1,) or float
            Longitude in degrees.

        Other Parameters
        ----------------
        nmax : int, positive, optional
            Maximum degree of the harmonic expansion (default is given by the
            model coefficients, but can also be smaller, if specified).
        deriv : int, positive, optional
            Derivative in time (default is 0). For secular variation, choose
            ``deriv=1``.
        extrapolate : {'linear', 'spline', 'constant', 'off'}, optional
            Extrapolate to times outside of the model bounds. Defaults to
            ``'linear'``.
        **kwargs : keywords
            Other options to pass to :func:`plot_utils.plot_timeseries`
            function.

        Notes
        -----
        For more customization get access to the figure and axes handles
        through matplotlib by using ``fig = plt.gcf()`` and ``axes = fig.axes``
        right after the call to this plotting method.

        See Also
        --------
        chaosmagpy.plot_utils.plot_timeseries

        """

        defaults = dict(deriv=0,
                        nmax=self.nmax,
                        titles=['$B_r$', '$B_\\theta$', '$B_\\phi$'],
                        extrapolate=None)

        kwargs = pu.defaultkeys(defaults, kwargs)

        # remove keywords that are not intended for pcolormesh
        nmax = kwargs.pop('nmax')
        deriv = kwargs.pop('deriv')
        extrapolate = kwargs.pop('extrapolate')

        # add options to dictionary
        kwargs.setdefault('ylabel', du.gauss_units(deriv))

        time = np.linspace(self.breaks[0], self.breaks[-1], num=500)

        B_radius, B_theta, B_phi = self.synth_values(
            time, radius, theta, phi, nmax=nmax, deriv=deriv,
            extrapolate=extrapolate)

        pu.plot_timeseries(time, B_radius, B_theta, B_phi, **kwargs)

    @classmethod
    def from_bspline(cls, name, knots, coeffs, order, source=None, meta=None):
        """
        Return BaseModel instance from a B-spline representation.

        Parameters
        ----------
        name : str
            User specified name of the model.
        knots : ndarray, shape (N,)
            B-spline knots. Knots must have endpoint multiplicity equal to
            ``order``. Zero-pad ``coeffs`` if needed.
        coeffs : ndarray, shape (M, D)
            Bspline coefficients for the `M` B-splines parameterizing
            `D` dimensions.
        order : int
            Order of the B-spline.
        source : {'internal', 'external'}
            Internal or external source (defaults to ``'internal'``)
        meta : dict, optional
            Dictionary containing additional information about the model.

        Returns
        -------
        model : :class:`BaseModel`
            Class :class:`BaseModel` instance.

        """

        coeffs_pp, breaks = mu.pp_from_bspline(coeffs, knots, order)

        return cls(name, breaks=breaks, order=order, coeffs=coeffs_pp,
                   source=source, meta=meta)

    def to_shc(self, filepath, *, leap_year=None, nmin=None, nmax=None,
               header=None):
        """
        Save spherical harmonic coefficients to a file in `shc`-format.

        Parameters
        ----------
        filepath : str
            Path and name of output file `*.shc`.
        leap_year : {False, True}, optional
            Take leap year in time conversion into account. By default set to
            ``False``, so that a conversion factor of 365.25 days per year is
            used.
        nmin : int, optional
            Minimum spherical harmonic degree (defaults to 1). This will remove
            first values from coeffs if greater than 1.
        nmax : int, optional
            Maximum spherical harmonic degree (defaults to the maximum degree
            of the model given by ``self.nmax``).
        header : str, optional
            Optional header at beginning of file (defaults to a comment line
            with the name of the model given by ``self.name``).

        """

        leap_year = False if leap_year is None else leap_year
        header = f"# {self.name}\n" if header is None else header
        nmin = 1 if nmin is None else int(nmin)
        nmax = self.nmax if nmax is None else int(nmax)

        if self.coeffs is None:
            raise ValueError("Spline coefficients are missing.")

        step = max(self.order - 1, 1)

        # compute times in mjd2000
        if (self.order == 1):
            # piecewise constant, drop coefficients at last break point
            times = self.breaks[:-1]

        else:
            # insert extra samples in between break points
            times = np.array([], dtype=float)

            for start, end in zip(self.breaks[:-1], self.breaks[1:]):
                delta = (end - start) / step
                times = np.append(times, np.arange(start, end, delta))

            times = np.append(times, self.breaks[-1])

        gauss_coeffs = self.synth_coeffs(times, nmax=self.nmax)

        du.save_shcfile(times, gauss_coeffs, order=self.order,
                        filepath=filepath, nmin=nmin, nmax=nmax,
                        leap_year=leap_year, header=header)

    @classmethod
    def from_shc(cls, filepath, *, name=None, leap_year=None,
                 source=None, meta=None):
        """
        Return BaseModel instance by loading a model from an SHC-file.

        Parameters
        ----------
        filepath : str
            Path to SHC-file.
        name : str, optional
            User defined name of the model. Defaults to the filename without
            the file extension.
        leap_year : {False, True}, optional
            Take leap year in time conversion into account. By default set to
            ``False``, so that a conversion factor of 365.25 days per year is
            used.
        source : {'internal', 'external'}
            Internal or external source (defaults to ``'internal'``)
        meta : dict, optional
            Dictionary containing additional information about the model.

        Returns
        -------
        model : :class:`BaseModel`
            Class :class:`BaseModel` instance.

        """

        if name is None:
            # get name without extension
            name = os.path.splitext(os.path.basename(filepath))[0]

        source = 'internal' if source is None else source
        leap_year = False if leap_year is None else leap_year

        time, coeffs, params = du.load_shcfile(filepath, leap_year=leap_year)

        coeffs = coeffs.T  # (Nt, Nc): simplifies array manipulations

        nmin = params['nmin']
        nmax = params['nmax']
        order = params['order']
        step = max(params['step'], 1)

        if order == 1:
            # piecewise constant
            # duplicate endpoint to have a proper interval for the polynomial
            breaks = np.append(time, time[-1])  # zero length interval is ok
        else:
            breaks = time[::step]

        # need to pad coefficients if nmin > 1 (no n < nmin coeffs in shc file)
        if nmin > 1:
            coeffs_pad = np.zeros((coeffs.shape[0], nmax*(nmax + 2)))
            coeffs_pad[:, int(nmin**2 - 1):] = coeffs
        else:
            coeffs_pad = coeffs

        if (order == 1):  # piecewise constant

            coeffs_pp = coeffs_pad[None, ...]  # insert singleton at 0th axis

            return cls(name, breaks=breaks, order=order, coeffs=coeffs_pp,
                       source=source, meta=meta)

        else:  # model must be time-dependent (incl. piecewise constant)

            # there may be extra sites to extend the model interval,
            # but these extrapolation sites are just discarded here
            end = ((time.size - 1) // step) * step + 1

            knots = mu.augment_breaks(breaks, order)
            spl = sip.make_lsq_spline(time[:end], coeffs_pad[:end, :],
                                      knots, order - 1)
            coeffs_pp = spl.c.copy()

            # convert B-spline basis to PPoly
            return cls.from_bspline(name, knots=knots, coeffs=coeffs_pp,
                                    order=order, source=source, meta=meta)


class CHAOS(object):
    """
    Class for the time-dependent geomagnetic field model CHAOS.

    Parameters
    ----------
    breaks : ndarray, shape (m+1,)
        Break points for piecewise polynomial representation of the
        time-dependent internal (i.e. large-scale core) field in modified
        Julian date format.
    order : int, positive
        Order `k` of polynomial pieces (e.g. 4 = cubic) of the time-dependent
        internal field.
    coeffs_tdep : ndarray, shape (`k`, `m`, ``n_tdep`` * (``n_tdep`` + 2))
        Coefficients of the time-dependent internal field as piecewise
        polynomial.
    coeffs_static : ndarray,  shape (``n_static`` * (``n_static`` + 2),)
        Coefficients of the static internal (i.e. small-scale crustal) field.
    coeffs_sm : ndarray, shape (``n_sm`` * (``n_sm`` + 2),)
        Coefficients of the static external field in SM coordinates.
    coeffs_gsm : ndarray, shape (``n_gsm`` * (``n_gsm`` + 2),)
        Coefficients of the static external field in GSM coordinates.
    breaks_delta : dict with ndarrays, shape (:math:`m_q` + 1,)
        Breaks of baseline corrections of static external field in SM
        coordinates. The dictionary keys are ``'q10'``, ``'q11'``, ``'s11'``.
    coeffs_delta : dict with ndarrays, shape (1, :math:`m_q`)
        Coefficients of baseline corrections of static external field in SM
        coordinates. The dictionary keys are ``'q10'``, ``'q11'``, ``'s11'``.
    breaks_euler : dict with ndarrays, shape (:math:`m_e` + 1,)
        Dictionary containing satellite name as key and corresponding break
        vectors of Euler angles (keys are ``'oersted'``, ``'champ'``,
        ``'sac_c'``, ``'swarm_a'``, ``'swarm_b'``, ``'swarm_c'``,
        ``'cryosat-2_1'``).
    coeffs_euler : dict with ndarrays, shape (1, :math:`m_e`, 3)
        Dictionary containing satellite name as key and arrays of the Euler
        angles alpha, beta and gamma as trailing dimension (keys are
        ``'oersted'``, ``'champ'``, ``'sac_c'``, ``'swarm_a'``, ``'swarm_b'``,
        ``'swarm_c'``, ``'cryosat-2_1'``).
    breaks_cal : dict with ndarrays, shape (:math:`m_c` + 1,)
        Dictionary containing satellite name as key and corresponding break
        vectors for the calibration parameters (keys are ``'cryosat-2_1'``).
    coeffs_cal : dict with ndarrays, shape (1, :math:`m_c`, 3)
        Dictionary containing satellite name as key and arrays of the 9 basic
        calibration parameters (3 offsets, 3 sensitivities, 3
        non-orthogonality angles) (keys are ``'cryosat-2_1'``).
    name : str, optional
        User defined name of the model. Defaults to ``'CHAOS'``.
    meta : dict, optional
        Dictionary containing additional information about the model.

    Attributes
    ----------
    timestamp : str
        UTC timestamp at initialization.
    model_tdep : :class:`BaseModel` instance
        Time-dependent internal field model.
    model_static : :class:`BaseModel` instance
        Static internal field model.
    model_euler : dict of :class:`Base` instances
        Dictionary containing the satellite's name as key and the Euler angles
        as :class:`Base` class instance.
    model_cal : dict of :class:`Base` instances
        Dictionary containing the satellite's name as key and the calibration
        parameters as :class:`Base` class instance.
    n_sm : int, positive
        Maximum spherical harmonic degree of external field in SM coordinates.
    coeffs_sm : ndarray, shape (``n_sm`` * (``n_sm`` + 2),)
        Coefficients of static external field in SM coordinates.
    n_gsm : int, positive
        Maximum spherical harmonic degree of external field in GSM coordinates.
    coeffs_gsm : ndarray, shape (``n_gsm`` * (``n_gsm`` + 2),)
        Coefficients of static external field in GSM coordinates.
    breaks_delta : dict with ndarrays, shape (:math:`m_q` +1,)
        Breaks of baseline corrections of static external field in SM
        coordinates. The dictionary keys are ``'q10'``, ``'q11'``, ``'s11'``.
    coeffs_delta : dict with ndarrays, shape (1, :math:`m_q`)
        Coefficients of baseline corrections of static external field in SM
        coordinates. The dictionary keys are ``'q10'``, ``'q11'``, ``'s11'``.
    name : str, optional
        User defined name of the model.
    meta : dict, optional
        Dictionary containing additional information about the model.

    Examples
    --------
    Load for example the MAT-file ``CHAOS-6-x7.mat`` in the current working
    directory like this:

    >>> import chaosmagpy as cp
    >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
    >>> print(model)

    For more examples, see the documentation of the methods below.

    """

    def __init__(
        self,
        breaks,
        order=None,
        *,
        coeffs_tdep=None,
        coeffs_static=None,
        coeffs_sm=None,
        coeffs_gsm=None,
        breaks_delta=None,
        coeffs_delta=None,
        breaks_euler=None,
        coeffs_euler=None,
        breaks_cal=None,
        coeffs_cal=None,
        name=None,
        meta=None
    ):
        """
        Initialize the CHAOS model.

        """

        self.timestamp = str(datetime.datetime.now(datetime.timezone.utc))

        # internal field
        if coeffs_tdep is None:
            self.model_tdep = None
        else:
            self.model_tdep = BaseModel(
                name='model_tdep',
                breaks=breaks,
                order=order,
                coeffs=coeffs_tdep,
                source='internal'
            )

        if coeffs_static is None:
            self.model_static = None
        else:
            self.model_static = BaseModel(
                name='model_static',
                breaks=breaks[[0, -1]],
                order=1,
                coeffs=coeffs_static,
                source='internal'
            )

        # helper for returning the degree of the provided coefficients
        def dimension(coeffs):
            if coeffs is None:
                return coeffs
            else:
                return int(np.sqrt(coeffs.shape[-1] + 1) - 1)

        # external source in SM reference
        self.coeffs_sm = coeffs_sm
        self.n_sm = dimension(coeffs_sm)

        # external source in GSM reference
        self.coeffs_gsm = coeffs_gsm
        self.n_gsm = dimension(coeffs_gsm)

        # external source in SM reference: RC offset
        self.breaks_delta = breaks_delta
        self.coeffs_delta = coeffs_delta

        # Euler angles
        if breaks_euler is None:
            self.model_euler = None
        else:
            satellites = tuple([*breaks_euler.keys()])
            self.model_euler = dict()

            for k, satellite in enumerate(satellites):

                try:
                    Euler_prerotation = meta['params']['Euler_prerotation'][k]
                except KeyError:
                    Euler_prerotation = None

                model = Base(
                    satellite,
                    order=1,
                    breaks=breaks_euler[satellite],
                    coeffs=coeffs_euler[satellite],
                    meta={
                        'Euler_prerotation': Euler_prerotation
                    }
                )
                self.model_euler[satellite] = model

        # calibration parameters
        if breaks_cal is None:
            self.model_cal = None
        else:
            satellites = tuple([*breaks_cal.keys()])
            self.model_cal = dict()

            for k, satellite in enumerate(satellites):

                model = Base(
                    satellite,
                    order=1,
                    breaks=breaks_cal[satellite],
                    coeffs=coeffs_cal[satellite],
                )
                self.model_cal[satellite] = model

        # give the model a name: CHAOS or user input
        if name is None:
            self.name = "CHAOS"
        else:
            self.name = name

        self.meta = meta

    def __call__(self, time, radius, theta, phi, rc_e=None, rc_i=None,
                 source_list=None, nmax_static=None, verbose=None):
        """
        Calculate the magnetic field of all sources from the CHAOS model.

        All sources means the time-dependent and static internal fields, and
        the external magnetospheric (SM/GSM) fields including their
        induced parts.

        Parameters
        ----------
        time : ndarray, shape (...) or float
            Array containing the time in modified Julian dates.
        radius : ndarray, shape (...) or float
            Radius of station in kilometers.
        theta : ndarray, shape (...) or float
            Colatitude in degrees :math:`[0^\\circ, 180^\\circ]`.
        phi : ndarray, shape (...) or float
            Longitude in degrees.
        rc_e : ndarray, shape (...), optional
            External part of the RC-index (defaults to linearly interpolating
            the hourly values given by the built-in RC-index file).
        rc_i : ndarray, shape (...), optional
            Internal part of the RC-index (defaults to linearly interpolating
            the hourly values given by the built-in RC-index file).
        source_list : list, ['tdep', 'static', 'gsm', 'sm'] or \
str, {'internal', 'external'}
            Specify sources in any order. Default is all sources. Instead of a
            list, pass ``source_list='internal'`` which is equivalent to
            ``source_list=['tdep', 'static']`` (internal sources) or
            ``source_list='external'`` which is the same as
            ``source_list=['gsm', 'sm']`` (external sources including induced
            part).
        nmax_static : int, optional
            Maximum spherical harmonic degree of the static internal magnetic
            field (defaults to 85).
        verbose : {False, True}, optional
            Print messages (defaults to ``False``).

        Returns
        -------
        B_radius, B_theta, B_phi : ndarray, shape (...)
            Radial, colatitude and azimuthal field components.

        Examples
        --------
        >>> import chaosmagpy as cp
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> Br, Bt, Bp = model(0., 6371.2, 45., 0., source_list=['tdep', \
'static'])  # only internal sources
        >>> Br
        array(-40418.23217586)

        """

        time = np.asarray(time, dtype=float)
        radius = np.asarray(radius, dtype=float)
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)

        if source_list is None:
            source_list = ['tdep', 'static', 'gsm', 'sm']

        elif source_list == 'internal':
            source_list = ['tdep', 'static']

        elif source_list == 'external':
            source_list = ['gsm', 'sm']

        source_list = np.ravel(np.array(source_list))

        verbose = bool(verbose)

        # get shape of broadcasted result
        try:
            b = np.broadcast(time, radius, theta, phi)
        except ValueError:
            print('Cannot broadcast grid shapes:')
            print(f'time:   {time.shape}')
            print(f'radius: {radius.shape}')
            print(f'theta:  {theta.shape}')
            print(f'phi:    {phi.shape}')
            raise

        grid_shape = b.shape

        B_radius = np.zeros(grid_shape)
        B_theta = np.zeros(grid_shape)
        B_phi = np.zeros(grid_shape)

        if 'tdep' in source_list:

            if verbose:
                print(f'Computing time-dependent internal field'
                      f' up to degree {self.model_tdep.nmax}.')

            s = timer()
            B_radius_new, B_theta_new, B_phi_new = self.synth_values_tdep(
                time, radius, theta, phi)

            B_radius += B_radius_new
            B_theta += B_theta_new
            B_phi += B_phi_new
            e = timer()

            if verbose:
                print('Finished in {:.6} seconds.'.format(e-s))

        if 'static' in source_list:

            nmax_static = 85 if nmax_static is None else nmax_static

            if verbose:
                print(f'Computing static internal (i.e. small-scale crustal) '
                      f'field up to degree {nmax_static}.')

            s = timer()
            B_radius_new, B_theta_new, B_phi_new = self.synth_values_static(
                radius, theta, phi, nmax=nmax_static)

            B_radius += B_radius_new
            B_theta += B_theta_new
            B_phi += B_phi_new
            e = timer()

            if verbose:
                print('Finished in {:.6} seconds.'.format(e-s))

        if 'gsm' in source_list:

            if verbose:
                print(f'Computing GSM field up to degree {self.n_gsm}.')

            s = timer()
            B_radius_new, B_theta_new, B_phi_new = self.synth_values_gsm(
                time, radius, theta, phi, source='all')

            B_radius += B_radius_new
            B_theta += B_theta_new
            B_phi += B_phi_new
            e = timer()

            if verbose:
                print('Finished in {:.6} seconds.'.format(e-s))

        if 'sm' in source_list:

            if verbose:
                print(f'Computing SM field up to degree {self.n_sm}.')

            s = timer()
            B_radius_new, B_theta_new, B_phi_new = self.synth_values_sm(
                time, radius, theta, phi, rc_e=rc_e, rc_i=rc_i, source='all')

            B_radius += B_radius_new
            B_theta += B_theta_new
            B_phi += B_phi_new
            e = timer()

            if verbose:
                print('Finished in {:.6} seconds.'.format(e-s))

        return B_radius, B_theta, B_phi

    def __str__(self):
        """
        Print model version and initialization timestamp.
        """

        string = (f"{self.name}: Initialized on {self.timestamp} UTC.")

        return string

    def synth_coeffs_tdep(self, time, *, nmax=None, **kwargs):
        """
        Compute the spherical harmonic coefficients of the time-dependent
        internal magnetic field from the CHAOS model.

        Parameters
        ----------
        time : ndarray, shape (...) or float
            Array containing the time in modified Julian dates.
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        **kwargs : keywords
            Other options to pass to :meth:`BaseModel.synth_coeffs`
            method.

        Returns
        -------
        coeffs : ndarray, shape (..., ``nmax`` * (``nmax`` + 2))
            Coefficients of the time-dependent internal field.

        Examples
        --------
        >>> import chaosmagpy as cp
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> time = np.array([0., 10.])

        >>> model.synth_coeffs_tdep(time, nmax=1)  # dipole coefficients
        array([[-29614.72797782,  -1728.47079907,   5185.50518939],
               [-29614.33800306,  -1728.13680075,   5184.89196286]])

        >>> model.synth_coeffs_tdep(time, nmax=1, deriv=1)  # SV coefficients
        array([[ 14.25577646,  12.20214856, -22.43412895],
               [ 14.2317297 ,  12.19625726, -22.36146885]])

        """

        if self.model_tdep is None:
            raise ValueError("Time-dependent internal field coefficients "
                             "are missing.")

        return self.model_tdep.synth_coeffs(time, nmax=nmax, **kwargs)

    def synth_values_tdep(self, time, radius, theta, phi, *, nmax=None,
                          deriv=None, grid=None, extrapolate=None):
        """
        Compute the vector components of the time-dependent internal
        magnetic field from the CHAOS model.

        Parameters
        ----------
        time : ndarray, shape (...) or float
            Array containing the time in modified Julian dates.
        radius : ndarray, shape (...) or float
            Radius of station in kilometers.
        theta : ndarray, shape (...) or float
            Colatitude in degrees :math:`[0^\\circ, 180^\\circ]`.
        phi : ndarray, shape (...) or float
            Longitude in degrees.
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        deriv : int, positive, optional
            Derivative in time (None defaults to 0). For secular variation,
            choose ``deriv=1``.
        grid : bool, optional
            If ``True``, field components are computed on a regular grid.
            Arrays ``theta`` and ``phi`` must have one dimension less than the
            output grid since the grid will be created as their outer product.
        extrapolate : {'linear', 'quadratic', 'cubic', 'spline', 'constant', \
'off'} or int, optional
            Extrapolate to times outside of the model bounds. Specify
            polynomial degree as string or any order as integer. Defaults to
            ``'linear'`` (equiv. to order 2 polynomials).

        Returns
        -------
        B_radius, B_theta, B_phi : ndarray, shape (...)
            Radial, colatitude and azimuthal field components.

        Examples
        --------
        >>> import chaosmagpy as cp
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> time = np.array([0., 10.])

        Compute magnetic field components at specific location.

        >>> Br, Bt, Bp = model.synth_values_tdep(time, 6371.2, 45., 0.)
        >>> Br
        array([-40422.44815265, -40423.15091334])

        Only dipole contribution:

        >>> Br, Bt, Bp = model.synth_values_tdep(time, 6371.2, 45., 0., nmax=1)
        >>> Br
        array([-44325.97679843, -44324.95294588])

        Secular variation:

        >>> Br, Bt, Bp = model.synth_values_tdep(time, 6371.2, 45., 0.,\
 deriv=1)
        >>> Br
        array([-25.64604374, -25.69002078])

        """

        if self.model_tdep is None:
            raise ValueError("Time-dependent internal field coefficients "
                             "are missing.")

        return self.model_tdep.synth_values(
            time, radius, theta, phi, nmax=nmax, deriv=deriv, grid=grid,
            extrapolate=extrapolate)

    def plot_timeseries_tdep(self, radius, theta, phi, **kwargs):
        """
        Plot the time series of the time-dependent internal field from the
        CHAOS model at a given location.

        Parameters
        ----------
        radius : ndarray, shape (), (1,) or float
            Radius of station in kilometers.
        theta : ndarray, shape (), (1,) or float
            Colatitude in degrees :math:`[0^\\circ, 180^\\circ]`.
        phi : ndarray, shape (), (1,) or float
            Longitude in degrees.
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        deriv : int, positive, optional
            Derivative in time (default is 0). For secular variation, choose
            ``deriv=1``.
        **kwargs : keywords
            Other options to pass to :meth:`BaseModel.plot_timeseries`
            method.

        Notes
        -----
        For more customization get access to the figure and axes handles
        through matplotlib by using ``fig = plt.gcf()`` and ``axes = fig.axes``
        right after the call to this plotting method.

        """

        if self.model_tdep is None:
            raise ValueError("Time-dependent internal field coefficients "
                             "are missing.")

        self.model_tdep.plot_timeseries(radius, theta, phi, **kwargs)

    def plot_maps_tdep(self, time, radius, *, nmax=None, deriv=None, **kwargs):
        """
        Plot global map of the time-dependent internal field from the CHAOS
        model.

        Parameters
        ----------
        time : ndarray, shape (), (1,) or float
            Time given as MJD2000 (modified Julian date).
        radius : ndarray, shape (), (1,) or float
            Array containing the radius in kilometers.
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        deriv : int, positive, optional
            Derivative in time (default is 0). For secular variation, choose
            ``deriv=1``.
        **kwargs : keywords
            Other options are passed to :meth:`BaseModel.plot_maps` method.

        Notes
        -----
        For more customization get access to the figure and axes handles
        through matplotlib by using ``fig = plt.gcf()`` and ``axes = fig.axes``
        right after the call to this plotting method.

        """

        if self.model_tdep is None:
            raise ValueError("Time-dependent internal field coefficients "
                             "are missing.")

        self.model_tdep.plot_maps(time, radius, nmax=nmax,
                                  deriv=deriv, **kwargs)

    def synth_coeffs_static(self, *, nmax=None, **kwargs):
        """
        Compute the spherical harmonic coefficients of the static internal
        magnetic field from the CHAOS model.

        Parameters
        ----------
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        **kwargs : keywords
            Other options are passed to :meth:`BaseModel.synth_coeffs`
            method.

        Returns
        -------
        coeffs : ndarray, shape (``nmax`` * (``nmax`` + 2),)
            Coefficients of the static internal field.

        Examples
        --------
        >>> import chaosmagpy as cp
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> model.synth_coeffs_static(nmax=50)
        array([ 0.     , 0.     ,  0.     , ...,  0.01655, -0.06339,  0.00715])

        """

        if self.model_static is None:
            raise ValueError("Static internal field coefficients are missing.")

        time = self.model_static.breaks[0]
        return self.model_static.synth_coeffs(time, nmax=nmax, **kwargs)

    def synth_values_static(self, radius, theta, phi, *, nmax=None, **kwargs):
        """
        Compute the vector components of the static internal magnetic field
        from the CHAOS model.

        Parameters
        ----------
        radius : ndarray, shape (...) or float
            Radius of station in kilometers.
        theta : ndarray, shape (...) or float
            Colatitude in degrees :math:`[0^\\circ, 180^\\circ]`.
        phi : ndarray, shape (...) or float
            Longitude in degrees.
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        **kwargs : keywords
            Other options are passed to :meth:`BaseModel.synth_values`
            method.

        Returns
        -------
        B_radius, B_theta, B_phi : ndarray, shape (...)
            Radial, colatitude and azimuthal field components.

        Examples
        --------
        >>> import chaosmagpy as cp
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> Br, Bt, Bp = model.synth_values_static(6371.2, 45., 0., nmax=50)
        >>> Br
        array(-7.5608993)

        """

        if self.model_static is None:
            raise ValueError("Static internal field coefficients are missing.")

        time = self.model_static.breaks[0]
        return self.model_static.synth_values(time, radius, theta, phi,
                                              nmax=nmax, **kwargs)

    def plot_maps_static(self, radius, *, nmax=None, **kwargs):
        """
        Plot global map of the static internal field from the CHAOS model.

        Parameters
        ----------
        radius : ndarray, shape (), (1,) or float
            Array containing the radius in kilometers.
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        **kwargs : keywords
            Other options are passed to :meth:`BaseModel.plot_maps`
            method.

        Notes
        -----
        For more customization get access to the figure and axes handles
        through matplotlib by using ``fig = plt.gcf()`` and ``axes = fig.axes``
        right after the call to this plotting method.

        """

        if self.model_static is None:
            raise ValueError("Static internal field coefficients are missing.")

        defaults = dict(cmap='nio',
                        deriv=0,
                        vmax=200,
                        vmin=-200)

        kwargs = pu.defaultkeys(defaults, kwargs)

        time = self.model_static.breaks[0]

        self.model_static.plot_maps(time, radius, nmax=nmax, **kwargs)

    def synth_coeffs_gsm(self, time, *, nmax=None, source=None):
        """
        Compute the spherical harmonic coefficients of the far-magnetospheric
        magnetic field from the CHAOS model.

        Parameters
        ----------
        time : ndarray, shape (...) or float
            Array containing the time in days.
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        source : {'external', 'internal'}, optional
            Choose source either external or internal (default is 'external').

        Returns
        -------
        coeffs : ndarray, shape (..., ``nmax`` * (``nmax`` + 2))
            Spherical harmonic coefficients of the far-magnetospheric magnetic
            field with respect to geographic coordinates (GEO).

        Examples
        --------
        >>> import chaosmagpy as cp
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> model.synth_coeffs_gsm(0.0)
        array([11.63982782, -4.9276483 , -2.36281582,  0.46063709, -0.37934517,
               -0.18234297,  0.06281656,  0.07757099])

        """

        if self.coeffs_gsm is None:
            raise ValueError("External GSM field coefficients are missing.")

        # handle optional argument: nmax
        if nmax is None:
            nmax = self.n_gsm
        elif nmax > self.n_gsm:
            warnings.warn(
                'Supplied nmax = {0} is incompatible with number of model '
                'coefficients. Using nmax = {1} instead.'.format(
                    nmax, self.n_gsm))
            nmax = self.n_gsm

        if source is None:
            source = 'external'

        # ensure ndarray input
        time = np.asarray(time, dtype=float)

        # use static part to define modelled period
        start = self.model_static.breaks[0]
        end = self.model_static.breaks[-1]

        if np.amin(time) < start or np.amax(time) > end:
            warnings.warn(
                'Requested coefficients are outside of the '
                f'model period from {start} to {end}. Doing linear '
                'extrapolation of the coefficients in the GSM reference '
                'frame.')

        # build rotation matrix from file
        frequency_spectrum = np.load(
            config_utils.basicConfig['file.GSM_spectrum'])
        assert np.all(frequency_spectrum['dipole']
                      == config_utils.basicConfig['params.dipole']), \
            "GSM rotation coefficients not compatible with the chosen dipole."

        if source == 'external':
            # unpack file: oscillations per day, complex spectrum
            frequency = frequency_spectrum['frequency']
            spectrum = frequency_spectrum['spectrum']
            scaled = frequency_spectrum['scaled']

            # build rotation matrix for external field coefficients GSM -> GEO
            rotate_gauss = cu.synth_rotate_gauss(
                time, frequency, spectrum, scaled=scaled)

            # rotate external GSM coefficients to GEO reference
            coeffs = np.matmul(rotate_gauss, self.coeffs_gsm)

        elif source == 'internal':
            # unpack file: oscillations per day, complex spectrum
            frequency_ind = frequency_spectrum['frequency_ind']
            spectrum_ind = frequency_spectrum['spectrum_ind']
            scaled = frequency_spectrum['scaled']

            # build rotation matrix for external field coefficients GSM -> GEO
            rotate_gauss_ind = cu.synth_rotate_gauss(
                time, frequency_ind, spectrum_ind, scaled=scaled)

            # rotate internal GSM coefficients to GEO reference
            coeffs = np.matmul(rotate_gauss_ind, self.coeffs_gsm)

        else:
            raise ValueError(f'Unknown source "{source}". '
                             'Use {''external'', ''internal''}.')

        return coeffs[..., :nmax*(nmax+2)]

    def synth_values_gsm(self, time, radius, theta, phi, *, nmax=None,
                         source=None, grid=None):
        """
        Compute the vector components of the far-magnetospheric magnetic
        field from the CHAOS model.

        Parameters
        ----------
        time : float or ndarray, shape (...)
            Time given as MJD2000 (modified Julian date).
        radius : float or ndarray, shape (...)
            Array containing the radius in kilometers.
        theta : float or ndarray, shape (...)
            Array containing the colatitude in degrees
            :math:`[0^\\circ,180^\\circ]`.
        phi : float or ndarray, shape (...)
            Array containing the longitude in degrees.
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        source : {'all', 'external', 'internal'}, optional
            Choose source to be external (inducing), internal (induced) or
            both added (default to 'all').
        grid : bool, optional
            If ``True``, field components are computed on a regular grid,
            which is created from ``theta`` and ``phi`` as their outer product
            (defaults to ``False``).

        Returns
        -------
        B_radius, B_theta, B_phi : ndarray, shape (...)
            Radial, colatitude and azimuthal field components.

        Examples
        --------
        >>> import chaosmagpy as cp
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> time = np.array([0., 10.])
        >>> Br, Bt, Bp = model.synth_values_gsm(time, 6371.2, 45., 0.)
        >>> Br
        array([-8.18751916, -8.25661729])

        """

        source = 'all' if source is None else source
        grid = False if grid is None else grid

        if source == 'all':
            coeffs_ext = self.synth_coeffs_gsm(time, nmax=nmax,
                                               source='external')
            coeffs_int = self.synth_coeffs_gsm(time, nmax=nmax,
                                               source='internal')

            B_radius_ext, B_theta_ext, B_phi_ext = mu.synth_values(
                coeffs_ext, radius, theta, phi, source='external', grid=grid)
            B_radius_int, B_theta_int, B_phi_int = mu.synth_values(
                coeffs_int, radius, theta, phi, source='internal', grid=grid)

            B_radius = B_radius_ext + B_radius_int
            B_theta = B_theta_ext + B_theta_int
            B_phi = B_phi_ext + B_phi_int

        elif source in ['external', 'internal']:
            coeffs = self.synth_coeffs_gsm(time, nmax=nmax, source=source)
            B_radius, B_theta, B_phi = mu.synth_values(
                coeffs, radius, theta, phi, source=source, grid=grid)

        else:
            raise ValueError(f'Unknown source "{source}". '
                             'Use {''all'', ''external'', ''internal''}.')

        return B_radius, B_theta, B_phi

    def synth_coeffs_sm(self, time, *, nmax=None, source=None, rc=None):
        """
        Compute the spherical harmonic coefficients of the near-magnetospheric
        magnetic field from the CHAOS model.

        Parameters
        ----------
        time : ndarray, shape (...) or float
            Array containing the time in days.
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        source : {'external', 'internal'}, optional
            Choose source either external or internal (default is 'external').
        rc : ndarray, shape (...), optional
            External (internal) part of the RC-index (defaults to linearly
            interpolating the hourly values given by the built-in RC-index
            file). Use the external part of the RC-index for
            ``source='external'`` (default) and the internal part for
            ``source='internal'``. See also the notes section below for more
            information on the RC-index file.

        Returns
        -------
        coeffs : ndarray, shape (..., ``nmax`` * (``nmax`` + 2))
            Spherical harmonic coefficients of the near-magnetospheric magnetic
            field with respect to geographic coordinates (GEO).

        Notes
        -----
        The computation of the near-magnetospheric field coefficients requires
        the RC-index. If the values of the RC-index are not supplied
        directly via the ``rc`` input argument, a built-in RC-index file is
        loaded. This file goes with a specific version of the CHAOS model,
        which can be inspected by running
        ``print(chaosmagpy.basicConfig['params.CHAOS_version'])`` in a Python
        session. It is recommended to use this version of the CHAOS model
        together with the built-in RC-index.

        For the latest CHAOS model, if necessary, a suitable RC-index file can
        be downloaded at :rc_url:`\\ `. Overwrite the use of the built-in
        RC-index file by either providing interpolated values from the
        downloaded file via the ``rc`` input argument, or by replacing
        the path in ChaosMagPy's configuration dictionary
        ``chaosmagpy.basicParams['file.RC_index']`` with the path to the
        downloaded file (for more details, see Sect.
        :ref:`sec-configuration-change-rc-index-file`).

        See Also
        --------
        CHAOS.synth_values_sm

        Examples
        --------
        >>> import chaosmagpy as cp
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> model.synth_coeffs_sm(0.)
        array([53.20309271,  3.79138724, -8.59458138, -0.62818711,  1.45506171,
               -0.57977672, -0.31660638, -0.43888236])

        """

        if self.coeffs_sm is None:
            raise ValueError("External SM field coefficients are missing.")

        # handle optional argument: nmax
        if nmax is None:
            nmax = self.n_sm
        elif nmax > self.n_sm:
            warnings.warn(
                'Supplied nmax = {0} is incompatible with number of model '
                'coefficients. Using nmax = {1} instead.'.format(
                    nmax, self.n_sm))
            nmax = self.n_sm

        if source is None:
            source = 'external'

        # find smallest overlapping time period for breaks_delta
        start = np.amax([self.breaks_delta['q10'][0],
                         self.breaks_delta['q11'][0],
                         self.breaks_delta['s11'][0]])
        end = np.amin([self.breaks_delta['q10'][-1],
                       self.breaks_delta['q11'][-1],
                       self.breaks_delta['s11'][-1]])

        # ensure ndarray input
        time = np.asarray(time, dtype=float)

        if np.amin(time) < start or np.amax(time) > end:
            warnings.warn(
                'Requested coefficients are outside of the '
                f'model period from {start} to {end}. Doing linear '
                'extrapolation of the coefficients in the SM reference frame.')

        # load the Fourier spectrum of the coordinate transformation from file
        frequency_spectrum = np.load(
            config_utils.basicConfig['file.SM_spectrum'])
        assert np.all(frequency_spectrum['dipole']
                      == config_utils.basicConfig['params.dipole']), \
            ("Coefficients for the SM coordinate transformation are not " +
             "compatible with the chosen dipole.")

        if rc is None:

            default = config_utils.basicConfig.defaults['file.RC_index'][0]
            file = config_utils.basicConfig['file.RC_index']

            if file == default:
                warnings.warn(
                    'HEALTH WARNING: ChaosMagPy is loading the built-in '
                    'RC-index file recommended for the CHAOS model version '
                    f'{config_utils.basicConfig["params.CHAOS_version"]}. If '
                    'this is not the CHAOS model version you are using, '
                    'consider changing the model and/or the built-in RC-index '
                    'file (see https://chaosmagpy.readthedocs.io/en/'
                    'master/configuration.html#change-rc-index-file '
                    'on how to change this file).'
                )

            # load RC-index file: first hdf5 then dat-file format
            try:
                with h5py.File(file, 'r') as f_RC:

                    # create linear interpolant of the RC-index
                    RC = sip.interp1d(f_RC['time'], f_RC['RC_' + source[0]],
                                      kind='linear', bounds_error=False)

            except OSError:
                f_RC = du.load_RC_datfile(file)

                # create linear interpolant of the RC-index
                RC = sip.interp1d(f_RC['time'], f_RC['RC_' + source[0]],
                                  kind='linear', bounds_error=False)

            # check RC bounds here to print specific error message
            rc_below, rc_above = RC._check_bounds(time)

            if rc_below.any() or rc_above.any():
                raise ValueError("Requested coefficients are outside of the "
                                 "period covered by the built-in RC-index "
                                 "file. Either supply values of the RC-index "
                                 "directly via the ``rc`` input argument or "
                                 "provide the path to an extended RC-index "
                                 "file by updating the path in "
                                 "chaosmagpy.basicParams['file.RC_index'] "
                                 "(see https://chaosmagpy.readthedocs.io/en/"
                                 "master/configuration.html#change-rc-"
                                 "index-file for details).")

            rc = RC(time)  # interpolate RC-index

        else:
            rc = np.asarray(rc, dtype=float)  # use supplied index values

        # use piecewise polynomials to evaluate baseline correction in bins
        delta_q10 = sip.PPoly.construct_fast(
            self.coeffs_delta['q10'].astype(float),
            self.breaks_delta['q10'].astype(float), extrapolate=True)

        delta_q11 = sip.PPoly.construct_fast(
            self.coeffs_delta['q11'].astype(float),
            self.breaks_delta['q11'].astype(float), extrapolate=True)

        delta_s11 = sip.PPoly.construct_fast(
            self.coeffs_delta['s11'].astype(float),
            self.breaks_delta['s11'].astype(float), extrapolate=True)

        # unpack file: oscillations per day, complex spectrum
        frequency = frequency_spectrum['frequency']
        spectrum = frequency_spectrum['spectrum']
        scaled = frequency_spectrum['scaled']

        # build rotation matrix for external field coefficients SM -> GEO
        rotate_gauss = cu.synth_rotate_gauss(
            time, frequency, spectrum, scaled=scaled)

        if source == 'external':
            coeffs_sm = np.empty(time.shape + (self.n_sm*(self.n_sm + 2),))

            coeffs_sm[..., 0] = rc*self.coeffs_sm[0] + delta_q10(time)
            coeffs_sm[..., 1] = rc*self.coeffs_sm[1] + delta_q11(time)
            coeffs_sm[..., 2] = rc*self.coeffs_sm[2] + delta_s11(time)
            coeffs_sm[..., 3:] = self.coeffs_sm[3:]

            # rotate external SM coefficients to GEO reference
            coeffs = np.einsum('...ij,...j', rotate_gauss, coeffs_sm)

        elif source == 'internal':
            # unpack file: oscillations per day, complex spectrum
            frequency = frequency_spectrum['frequency_ind']
            spectrum = frequency_spectrum['spectrum_ind']

            # build rotation matrix for induced coefficients SM -> GEO
            rotate_gauss_ind = cu.synth_rotate_gauss(
                time, frequency, spectrum, scaled=scaled)

            # take degree 1 matrix elements of unmodified rotation matrix
            # since induction effect will be accounted for by RC_i
            rotate_gauss = rotate_gauss[..., :3, :3]

            coeffs_sm = np.empty(time.shape + (3,))

            coeffs_sm[..., 0] = rc*self.coeffs_sm[0]
            coeffs_sm[..., 1] = rc*self.coeffs_sm[1]
            coeffs_sm[..., 2] = rc*self.coeffs_sm[2]

            coeffs_sm_ind = np.empty(time.shape + (self.n_sm*(self.n_sm + 2),))

            coeffs_sm_ind[..., 0] = delta_q10(time)
            coeffs_sm_ind[..., 1] = delta_q11(time)
            coeffs_sm_ind[..., 2] = delta_s11(time)
            coeffs_sm_ind[..., 3:] = self.coeffs_sm[3:]

            # rotate internal SM coefficients to GEO reference
            coeffs = np.einsum('...ij,...j', rotate_gauss_ind, coeffs_sm_ind)
            coeffs[..., :3] += np.einsum('...ij,...j', rotate_gauss, coeffs_sm)

        else:
            raise ValueError(f'Unknown source "{source}". '
                             'Use one of {"external", "internal"}.')

        return coeffs[..., :nmax*(nmax+2)]

    def synth_values_sm(self, time, radius, theta, phi, *,
                        rc_e=None, rc_i=None, nmax=None, source=None,
                        grid=None):
        """
        Compute the vector components of the near-magnetospheric magnetic
        field from the CHAOS model.

        Parameters
        ----------
        time : float or ndarray, shape (...)
            Time given as MJD2000 (modified Julian date).
        radius : float or ndarray, shape (...)
            Array containing the radius in kilometers.
        theta : float or ndarray, shape (...)
            Array containing the colatitude in degrees
            :math:`[0^\\circ,180^\\circ]`.
        phi : float or ndarray, shape (...)
            Array containing the longitude in degrees.
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        source : {'all', 'external', 'internal'}, optional
            Choose source to be external (inducing), internal (induced) or
            both added (default to 'all').
        grid : bool, optional
            If ``True``, field components are computed on a regular grid,
            which is created from ``theta`` and ``phi`` as their outer product
            (defaults to ``False``).
        rc_e : ndarray, shape (...), optional
            External part of the RC-index (defaults to linearly interpolating
            the hourly values given by the built-in RC-index file).
        rc_i : ndarray, shape (...), optional
            Internal part of the RC-index (defaults to linearly interpolating
            the hourly values given by the built-in RC-index file).

        Returns
        -------
        B_radius, B_theta, B_phi : ndarray, shape (...)
            Radial, colatitude and azimuthal field components.

        Notes
        -----
        The computation of the near-magnetospheric field coefficients requires
        the RC-index. If the values of the RC-index are not supplied
        directly via the ``rc`` input argument, a built-in RC-index file is
        loaded. This file goes with a specific version of the CHAOS model,
        which can be inspected by running
        ``print(chaosmagpy.basicConfig['params.CHAOS_version'])`` in a Python
        session. It is recommended to use this version of the CHAOS model
        together with the built-in RC-index.

        For the latest CHAOS model, if necessary, a suitable RC-index file can
        be downloaded at :rc_url:`\\ `. Overwrite the use of the built-in
        RC-index file by either providing interpolated values from the
        downloaded file via the ``rc`` input argument, or by replacing
        the path in ChaosMagPy's configuration dictionary
        ``chaosmagpy.basicParams['file.RC_index']`` with the path to the
        downloaded file (for more details, see Sect.
        :ref:`sec-configuration-change-rc-index-file`).

        See Also
        --------
        CHAOS.synth_values_sm

        Examples
        --------
        >>> import chaosmagpy as cp
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> time = np.array([0., 10.])
        >>> Br, Bt, Bp = model.synth_values_sm(time, 6371.2, 45., 0.)
        >>> Br
        array([-18.85890361, -13.99893523])

        """

        source = 'all' if source is None else source
        grid = False if grid is None else grid

        if source == 'all':

            coeffs_ext = self.synth_coeffs_sm(
                time, nmax=nmax, source='external', rc=rc_e)
            coeffs_int = self.synth_coeffs_sm(
                time, nmax=nmax, source='internal', rc=rc_i)

            B_radius_ext, B_theta_ext, B_phi_ext = mu.synth_values(
                coeffs_ext, radius, theta, phi, source='external', grid=grid)
            B_radius_int, B_theta_int, B_phi_int = mu.synth_values(
                coeffs_int, radius, theta, phi, source='internal', grid=grid)

            B_radius = B_radius_ext + B_radius_int
            B_theta = B_theta_ext + B_theta_int
            B_phi = B_phi_ext + B_phi_int

        elif source in ['external', 'internal']:

            # set the RC-index
            rc = rc_e if source == 'external' else rc_i

            coeffs = self.synth_coeffs_sm(
                time, nmax=nmax, source=source, rc=rc)
            B_radius, B_theta, B_phi = mu.synth_values(
                coeffs, radius, theta, phi, source=source, grid=grid)

        else:
            raise ValueError(f'Unknown source "{source}". '
                             'Use {''all'', ''external'', ''internal''}.')

        return B_radius, B_theta, B_phi

    def plot_maps_external(self, time, radius, *, nmax=None, reference=None,
                           source=None):
        """
        Plot global map of the external field from the CHAOS model.

        Parameters
        ----------
        time : ndarray, shape (), (1,) or float
            Time given as MJD2000 (modified Julian date).
        radius : ndarray, shape (), (1,) or float
            Radius in kilometers.
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        reference : {'all', 'GSM', 'SM'}, optional
            Choose contribution either from GSM, SM or the sum of both
            (default).
        source : {'all', 'external', 'internal'}, optional
            Choose source to be external (inducing), internal (induced) or
            the sum of both (default).

        Notes
        -----
        For more customization get access to the figure and axes handles
        through matplotlib by using ``fig = plt.gcf()`` and ``axes = fig.axes``
        right after the call to this plotting method.

        """

        reference = 'all' if reference is None else reference.lower()
        source = 'all' if source is None else source.lower()

        time = np.array(time, dtype=float)
        radius = np.array(radius, dtype=float)
        theta = np.linspace(1, 179, num=320)
        phi = np.linspace(-180, 180, num=721)

        # compute GSM contribution: external, internal
        if reference == 'all' or reference == 'gsm':

            # compute magnetic field components
            B_radius_gsm, B_theta_gsm, B_phi_gsm = self.synth_values_gsm(
                time, radius, theta, phi, nmax=nmax, source=source, grid=True)

        # compute SM contribution: external, internal
        if reference == 'all' or reference == 'sm':

            B_radius_sm, B_theta_sm, B_phi_sm = self.synth_values_sm(
                time, radius, theta, phi, nmax=nmax, source=source, grid=True)

        if reference == 'all':
            B_radius = B_radius_gsm + B_radius_sm
            B_theta = B_theta_gsm + B_theta_sm
            B_phi = B_phi_gsm + B_phi_sm
        elif reference == 'gsm':
            B_radius = B_radius_gsm
            B_theta = B_theta_gsm
            B_phi = B_phi_gsm
        elif reference == 'sm':
            B_radius = B_radius_sm
            B_theta = B_theta_sm
            B_phi = B_phi_sm
        else:
            raise ValueError(f'Unknown reference "{reference}". '
                             'Use {''all'', ''external'', ''internal''}.')

        units = du.gauss_units(0)

        if reference == 'all':
            titles = [f'$B_r$ ({source} sources)',
                      f'$B_\\theta$ ({source} sources)',
                      f'$B_\\phi$ ({source} sources)']
        else:
            titles = [f'$B_r$ ({reference.upper()} {source} sources)',
                      f'$B_\\theta$ ({reference.upper()} {source} sources)',
                      f'$B_\\phi$ ({reference.upper()} {source} sources)']

        pu.plot_maps(theta, phi, B_radius, B_theta, B_phi,
                     titles=titles, label=units)

    def synth_euler_angles(self, time, satellite, *, dim=None, deriv=None,
                           extrapolate=None):
        """
        Extract Euler angles for specified satellite.

        Parameters
        ----------
        time : float or ndarray, shape (...)
            Time given as MJD2000 (modified Julian date).
        satellite : str
            Satellite from which to get the euler angles.
        dim : int, positive, optional
            Maximum dimension (default is 3, for the three angles alpha, beta,
            gamma).
        deriv : int, positive, optional
            Derivative in time (default is 0).
        extrapolate : {'linear', 'spline', 'constant', 'off'}, optional
            Extrapolate to times outside of the model bounds. Defaults to
            ``'linear'``.

        Returns
        -------
        angles : ndarray, shape (..., 3)
            Euler angles alpha, beta and gamma in degrees, stored in trailing
            dimension.

        Examples
        --------
        >>> import chaosmagpy as cp
        >>> import numpy as np
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> time = np.array([500., 600.])
        >>> model.meta['satellites']  # check satellite names
        ('oersted', 'champ', 'sac_c', 'swarm_a', 'swarm_b', 'swarm_c')

        >>> model.synth_euler_angles(time, 'champ')
        array([[-0.05521985, -1.5763316 ,  0.48787601],
               [-0.05440427, -1.57966925,  0.49043057]])

        """

        if satellite not in self.model_euler:
            string = '", "'.join([*self.model_euler.keys()])
            raise ValueError(
                        f'Unknown satellite "{satellite}". Use '
                        f'one of {{"{string}"}}.')

        coeffs = self.model_euler[satellite].synth_coeffs(
            time, dim=dim, deriv=deriv, extrapolate=extrapolate)

        # add Euler prerotation angles if available in meta
        pre = self.model_euler[satellite].meta['Euler_prerotation']
        if pre is not None:
            warnings.warn(f'Euler pre-rotation has not been applied. '
                          f'The pre-rotation angles are {pre}.')

        return coeffs

    def save_shcfile(self, filepath, *, model=None, leap_year=None):
        """
        Save spherical harmonic coefficients to a file in `shc`-format.

        Parameters
        ----------
        filepath : str
            Path and name of output file `*.shc`.
        model : {'tdep', 'static'}, optional
            Choose part of the model to save (default is 'tdep').
        leap_year : {False, True}, optional
            Take leap year in time conversion into account. By default set to
            ``False``, so that a conversion factor of 365.25 days per year is
            used.

        """

        model = 'tdep' if model is None else model

        leap_year = False if leap_year is None else leap_year

        if model == 'tdep':

            if self.model_tdep is None:
                raise ValueError("Time-dependent internal field coefficients "
                                 "are missing.")

            nmin = 1
            nmax = self.model_tdep.nmax
            order = self.model_tdep.order
            pieces = self.model_tdep.breaks.size - 1
            step = max(order - 1, 1)
            np = pieces if order == 1 else (pieces * step + 1)

            # create header lines
            header = textwrap.dedent(f"""\
                # {self.name}
                # Spherical harmonic coefficients of the time-dependent
                # internal field from degree {nmin} to {nmax}. Coefficients
                # (units of nT) are given at {np} points in time and were
                # extracted from a {order}-order piecewise polynomial. The
                # break points are every {step} step(s) of the time sequence.
                """)

            self.model_tdep.to_shc(
                filepath,
                leap_year=leap_year,
                nmin=nmin,
                header=header
            )

        # output static field model coefficients
        if model == 'static':

            if self.model_static is None:
                raise ValueError("Static internal field coefficients "
                                 "are missing.")

            nmin = self.model_tdep.nmax + 1
            nmax = self.model_static.nmax
            order = 1

            # create additonal header lines
            header = textwrap.dedent(f"""\
                # {self.name}
                # Spherical harmonic coefficients (units of nT) of the static
                # internal field model from degree {nmin} to {nmax}. Given at
                # the first break point.
                """)

            self.model_static.to_shc(
                filepath,
                leap_year=leap_year,
                nmin=nmin,
                header=header
            )

    def save_matfile(self, filepath):
        """
        Save CHAOS model to a MAT-file.

        The model must be fully specified as is the case if originally loaded
        from a MAT-file.

        Parameters
        ----------
        filepath : str
            Path and name of MAT-file that is to be saved.

        """

        # write time-dependent internal field model to matfile
        if self.model_tdep:
            self.model_tdep.save_matfile(filepath)

        if self.coeffs_delta:
            # write time-dependent external field model to matfile
            q10 = self.coeffs_delta['q10'].reshape((-1, 1))
            q11 = np.ravel(self.coeffs_delta['q11'])
            s11 = np.ravel(self.coeffs_delta['s11'])
            qs11 = np.stack((q11, s11), axis=-1)

            t_break_q10 = self.breaks_delta['q10'].reshape(
                (-1, 1)).astype(float)
            t_break_q11 = self.breaks_delta['q11'].reshape(
                (-1, 1)).astype(float)

            m_sm = np.array([np.mean(q10), np.mean(q11), np.mean(s11)])
            m_sm = np.append(m_sm, self.coeffs_sm[3:]).reshape((-1, 1))

            m_Dst = self.coeffs_sm[:3].reshape((3, 1))

            # process gsm coefficients
            m_gsm = self.coeffs_gsm[[0, 3]].reshape((2, 1))

            model_ext = dict(
                t_break_q10=t_break_q10,
                q10=q10,
                t_break_qs11=t_break_q11,
                qs11=qs11,
                m_sm=m_sm,
                m_gsm=m_gsm,
                m_Dst=m_Dst
            )

            hdf.write(model_ext, path='/model_ext', filename=filepath,
                      matlab_compatible=True)

        if self.model_euler:
            # write Euler angles to matfile for each satellite
            satellites = self.meta['satellites']

            t_break_Euler = []  # list of Euler angle breaks for satellites
            alpha = []  # list of alpha for each satellite
            beta = []  # list of beta for each satellite
            gamma = []  # list of gamma for each satellite
            for satellite in satellites:

                # reduce breaks if start and end are equal
                breaks = self.model_euler[satellite].breaks
                if breaks[0] == breaks[-1]:
                    breaks = breaks[0]

                t_break_Euler.append(breaks.reshape((-1, 1)).astype(float))
                alpha.append(self.model_euler[satellite].coeffs[
                    0, :, 0].reshape((-1, 1)).astype(float))
                beta.append(self.model_euler[satellite].coeffs[
                    0, :, 1].reshape((-1, 1)).astype(float))
                gamma.append(self.model_euler[satellite].coeffs[
                    0, :, 2].reshape((-1, 1)).astype(float))

            hdf.write(np.array(t_break_Euler, dtype='object'),
                      path='/model_Euler/t_break_Euler/',
                      filename=filepath, matlab_compatible=True)
            hdf.write(np.array(alpha, dtype='object'),
                      path='/model_Euler/alpha/',
                      filename=filepath, matlab_compatible=True)
            hdf.write(np.array(beta, dtype='object'),
                      path='/model_Euler/beta/',
                      filename=filepath, matlab_compatible=True)
            hdf.write(np.array(gamma, dtype='object'),
                      path='/model_Euler/gamma/',
                      filename=filepath, matlab_compatible=True)

        if self.model_cal:
            # write calibration parameters to matfile for each satellite
            cal = []
            for satellite, model in self.model_cal.items():
                cal.append(model.to_ppdict())

            hdf.write(np.array(cal, dtype=object),
                      path='/pp_CAL/',
                      filename=filepath, matlab_compatible=True)

        if self.model_static:
            # write static internal field model to matfile
            g = np.ravel(self.model_static.coeffs).reshape((-1, 1))

            hdf.write(g, path='/g', filename=filepath, matlab_compatible=True)

        if self.meta:
            hdf.write(self.meta['params'], path='/params', filename=filepath,
                      matlab_compatible=True)

        print('CHAOS saved to {}.'.format(
            os.path.join(os.getcwd(), filepath)))

    @classmethod
    def from_mat(self, filepath, name=None, satellites=None):
        """
        Alternative constructor for creating a :class:`CHAOS` class instance.

        Parameters
        ----------
        filepath : str
            Path to MAT-file containing the CHAOS model.
        name : str, optional
            User defined name of the model. Defaults to the filename without
            the file extension.
        satellites : list of strings, optional
            List of satellite names whose Euler angles are stored in the
            MAT-file. This is needed for correct referencing as this
            information is not given in the standard CHAOS MAT-file format
            (defaults to ``['oersted', 'champ', 'sac_c', 'swarm_a', 'swarm_b',
            'swarm_c', 'cryosat-2_1', 'cryosat-2_2', 'cryosat-2_3']``.)

        Returns
        -------
        model : :class:`CHAOS` instance
            Fully initialized model.

        Examples
        --------
        Load for example the MAT-file ``CHAOS-6-x7.mat`` in the current working
        directory like this:

        >>> from chaosmagpy import CHAOS
        >>> model = CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> print(model)

        See Also
        --------
        load_CHAOS_matfile

        """

        return load_CHAOS_matfile(filepath, name=name, satellites=satellites)

    @classmethod
    def from_shc(self, filepath, *, name=None, leap_year=None):
        """
        Alternative constructor for creating a :class:`CHAOS` class instance.

        Parameters
        ----------
        filepath : str
            Path to SHC-file.
        name : str, optional
            User defined name of the model. Defaults to ``'CHAOS-<version>'``,
            where <version> is the default in
            ``basicConfig['params.version']``.
        leap_year : {False, True}, optional
            Take leap year in time conversion into account. By default set to
            ``False``, so that a conversion factor of 365.25 days per year is
            used.

        Returns
        -------
        model : :class:`CHAOS` instance
            Class instance with either the time-dependent or static internal
            part initialized.

        Examples
        --------
        Load for example the SHC-file ``CHAOS-6-x7_tdep.shc`` in the current
        working directory, containing the coefficients of time-dependent
        internal part of the CHAOS-6-x7 model.

        >>> import chaosmagpy as cp
        >>> model = cp.CHAOS.from_shc('CHAOS-6-x7_tdep.shc')
        >>> print(model)

        See Also
        --------
        load_CHAOS_shcfile

        """

        leap_year = False if leap_year is None else leap_year

        return load_CHAOS_shcfile(filepath, name=name, leap_year=leap_year)


def load_CHAOS_matfile(filepath, name=None, satellites=None):
    """
    Load CHAOS model from MAT-file.

    Parameters
    ----------
    filepath : str
        Path to MAT-file containing the CHAOS model.
    name : str, optional
        User defined name of the model. Defaults to the filename without the
        file extension.
    satellites : list of strings, optional
        List of satellite names whose Euler angles are stored in the MAT-file.
        This is needed for correct referencing as this information is not
        given in the standard CHAOS MAT-file format (defaults to
        ``['oersted', 'champ', 'sac_c', 'swarm_a', 'swarm_b', 'swarm_c',
        'cryosat-2_1', 'cryosat-2_2', 'cryosat-2_3']``.)

    Returns
    -------
    model : :class:`CHAOS` instance
        Fully initialized model.

    Examples
    --------
    Load for example the MAT-file ``CHAOS-6-x7.mat`` in the current working
    directory like this:

    >>> import chaosmagpy as cp
    >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
    >>> print(model)

    Compute the Gauss coefficients of the time-dependent internal field on
    January 1, 2018 at 0:00 UTC. First, convert the date to modified Julian
    date:

    >>> cp.data_utils.mjd2000(2018, 1, 1)
    6575.0

    Now, compute the Gauss coefficients:

    >>> coeffs = model.synth_coeffs_tdep(6575.)
    >>> coeffs
    array([-2.94172133e+04, -1.46670696e+03, ..., -8.23461504e-02])

    Compute only the dipolar part by restricting the spherical harmonic degree
    with the help of the ``nmax`` keyword argument:

    >>> coeffs = model.synth_coeffs_tdep(6575., nmax=1)
    >>> coeffs
    array([-29417.21325337,  -1466.70696158,   4705.96297947])

    Compute the first time derivative of the internal Gauss coefficients in
    nT/yr with the help of the ``deriv`` keyword argument:

    >>> coeffs = model.synth_coeffs_tdep(6575., deriv=1)  # nT/yr
    >>> coeffs
    array([ 6.45476360e+00,  8.56693199e+00, ..., 1.13856347e-03])

    Compute values of the time-dependent internal magnetic field on
    January 1, 2018 at 0:00 UTC at
    :math:`(\\theta, \\phi) = (90^\\circ, 0^\\circ)` on Earth's surface
    (:math:`r=6371.2` km):

    >>> B_radius, B_theta, B_phi = model.synth_values_tdep(\
6575., 6371.2, 90., 0.)
    >>> B_theta
    array(-27642.31732596)

    See Also
    --------
    CHAOS, load_CHAOS_shcfile

    """

    filepath = str(filepath)

    if name is None:
        # get name without extension
        name = os.path.splitext(os.path.basename(filepath))[0]

    # define satellite names
    if satellites is None:
        satellites = ['oersted', 'champ', 'sac_c', 'swarm_a', 'swarm_b',
                      'swarm_c', 'cryosat-2_1', 'cryosat-2_2', 'cryosat-2_3']

    mat_contents = du.load_matfile(filepath)

    pp = mat_contents['pp']

    order = int(pp['order'])
    pieces = int(pp['pieces'])
    dim = int(pp['dim'])
    breaks = pp['breaks']
    coefs = pp['coefs']

    # reshaping coeffs_tdep from 2-D to 3-D: (order, pieces, coefficients)
    coeffs_tdep = coefs.transpose().reshape((order, pieces, dim))

    # load the static internal field model
    try:
        coeffs_static = mat_contents['g'].reshape((1, 1, -1))

    except KeyError as err:
        warnings.warn(f'Missing static internal field coefficients: {err}')
        coeffs_static = None

    # load the external field model
    try:
        model_ext = mat_contents['model_ext']

    except KeyError as err:
        warnings.warn(f'Missing external field coefficients: {err}')
        coeffs_delta = None
        breaks_delta = None
        coeffs_gsm = None
        coeffs_sm = None

    else:
        # external field (SM): n=1, 2 (n=1 are sm offset time averages!)
        coeffs_sm = np.copy(model_ext['m_sm'])
        coeffs_sm[:3] = model_ext['m_Dst']  # replace with m_Dst

        # external field (GSM): n=1, 2
        n_gsm = int(2)
        coeffs_gsm = np.zeros((n_gsm*(n_gsm+2),))  # correct number of coeffs
        # only m=0 are non-zero
        coeffs_gsm[[0, 3]] = model_ext['m_gsm']

        # coefficients and breaks of external SM offsets for q10, q11, s11
        breaks_delta = dict()
        breaks_delta['q10'] = model_ext['t_break_q10']
        breaks_delta['q11'] = model_ext['t_break_qs11']
        breaks_delta['s11'] = model_ext['t_break_qs11']

        # reshape to comply with scipy PPoly coefficients
        coeffs_delta = dict()
        coeffs_delta['q10'] = model_ext['q10'].reshape((1, -1))
        qs11 = model_ext['qs11']
        coeffs_delta['q11'] = qs11[:, 0].reshape((1, -1))
        coeffs_delta['s11'] = qs11[:, 1].reshape((1, -1))

    # load euler angles
    try:
        model_euler = mat_contents['model_Euler']

    except KeyError as err:
        warnings.warn(f'Missing Euler angles: {err}')
        coeffs_euler = None
        breaks_euler = None

    else:
        # append generic satellite name if more data available or reduce
        t_break_Euler = model_euler['t_break_Euler']
        n = len(t_break_Euler)
        m = len(satellites)

        if n < m:
            satellites = satellites[:n]
        elif n > m:
            for counter in range(m+1, n+1):
                satellites.append(f'satellite_{counter}')

        # coefficients and breaks of euler angles
        breaks_euler = dict()
        for num, satellite in enumerate(satellites):
            breaks_euler[satellite] = t_break_Euler[num].squeeze()

        coeffs_euler = dict()
        for num, satellite in enumerate(satellites):

            angles = [model_euler[angle][num] for angle in [
                'alpha', 'beta', 'gamma']]
            euler = np.concatenate(angles, axis=-1).astype(float)

            # first: order, second: number of intervals, third: 3 angles
            coeffs_euler[satellite] = np.expand_dims(euler, axis=0)

    # load calibration parameters
    try:
        model_cal = mat_contents['pp_CAL']

    except KeyError as err:
        warnings.warn(f'Missing calibration parameters: {err}')
        breaks_cal = None
        coeffs_cal = None

    else:
        # only support of single satellite
        breaks_cal = {'cryosat-2_1': model_cal['breaks']}
        coeffs_cal = {'cryosat-2_1': model_cal['coefs'].reshape((1, -1, 9))}

    # load additional parameters
    try:
        params = mat_contents['params']
        dict_params = {'Euler_prerotation': params['Euler_prerotation']}

    except KeyError as err:
        warnings.warn(f'Missing params dictionary of Euler prerotation: {err}')
        dict_params = {'Euler_prerotation': None}

    meta = dict(params=dict_params,
                satellites=tuple(satellites))

    model = CHAOS(breaks=breaks,
                  order=order,
                  coeffs_tdep=coeffs_tdep,
                  coeffs_static=coeffs_static,
                  coeffs_sm=coeffs_sm,
                  coeffs_gsm=coeffs_gsm,
                  breaks_delta=breaks_delta,
                  coeffs_delta=coeffs_delta,
                  breaks_euler=breaks_euler,
                  coeffs_euler=coeffs_euler,
                  breaks_cal=breaks_cal,
                  coeffs_cal=coeffs_cal,
                  name=name,
                  meta=meta)

    return model


def load_CHAOS_shcfile(filepath, name=None, leap_year=None):
    """
    Load CHAOS model from SHC-file.

    The file should contain the coefficients of the time-dependent or static
    internal part of the CHAOS model. In case of the time-dependent part, a
    reconstruction of the piecewise polynomial is performed.

    Parameters
    ----------
    filepath : str
        Path to SHC-file.
    name : str, optional
        User defined name of the model. Defaults to the filename without the
        file extension.
    leap_year : {False, True}, optional
        Take leap year in time conversion into account. By default set to
        ``False``, so that a conversion factor of 365.25 days per year is
        used.

    Returns
    -------
    model : :class:`CHAOS` instance
        Class instance with either the time-dependent or static internal part
        initialized.

    Examples
    --------
    Load for example the SHC-file ``CHAOS-6-x7_tdep.shc`` in the current
    working directory, containing the coefficients of time-dependent internal
    part of the CHAOS-6-x7 model.

    >>> import chaosmagpy as cp
    >>> model = cp.load_CHAOS_shcfile('CHAOS-6-x7_tdep.shc')
    >>> print(model)

    Compute the Gauss coefficients of the time-dependent internal field on
    January 1, 2018 at 0:00 UTC. First, convert the date to modified Julian
    date:

    >>> cp.data_utils.mjd2000(2018, 1, 1)
    6575.0

    Now, compute the Gauss coefficients:

    >>> coeffs = model.synth_coeffs_tdep(6575.)
    >>> coeffs
    array([-2.94172133e+04, -1.46670696e+03, ..., -8.23461504e-02])

    Compute only the dipolar part by restricting the spherical harmonic degree
    with the help of the ``nmax`` keyword argument:

    >>> coeffs = model.synth_coeffs_tdep(6575., nmax=1)
    >>> coeffs
    array([-29417.21325337,  -1466.70696158,   4705.96297947])

    Compute the first time derivative of the internal Gauss coefficients in
    nT/yr with the help of the ``deriv`` keyword argument:

    >>> coeffs = model.synth_coeffs_tdep(6575., deriv=1)  # nT/yr
    >>> coeffs
    array([ 6.45476360e+00,  8.56693199e+00, ..., 1.13856347e-03])

    Compute values of the time-dependent internal magnetic field on
    January 1, 2018 at 0:00 UTC at
    :math:`(\\theta, \\phi) = (90^\\circ, 0^\\circ)` on Earth's surface
    (:math:`r=6371.2` km):

    >>> B_radius, B_theta, B_phi = model.synth_values_tdep(\
6575., 6371.2, 90., 0.)
    >>> B_theta
    array(-27642.31732596)

    See Also
    --------
    CHAOS, load_CHAOS_matfile

    """

    if name is None:
        # get name without extension
        name = os.path.splitext(os.path.basename(filepath))[0]

    leap_year = False if leap_year is None else bool(leap_year)

    # create dummy BaseModel since there is no entry point to directly modify
    # CHAOS.model_tdep or CHAOS.model_static at the moment
    model = BaseModel.from_shc(filepath, name=name, leap_year=leap_year)

    if (model.pieces == 1) and (model.order == 1):  # static in single bin

        return CHAOS(
            breaks=model.breaks,
            coeffs_static=model.coeffs,
            name=model.name
        )

    else:  # must be time-dependent, incl. piecewise constant

        return CHAOS(
            breaks=model.breaks,
            order=model.order,
            coeffs_tdep=model.coeffs,
            name=model.name
        )


def load_CovObs_txtfile(filepath, name=None):
    """
    Load the ensemble mean of the internal model from the COV-OBS
    TXT-file in spline format.

    Parameters
    ----------
    filepath : str
        Path to spline-formatted TXT-file (not part of ChaosMagPy).
    name : str, optional
        User defined name of the model. Defaults to the filename without the
        file extension.

    Returns
    -------
    model : :class:`BaseModel`
        Class :class:`BaseModel` instance.

    References
    ----------
    For details on the COV-OBS model (COV-OBS.x2), see the original
    publication:

    Huder, L., Gillet, N., Finlay, C. C., Hammer, M. D. and H. Tchoungui
    (2020), "COV-OBS.x2: 180 yr of geomagnetic field evolution from
    ground-based and satellite observations", Earth, Planets and Space.

    Examples
    --------
    Load the ensemble mean internal model and plot the degree-1 secular
    variation. Here, the model parameter file, e.g. "COV-OBS.x2-int", is in
    the current working directory.

    .. code-block:: python

        import chaosmagpy as cp
        import matplotlib.pyplot as plt
        import numpy as np

        model = cp.load_CovObs_txtfile('COV-OBS.x2-int')  # load model TXT-file

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # sample model timespan
        time = np.linspace(1840., 2020., 1000)  # decimal years
        mjd = cp.data_utils.dyear_to_mjd(time, leap_year=False)

        coeffs = model.synth_coeffs(mjd, nmax=1, deriv=1)

        ax.plot(cp.data_utils.timestamp(time), coeffs)
        ax.set_title(model.name)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('nT/yr')

        ax.legend(['$g_1^0$', '$g_1^1$', '$h_1^1$'])

        plt.tight_layout()
        plt.show()

    """

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if name is None:
        # get name without extension
        name = os.path.splitext(os.path.basename(filepath))[0]

    tmp = np.fromstring(lines[1], sep=' ')  # read parameters and breaks

    nmax = int(tmp[0])
    order = int(tmp[2])
    degree = order - 1  # polynomial degree

    # convert decimal year to modified Julian date (using 365.25 days/year)
    breaks = du.dyear_to_mjd(tmp[3:], leap_year=False)

    # add endpoint multiplicity to conform with scipy's BSpline routine
    knots = mu.augment_breaks(breaks, order)

    data = np.fromstring(' '.join(lines[2:]), sep=' ')

    # add zeros at endpoints to match manually extended knots
    coeffs = np.zeros((knots.size - order, nmax * (nmax + 2)))

    # insert actual coefficients, endpoint coefficients are now zero
    coeffs[degree:-degree, :] = data.reshape(
        (breaks.size - order, nmax * (nmax + 2)))

    return BaseModel.from_bspline(name, knots, coeffs, order,
                                  source='internal')


def load_gufm1_txtfile(filepath, name=None):
    """
    Load model parameter file of the gufm1 model.

    Parameters
    ----------
    filepath : str
        Path to TXT-file (provided by the modellers).
    name : str, optional
        User defined name of the model. Defaults to the filename without the
        file extension.

    Returns
    -------
    model : :class:`BaseModel`
        Class :class:`BaseModel` instance.

    References
    ----------
    For details on the gufm1 model, see the original publication:

    Andrew Jackson, Art R. T. Jonkers and Matthew R. Walker (2000),
    "Four centuries of geomagnetic secular variation from historical records",
    Phil. Trans. R. Soc. A.358957–990, http://doi.org/10.1098/rsta.2000.0569

    Examples
    --------
    Load the model and plot the degree-1 secular variation. Here, the
    model parameter file, e.g. "gufm1", is in the current working directory.

    .. code-block:: python

        import chaosmagpy as cp
        import matplotlib.pyplot as plt
        import numpy as np

        model = cp.load_gufm1_txtfile('gufm1')  # load model TXT-file

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # sample model timespan 7.5 years away from endpoints
        time = np.linspace(1590., 1990., 1000)  # decimal years
        mjd = cp.data_utils.dyear_to_mjd(time, leap_year=False)

        coeffs = model.synth_coeffs(mjd, nmax=1, deriv=1)

        ax.plot(time, coeffs)
        ax.set_title(model.name)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('nT/yr')

        ax.legend(['$g_1^0$', '$g_1^1$', '$h_1^1$'])

        plt.tight_layout()
        plt.show()

    """

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if name is None:
        # get name without extension
        name = os.path.splitext(os.path.basename(filepath))[0]

    data = np.fromstring('  '.join(lines[1:]), sep=' ')

    nmax = int(data[0])
    order = 4  # hard-coded since not really provided in file
    nbreaks = int(data[1]) + order  # data[1] is number of B-spline functions?
    degree = order - 1  # polynomial degree

    # convert decimal year to modified Julian date (using 365.25 days/year)
    breaks = du.dyear_to_mjd(data[2:(nbreaks + 2)], leap_year=False)

    # add endpoint multiplicity to "trick" scipy's BSpline routine
    knots = mu.augment_breaks(breaks, order)

    # add zeros at endpoints to match manually extended knots
    coeffs = np.zeros((knots.size - order, nmax * (nmax + 2)))

    # insert actual coefficients, endpoint coefficients are now zero
    coeffs[degree:-degree, :] = data[(nbreaks + 2):].reshape(
        (breaks.size - order, nmax * (nmax + 2)))

    return BaseModel.from_bspline(name, knots, coeffs, order,
                                  source='internal')


def load_CALS7K_txtfile(filepath, name=None):
    """
    Load the model parameter file of the CALS7K model.

    Parameters
    ----------
    filepath : str
        Path to TXT-file (available from the modellers).
    name : str, optional
        User defined name of the model. Defaults to the filename.

    Returns
    -------
    model : :class:`BaseModel`
        Class :class:`BaseModel` instance.

    References
    ----------
    More information about the model and a list of relevant publications can be
    found at `<https://igppweb.ucsd.edu/~cathy/Projects/Holocene/CALS7K/>`_.
    The model coefficients file can be downloaded from
    `<https://earthref.org/ERDA/413/>`_.

    Examples
    --------

    .. code-block:: python

        import chaosmagpy as cp
        import matplotlib.pyplot as plt
        import numpy as np

        model = cp.load_CALS7K_txtfile('CALS7K.2')  # load model TXT-file

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        time = np.linspace(-5000., 1950., 1000)  # decimal years
        mjd = cp.data_utils.dyear_to_mjd(time, leap_year=False)

        coeffs = model.synth_coeffs(mjd, nmax=1)

        ax.plot(time, coeffs)
        ax.set_title(model.name)
        ax.set_xlabel('years')
        ax.set_ylabel('nT')

        ax.legend(['$g_1^0$', '$g_1^1$', '$h_1^1$'])

        plt.tight_layout()
        plt.show()

    """

    with open(filepath, 'r') as f:
        lines = f.read()  # read everything as a continuous string

    if name is None:
        # get name with extension
        name = os.path.basename(filepath)

    # numpy doesn't understand "D" in decimal representation, replace with E
    data = np.fromstring(lines.replace('D', 'E'), sep=' ', dtype=float)

    # ts = data[0]  # start time
    # te = data[1]  # end time
    order = data[2]  # cubic B-splines
    # discard index 3
    nmax = int(data[4])  # maximum spherical harmonic degree
    # discard index 5
    inspl = int(data[6])  # number of inner B-spline segments

    dim = nmax*(nmax+2)  # number of spherical harmonic coefficients
    order = 4  # cubic splines
    degree = order - 1  # polynomial degree
    nbreaks = inspl + order  # number of breaks

    breaks = data[7:(7+nbreaks)]
    # convert decimal year to modified Julian date (using 365.25 days/year)
    breaks = du.dyear_to_mjd(breaks, leap_year=False)

    # add endpoint multiplicity to "trick" scipy's BSpline routine
    knots = mu.augment_breaks(breaks, order)

    # add zeros at endpoints to match manually extended knots
    coeffs = np.zeros((knots.size - order, dim))

    # insert actual coefficients, endpoint coefficients are now zero
    coeffs[degree:-degree, :] = data[(7+nbreaks):].reshape(
        (breaks.size - order, dim))

    return BaseModel.from_bspline(name, knots, coeffs, order,
                                  source='internal')


def load_IGRF_txtfile(filepath, name=None):
    """
    Load the IGRF internal field model from the TXT-file with the
    piecewise-polynomial coefficients.

    Parameters
    ----------
    filepath : str
        Path to the IGRF TXT-file (not part of ChaosMagPy).
    name : str, optional
        User defined name of the model. Defaults to the filename without the
        file extension.

    Returns
    -------
    model : :class:`BaseModel`
        Class :class:`BaseModel` instance.

    References
    ----------
    The latest IGRF coefficients can be downloaded at
    `<https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field>`_.

    Notes
    -----
    The field is linearly extrapolated in the 5-year period at the
    end of the model time interval (i.e. in the period 2020-2025 for IGRF-13)
    using the predictive secular variation in the IGRF.

    Examples
    --------

    .. code-block:: python

        import chaosmagpy as cp
        import matplotlib.pyplot as plt
        import numpy as np

        model = cp.load_IGRF_txtfile('irgf13coeffs.txt')  # load model TXT-file

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        time = np.linspace(1900., 2020., 1000)  # decimal years
        mjd = cp.data_utils.dyear_to_mjd(time, leap_year=True)

        coeffs = model.synth_coeffs(mjd, nmax=1)

        ax.plot(time, coeffs)
        ax.set_title(model.name)
        ax.set_xlabel('years')
        ax.set_ylabel('nT')

        ax.legend(['$g_1^0$', '$g_1^1$', '$h_1^1$'])

        plt.tight_layout()
        plt.show()

    """

    if name is None:
        # get name without extension
        name = os.path.splitext(os.path.basename(filepath))[0]

    first_line = True
    data = np.array([])

    with open(filepath, 'r') as f:
        for line in f.readlines():

            if line.lstrip().startswith(('#', 'c/s')):
                continue

            tmp = line.split()  # also splits consecutive whitespaces

            if first_line:  # first non-comment line contains breaks
                breaks = np.array(tmp[3:-1], dtype=float)
                breaks = np.append(breaks, breaks[-1] + 5.)

                first_line = False

            else:
                newline = np.array(tmp[3:], dtype=float)  # strip degree/order
                data = np.append(data, newline)

    data = np.reshape(data, (-1, breaks.size))
    data[:, -1] = 5*data[:, -1] + data[:, -2]  # sv is in the last column
    coeffs = data.T

    order = 2
    breaks_mjd = du.dyear_to_mjd(breaks, leap_year=True)
    knots = mu.augment_breaks(breaks_mjd, order)

    model = BaseModel.from_bspline(name, knots, coeffs, order,
                                   source='internal')

    return model
