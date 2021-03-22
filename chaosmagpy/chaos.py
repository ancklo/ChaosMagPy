"""
Classes and functions to read the CHAOS model.

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

"""

import numpy as np
import os
import warnings
import scipy.interpolate as sip
import hdf5storage as hdf
import h5py
import chaosmagpy.coordinate_utils as cu
import chaosmagpy.model_utils as mu
import chaosmagpy.data_utils as du
import matplotlib.pyplot as plt
from datetime import datetime
from timeit import default_timer as timer
from chaosmagpy.config_utils import basicConfig
from chaosmagpy.plot_utils import (defaultkeys, plot_power_spectrum,
                                   plot_timeseries, plot_maps)


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
            Specify polynomial degree as string or any order as integer.
            Defaults to ``'linear'`` (equiv. to order 2 polynomials).

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

        if np.amin(time) < start or np.amax(time) > end:
            if isinstance(extrapolate, str):  # convert string to integer
                dkey = {'linear': 2,
                        'quadratic': 3,
                        'cubic': 4,
                        'constant': 1,
                        'spline': self.order,
                        'off': 0}
                try:
                    key = min(dkey[extrapolate], self.order)
                except KeyError:
                    string = '", "'.join(list(dkey.keys()))
                    raise ValueError(
                        f'Unknown extrapolation method "{extrapolate}". Use '
                        f'one of {{"{string}"}}.')

            else:
                key = min(extrapolate, self.order)

            message = 'no' if extrapolate == 'off' else extrapolate
            warnings.warn("Requested coefficients are "
                          "outside of the modelled period from "
                          f"{start} to {end}. Doing {message} extrapolation.")

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

    def save_matfile(self, filepath, path=None):
        """
        Save piecewise polynomial as mat-file.

        Parameters
        ----------
        filepath : str
            Filepath and name of mat-file.
        path : str
            Location in mat-file. Defaults to ``'/pp'``.

        """

        if path is None:
            path = '/pp'

        dim = self.dim
        coeffs = self.coeffs
        coefs = coeffs.reshape((self.order, -1)).transpose()

        pp = dict(
            form='pp',
            order=self.order,
            pieces=self.pieces,
            dim=dim,
            breaks=self.breaks.reshape((1, -1)),  # ensure 2d
            coefs=coefs)

        hdf.write(pp, path=path, filename=filepath, matlab_compatible=True)


class BaseModel(Base):
    """
    Class for piecewise polynomial spherical harmonic model.

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
            Radius in kilometers (defaults to mean Earth's surface defined in
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

        radius = basicConfig['params.r_surf'] if radius is None else radius

        coeffs = self.synth_coeffs(time, **kwargs)

        return mu.power_spectrum(coeffs, radius)

    def plot_power_spectrum(self, time, **kwargs):
        """
        Plot the spatial power spectrum.

        Parameters
        ----------
        time : float
            Time in modified Julian date.

        See Also
        --------
        BaseModel.power_spectrum

        """

        defaults = dict(radius=None,
                        deriv=0,
                        nmax=self.nmax,
                        titles='spatial power spectrum')

        kwargs = defaultkeys(defaults, kwargs)

        radius = kwargs.pop('radius')
        nmax = kwargs.pop('nmax')
        deriv = kwargs.pop('deriv')

        units = f'({du.gauss_units(deriv)})$^2$'
        kwargs.setdefault('ylabel', units)

        R_n = self.power_spectrum(time, radius, nmax=nmax, deriv=deriv)

        plot_power_spectrum(R_n, **kwargs)
        plt.show()

    def plot_maps(self, time, radius, **kwargs):
        """
        Plot global maps of the field components.

        Parameters
        ----------
        time : ndarray, shape (), (1,) or float
            Time given as MJD2000 (modified Julian date).
        radius : ndarray, shape (), (1,) or float
            Array containing the radius in kilometers.

        Returns
        -------
        B_radius, B_theta, B_phi
            Global map of the radial, colatitude and azimuthal field
            components.

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

        See Also
        --------
        chaosmagpy.plot_utils.plot_maps

        """

        defaults = dict(deriv=0,
                        nmax=self.nmax)

        kwargs = defaultkeys(defaults, kwargs)

        # remove keywords that are not intended for pcolormesh
        nmax = kwargs.pop('nmax')
        deriv = kwargs.pop('deriv')
        titles = [f'$B_r$ ($n\\leq{nmax}$, deriv={deriv})',
                  f'$B_\\theta$ ($n\\leq{nmax}$, deriv={deriv})',
                  f'$B_\\phi$ ($n\\leq{nmax}$, deriv={deriv})']

        # add plot_maps options to dictionary
        kwargs.setdefault('label', du.gauss_units(deriv))
        kwargs.setdefault('titles', titles)

        # handle optional argument: nmax > coefficent nmax
        if nmax > self.nmax:
            warnings.warn(
                'Supplied nmax = {0} is incompatible with number of model '
                'coefficients. Using nmax = {1} instead.'.format(
                    nmax, self.nmax))
            nmax = self.nmax

        time = np.array(time, dtype=float)
        theta = np.linspace(1, 179, num=320)
        phi = np.linspace(-180, 180, num=721)

        B_radius, B_theta, B_phi = self.synth_values(
            time, radius, theta, phi, nmax=nmax, deriv=deriv,
            grid=True, extrapolate=None)

        plot_maps(theta, phi, B_radius, B_theta, B_phi, **kwargs)
        plt.show()

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

        Returns
        -------
        B_radius, B_theta, B_phi
            Time series plot of the radial, colatitude and azimuthal field
            components.

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

        See Also
        --------
        chaosmagpy.plot_utils.plot_timeseries

        """

        defaults = dict(deriv=0,
                        nmax=self.nmax,
                        titles=['$B_r$', '$B_\\theta$', '$B_\\phi$'],
                        extrapolate=None)

        kwargs = defaultkeys(defaults, kwargs)

        # remove keywords that are not intended for pcolormesh
        nmax = kwargs.pop('nmax')
        deriv = kwargs.pop('deriv')
        extrapolate = kwargs.pop('extrapolate')

        # add plot_maps options to dictionary
        kwargs.setdefault('ylabel', du.gauss_units(deriv))

        time = np.linspace(self.breaks[0], self.breaks[-1], num=500)

        B_radius, B_theta, B_phi = self.synth_values(
            time, radius, theta, phi, nmax=nmax, deriv=deriv,
            extrapolate=extrapolate)

        plot_timeseries(time, B_radius, B_theta, B_phi, **kwargs)
        plt.show()

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

        degree = order - 1
        breaks = np.unique(knots)  # remove endpoint multiplicity
        pieces = breaks.size - 1  # number of polynomial pieces
        dim = coeffs.shape[-1]  # dimension = number of Gauss coefficients

        coeffs_pp = np.zeros((order, pieces, dim))

        # have to do it manually for each dimension, scipy only univariate
        for d in range(dim):

            bs = sip.BSpline(knots, coeffs[:, d], degree)
            pp = sip.PPoly.from_spline(bs, extrapolate=False)

            # remove multiple pp-coefficients at endpoints
            coeffs_pp[:, :, d] = pp.c[:, degree:(degree+pieces)]

        return cls(name, breaks=breaks, order=order, coeffs=coeffs_pp,
                   source=source, meta=meta)

    @classmethod
    def from_shc(cls, filepath, *, name=None, leap_year=None,
                 source=None, meta=None):
        """
        Return BaseModel instance by loading a model from an shc-file.

        Parameters
        ----------
        filepath : str
            Path to shc-file.
        name : str, optional
            User defined name of the model. Defaults to the filename without
            the file extension.
        leap_year : {True, False}, optional
            Take leap year in time conversion into account (default).
            Otherwise, use conversion factor of 365.25 days per year.
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
        leap_year = True if leap_year is None else leap_year

        time, coeffs, params = du.load_shcfile(filepath, leap_year=leap_year)

        nmin = params['nmin']
        nmax = params['nmax']
        order = params['order']
        step = order - 1  # actually params['step'], but not reliable

        # make single piece by setting both ends equal
        if time.size == 1:
            breaks = np.append(time, time)
        else:
            breaks = time[::step]

        # need to pad coefficients
        if nmin > 1:
            data = np.zeros((nmax*(nmax+2),) + coeffs.shape[1:])
            data[int(nmin**2-1):, ...] = coeffs
        else:
            data = coeffs

        pieces = breaks.size - 1  # number of polynomial pieces

        if (pieces == 1) and (order == 1):  # static field in single bin
            coeffs_out = data.reshape((1, 1, -1))

        else:  # model must be time-dependent (incl. piecewise constant)

            # interpolate with piecewise polynomial of given order
            coeffs_out = np.empty((order, pieces, coeffs.shape[0]))
            for m, left_break in enumerate(breaks[:-1]):
                left = m * step  # time index of left_break
                x = time[left:left+order] - left_break
                c = np.linalg.solve(
                    np.vander(x, order), coeffs[:, left:left+order].T)

                for k in range(order):
                    coeffs_out[k, m] = c[k]

        return cls(name, breaks=breaks, order=order, coeffs=coeffs_out,
                   source=source, meta=meta)


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
        ``'sac_c'``, ``'swarm_a'``, ``'swarm_b'``, ``'swarm_c'``).
    coeffs_euler : dict with ndarrays, shape (1, :math:`m_e`, 3)
        Dictionary containing satellite name as key and arrays of the Euler
        angles alpha, beta and gamma as trailing dimension (keys are
        ``'oersted'``, ``'champ'``, ``'sac_c'``, ``'swarm_a'``, ``'swarm_b'``,
        ``'swarm_c'``).
    name : str, optional
        User defined name of the model. Defaults to ``'CHAOS-<version>'``,
        where <version> is the version in ``basicConfig['params.version']``.
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
    Load for example the mat-file ``CHAOS-6-x7.mat`` in the current working
    directory like this:

    >>> import chaosmagpy as cp
    >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
    >>> print(model)

    For more examples, see the dcoumentation of the methods below.

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
        name=None,
        meta=None
    ):
        """
        Initialize the CHAOS model.

        """

        self.timestamp = str(datetime.utcnow())

        # time-dependent internal field
        self.model_tdep = BaseModel('model_tdep', breaks, order, coeffs_tdep)
        self.model_static = BaseModel('model_static', breaks[[0, -1]], 1,
                                      coeffs_static)

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
            satellites = None
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

        # give the model a name: CHAOS-x.x or user input
        if name is None:
            self.name = f"CHAOS-{basicConfig['params.version']}"
        else:
            self.name = name

        self.meta = meta

    def __call__(self, time, radius, theta, phi, source_list=None,
                 verbose=None):
        """
        Calculate the magnetic field of all sources from the CHAOS model.

        All sources means the time-dependent and static internal field, the
        external SM/GSM fields both including and induced parts.

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
        source_list : list, ['tdep', 'static', 'gsm', 'sm'] or \
str, {'internal', 'external'}
            Specify sources in any order. Default is all sources. Instead of a
            list, pass ``source_list='internal'`` which is equivalent to
            ``source_list=['tdep', 'static']`` (internal sources) or
            ``source_list='external'`` which is the same as
            ``source_list=['gsm', 'sm']`` (external sources including induced
            part).
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

            nmax_static = 85

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
                time, radius, theta, phi, source='all')

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

        string = (f"This is {self.name} initialized on {self.timestamp} UTC.")

        return string

    def synth_coeffs_tdep(self, time, *, nmax=None, **kwargs):
        """
        Compute the time-dependent internal field coefficients from the CHAOS
        model.

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
        >>> time = np.linspace(0., 10., num=2)

        >>> model.synth_coeffs_tdep(time, nmax=1)  # dipole coefficients
        array([[-29614.72797782,  -1728.47079907,   5185.50518939],
               [-29614.33800306,  -1728.13680075,   5184.89196286]])

        >>> model.synth_coeffs_tdep(time, nmax=1, deriv=1)  # SV coefficients
        array([[ 14.25577646,  12.20214856, -22.43412895],
               [ 14.2317297 ,  12.19625726, -22.36146885]])

        """

        return self.model_tdep.synth_coeffs(time, nmax=nmax, **kwargs)

    def synth_values_tdep(self, time, radius, theta, phi, *, nmax=None,
                          deriv=None, grid=None, extrapolate=None):
        """
        Compute magnetic components of the internal
        time-dependent field.

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
        >>> time = np.linspace(0., 10., num=2)

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

        return self.model_tdep.synth_values(
            time, radius, theta, phi, nmax=nmax, deriv=deriv, grid=grid,
            extrapolate=extrapolate)

    def plot_timeseries_tdep(self, radius, theta, phi, **kwargs):
        """
        Plot the time series of the time-dependent internal field from the
        CHAOS model at a specific location.

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

        Returns
        -------
        B_radius, B_theta, B_phi
            Time series plot of the radial, colatitude and azimuthal field
            components.

        """

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

        Returns
        -------
        B_radius, B_theta, B_phi
            Global map of the radial, colatitude and azimuthal field
            components.

        """

        self.model_tdep.plot_maps(time, radius,
                                  nmax=nmax, deriv=deriv, **kwargs)

    def synth_coeffs_static(self, *, nmax=None, **kwargs):
        """
        Compute the static internal field coefficients from the CHAOS model.

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
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> model.synth_coeffs_static(nmax=50)
        array([ 0.     , 0.     ,  0.     , ...,  0.01655, -0.06339,  0.00715])

        """

        time = self.model_static.breaks[0]
        return self.model_static.synth_coeffs(time, nmax=nmax, **kwargs)

    def synth_values_static(self, radius, theta, phi, *, nmax=None, **kwargs):
        """
        Compute magnetic field components of the internal static field.

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

        Returns
        -------
        B_radius, B_theta, B_phi
            Global map of the radial, colatitude and azimuthal field
            components.

        """

        defaults = dict(cmap='nio',
                        deriv=0,
                        vmax=200,
                        vmin=-200)

        kwargs = defaultkeys(defaults, kwargs)

        time = self.model_static.breaks[0]

        self.model_static.plot_maps(time, radius, nmax=nmax, **kwargs)

    def synth_coeffs_gsm(self, time, *, nmax=None, source=None):
        """
        Compute the external GSM field coefficients from the CHAOS model.

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
            Coefficients of the external GSM field in term of geographic
            coordinates (GEO).

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
            warnings.warn("Requested coefficients are "
                          "outside of the modelled period from "
                          f"{start} to {end}. Doing linear extrapolation in "
                          "GSM reference frame.")

        # build rotation matrix from file
        frequency_spectrum = np.load(basicConfig['file.GSM_spectrum'])
        assert np.all(
            frequency_spectrum['dipole'] == basicConfig['params.dipole']), \
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
        Compute GSM magnetic field components.

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
        >>> time = np.linspace(0., 10., num=2)
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

    def synth_coeffs_sm(self, time, *, nmax=None, source=None):
        """
        Compute the external SM field from the CHAOS model.

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
            Coefficients of the external SM field coefficients in terms of
            geographic coordinates (GEO).

        Examples
        --------
        >>> import chaosmagpy as cp
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> model.synth_coeffs_sm(0.0)
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
        time = np.array(time, dtype=float)

        if np.amin(time) < start or np.amax(time) > end:
            warnings.warn(
                'Requested coefficients are outside of the '
                f'modelled period from {start} to {end}. Doing linear '
                'extrapolation in SM reference frame.')

        # load rotation matrix spectrum from file
        frequency_spectrum = np.load(basicConfig['file.SM_spectrum'])
        assert np.all(
            frequency_spectrum['dipole'] == basicConfig['params.dipole']), \
            "SM rotation coefficients not compatible with the chosen dipole."

        # load RC-index file: first hdf5 then dat-file format
        try:
            with h5py.File(basicConfig['file.RC_index'], 'r') as f_RC:
                # check RC index time and input times
                start = f_RC['time'][0]
                end = f_RC['time'][-1]
                # interpolate RC (linear) at input times: RC is callable
                RC = sip.interp1d(f_RC['time'], f_RC['RC_' + source[0]],
                                  kind='linear')
        except OSError:  # dat file second
            f_RC = du.load_RC_datfile(basicConfig['file.RC_index'])
            start = f_RC['time'].iloc[0]
            end = f_RC['time'].iloc[-1]
            # interpolate RC (linear) at input times: RC is callable
            RC = sip.interp1d(f_RC['time'], f_RC['RC_' + source[0]],
                              kind='linear')

        if np.amin(time) < start:
            raise ValueError(
                'Insufficient RC time series. Input times must be between '
                '{:.2f} and {:.2f}, but found {:.2f}.'.format(
                    start, end, np.amin(time)))

        # check RC index time and inputs time
        if np.amax(time) > end:
            raise ValueError(
                'Insufficient RC time series. Input times must be between '
                '{:.2f} and {:.2f}, but found {:.2f}.'.format(
                    start, end, np.amax(time)))

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
            coeffs_sm = np.empty(time.shape + (self.n_sm*(self.n_sm+2),))

            coeffs_sm[..., 0] = (RC(time)*self.coeffs_sm[0]
                                 + delta_q10(time))
            coeffs_sm[..., 1] = (RC(time)*self.coeffs_sm[1]
                                 + delta_q11(time))
            coeffs_sm[..., 2] = (RC(time)*self.coeffs_sm[2]
                                 + delta_s11(time))
            coeffs_sm[..., 3:] = self.coeffs_sm[3:]

            # rotate external SM coefficients to GEO reference
            coeffs = np.einsum('...ij,...j', rotate_gauss, coeffs_sm)

        elif source == 'internal':
            # unpack file: oscillations per day, complex spectrum
            frequency = frequency_spectrum['frequency_ind']
            spectrum = frequency_spectrum['spectrum_ind']
            scaled = frequency_spectrum['scaled']

            # build rotation matrix for induced coefficients SM -> GEO
            rotate_gauss_ind = cu.synth_rotate_gauss(
                time, frequency, spectrum, scaled=scaled)

            # take degree 1 matrix elements of unmodified rotation matrix
            # since induction effect will be accounted for by RC_i
            rotate_gauss = rotate_gauss[..., :3, :3]

            coeffs_sm = np.empty(time.shape + (3,))

            coeffs_sm[..., 0] = RC(time)*self.coeffs_sm[0]
            coeffs_sm[..., 1] = RC(time)*self.coeffs_sm[1]
            coeffs_sm[..., 2] = RC(time)*self.coeffs_sm[2]

            coeffs_sm_ind = np.empty(
                time.shape + (self.n_sm*(self.n_sm+2),))

            coeffs_sm_ind[..., 0] = delta_q10(time)
            coeffs_sm_ind[..., 1] = delta_q11(time)
            coeffs_sm_ind[..., 2] = delta_s11(time)
            coeffs_sm_ind[..., 3:] = self.coeffs_sm[3:]

            # rotate internal SM coefficients to GEO reference
            coeffs = np.einsum('...ij,...j', rotate_gauss_ind, coeffs_sm_ind)
            coeffs[..., :3] += np.einsum('...ij,...j', rotate_gauss, coeffs_sm)

        else:
            raise ValueError(f'Unknown source "{source}". '
                             'Use {external'', ''internal''}.')

        return coeffs[..., :nmax*(nmax+2)]

    def synth_values_sm(self, time, radius, theta, phi, *, nmax=None,
                        source=None, grid=None):
        """
        Compute SM magnetic field components.

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
        >>> time = np.linspace(0., 10., num=2)
        >>> Br, Bt, Bp = model.synth_values_sm(time, 6371.2, 45., 0.)
        >>> Br
        array([-18.85890361, -13.99893523])

        """

        source = 'all' if source is None else source
        grid = False if grid is None else grid

        if source == 'all':
            coeffs_ext = self.synth_coeffs_sm(time, nmax=nmax,
                                              source='external')
            coeffs_int = self.synth_coeffs_sm(time, nmax=nmax,
                                              source='internal')

            B_radius_ext, B_theta_ext, B_phi_ext = mu.synth_values(
                coeffs_ext, radius, theta, phi, source='external', grid=grid)
            B_radius_int, B_theta_int, B_phi_int = mu.synth_values(
                coeffs_int, radius, theta, phi, source='internal', grid=grid)

            B_radius = B_radius_ext + B_radius_int
            B_theta = B_theta_ext + B_theta_int
            B_phi = B_phi_ext + B_phi_int

        elif source in ['external', 'internal']:
            coeffs = self.synth_coeffs_sm(time, nmax=nmax, source=source)
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
            Choose contribution either from GSM, SM or both added
            (default is 'all').
        source : {'all', 'external', 'internal'}, optional
            Choose source to be external (inducing), internal (induced) or
            both added (default is 'all').

        Returns
        -------
        B_radius, B_theta, B_phi
            Global map of the radial, colatitude and azimuthal field
            components.

        """

        reference = 'all' if reference is None else reference.lower()
        source = 'all' if source is None else source

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

        plot_maps(theta, phi, B_radius, B_theta, B_phi,
                  titles=titles, label=units)
        plt.show()

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
            Derivative in time (default is 0). For secular variation, choose
            ``deriv=1``.
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
        >>> model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
        >>> time = np.linspace(500., 600., num=2)
        >>> model.meta['satellites']  # check satellite names
        ('oersted', 'champ', 'sac_c', 'swarm_a', 'swarm_b', 'swarm_c')

        >>> model.synth_euler_angles(time, 'swarm_a')
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
            coeffs += pre

        return coeffs

    def save_shcfile(self, filepath, *, model=None, deriv=None,
                     leap_year=None):
        """
        Save spherical harmonic coefficients to a file in `shc`-format.

        Parameters
        ----------
        filepath : str
            Path and name of output file `*.shc`.
        model : {'tdep', 'static'}, optional
            Choose part of the model to save (default is 'tdep').
        deriv : int, optional
            Derivative of the time-dependent field (default is 0, ignored for
            static source).
        leap_year : {True, False}, optional
            Take leap years for decimal year conversion into account
            (defaults to ``True``).

        """

        model = 'tdep' if model is None else model

        deriv = 0 if deriv is None else deriv

        leap_year = True if leap_year is None else leap_year

        if model == 'tdep':
            if self.model_tdep.coeffs is None:
                raise ValueError("Time-dependent internal field coefficients "
                                 "are missing.")

            nmin = 1
            nmax = self.model_tdep.nmax
            breaks = self.model_tdep.breaks
            order = self.model_tdep.order

            # compute times in mjd2000
            times = np.array([], dtype=float)
            for start, end in zip(breaks[:-1], breaks[1:]):
                step = (end - start)/(order-1)
                times = np.append(times, np.arange(start, end, step))
            times = np.append(times, breaks[-1])

            # create header lines
            header = (
                f"# {self}\n"
                f"# Spherical harmonic coefficients of the time-dependent"
                f" internal field model (derivative = {deriv})"
                f" from degree {nmin} to {nmax}.\n"
                f"# Coefficients ({du.gauss_units(deriv)}) are given at"
                f" {(breaks.size-1) * (order-1) + 1} points in time\n"
                f"# and were extracted from order-{order}"
                f" piecewise polynomial (i.e. break points are every"
                f" {order} steps of the time sequence).\n"
                )

            gauss_coeffs = self.synth_coeffs_tdep(
                times, nmax=nmax, deriv=deriv)

        # output static field model coefficients
        if model == 'static':
            if self.model_static.coeffs is None:
                raise ValueError("Static internal field coefficients "
                                 "are missing.")

            nmin = self.model_tdep.nmax + 1
            nmax = self.model_static.nmax
            order = 1

            # compute times in mjd2000
            times = np.array([self.model_static.breaks[0]])

            # create additonal header lines
            header = (
                f"# {self}\n"
                f"# Spherical harmonic coefficients of the static internal"
                f" field model from degree {nmin} to {nmax}.\n"
                f"# Given at the first break point.\n"
                )

            gauss_coeffs = self.synth_coeffs_static(nmax=nmax)

        du.save_shcfile(times, gauss_coeffs, order=order, filepath=filepath,
                        nmin=nmin, nmax=nmax, leap_year=leap_year,
                        header=header)

    def save_matfile(self, filepath):
        """
        Save CHAOS model to a `mat`-file.

        The model must be fully specified as is the case if originally loaded
        from a `mat`-file.

        Parameters
        ----------
        filepath : str
            Path and name of `mat`-file that is to be saved.

        """

        # write time-dependent internal field model to matfile
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
                m_Dst=m_Dst)

            hdf.write(model_ext, path='/model_ext', filename=filepath,
                      matlab_compatible=True)

        if self.model_euler:
            # write Euler angles to matfile for each satellite
            satellites = self.meta['satellites']

            t_break_Euler = []  # list of Euler angle breaks for satellites
            alpha = []  # list of alpha for each satellite
            beta = []  # list of beta for each satellite
            gamma = []  # list of gamma for each satellite
            for num, satellite in enumerate(satellites):

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

        if self.model_static.coeffs is not None:
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
            Path to mat-file containing the CHAOS model.
        name : str, optional
            User defined name of the model. Defaults to the filename without
            the file extension.
        satellites : list of strings, optional
            List of satellite names whose Euler angles are stored in the
            mat-file. This is needed for correct referencing as this
            information is not given in the standard CHAOS mat-file format
            (defaults to ``['oersted', 'champ', 'sac_c', 'swarm_a', 'swarm_b',
            'swarm_c', 'cryosat-2_1', 'cryosat-2_2', 'cryosat-2_3']``.)

        Returns
        -------
        model : :class:`CHAOS` instance
            Fully initialized model.

        Examples
        --------
        Load for example the mat-file ``CHAOS-6-x7.mat`` in the current working
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
            Path to shc-file.
        name : str, optional
            User defined name of the model. Defaults to ``'CHAOS-<version>'``,
            where <version> is the default in
            ``basicConfig['params.version']``.
        leap_year : {True, False}, optional
            Take leap year in time conversion into account (default).
            Otherwise, use conversion factor of 365.25 days per year.

        Returns
        -------
        model : :class:`CHAOS` instance
            Class instance with either the time-dependent or static internal
            part initialized.

        Examples
        --------
        Load for example the shc-file ``CHAOS-6-x7_tdep.shc`` in the current
        working directory, containing the coefficients of time-dependent
        internal part of the CHAOS-6-x7 model.

        >>> import chaosmagpy as cp
        >>> model = cp.CHAOS.from_shc('CHAOS-6-x7_tdep.shc')
        >>> print(model)

        See Also
        --------
        load_CHAOS_shcfile

        """

        leap_year = True if leap_year is None else leap_year

        return load_CHAOS_shcfile(filepath, name=name, leap_year=leap_year)


def load_CHAOS_matfile(filepath, name=None, satellites=None):
    """
    Load CHAOS model from mat-file.

    Parameters
    ----------
    filepath : str
        Path to mat-file containing the CHAOS model.
    name : str, optional
        User defined name of the model. Defaults to the filename without the
        file extension.
    satellites : list of strings, optional
        List of satellite names whose Euler angles are stored in the mat-file.
        This is needed for correct referencing as this information is not
        given in the standard CHAOS mat-file format (defaults to
        ``['oersted', 'champ', 'sac_c', 'swarm_a', 'swarm_b', 'swarm_c',
        'cryosat-2_1', 'cryosat-2_2', 'cryosat-2_3']``.)

    Returns
    -------
    model : :class:`CHAOS` instance
        Fully initialized model.

    Examples
    --------
    Load for example the mat-file ``CHAOS-6-x7.mat`` in the current working
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

    mat_contents = du.load_matfile(filepath)

    pp = mat_contents['pp']

    order = int(pp['order'])
    pieces = int(pp['pieces'])
    dim = int(pp['dim'])
    breaks = pp['breaks']  # flatten 2-D array
    coefs = pp['coefs']

    # reshaping coeffs_tdep from 2-D to 3-D: (order, pieces, coefficients)
    coeffs_tdep = coefs.transpose().reshape((order, pieces, dim))

    try:
        coeffs_static = mat_contents['g'].reshape((1, 1, -1))

    except KeyError as err:
        warnings.warn(f'Missing static internal field coefficients: {err}')
        coeffs_static = None

    try:
        model_ext = mat_contents['model_ext']

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

    except KeyError as err:
        warnings.warn(f'Missing external field coefficients: {err}')
        coeffs_delta = None
        breaks_delta = None
        coeffs_gsm = None
        coeffs_sm = None

    # define satellite names
    if satellites is None:
        satellites = ['oersted', 'champ', 'sac_c', 'swarm_a', 'swarm_b',
                      'swarm_c', 'cryosat-2_1', 'cryosat-2_2', 'cryosat-2_3']

    try:
        model_euler = mat_contents['model_Euler']

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

    except KeyError as err:
        warnings.warn(f'Missing Euler angles: {err}')
        coeffs_euler = None
        breaks_euler = None

    try:
        params = mat_contents['params']
        dict_params = {'Euler_prerotation': params['Euler_prerotation']}

    except KeyError as err:
        warnings.warn(f'Missing params dictionary: {err}')
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
                  name=name,
                  meta=meta)

    return model


def load_CHAOS_shcfile(filepath, name=None, leap_year=None):
    """
    Load CHAOS model from shc-file.

    The file should contain the coefficients of the time-dependent or static
    internal part of the CHAOS model. In case of the time-dependent part, a
    reconstruction of the piecewise polynomial is performed.

    Parameters
    ----------
    filepath : str
        Path to shc-file.
    name : str, optional
        User defined name of the model. Defaults to the filename without the
        file extension.
    leap_year : {True, False}, optional
        Take leap year in time conversion into account (default). Otherwise,
        use conversion factor of 365.25 days per year.

    Returns
    -------
    model : :class:`CHAOS` instance
        Class instance with either the time-dependent or static internal part
        initialized.

    Examples
    --------
    Load for example the shc-file ``CHAOS-6-x7_tdep.shc`` in the current
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

    leap_year = True if leap_year is None else leap_year

    time, coeffs, params = du.load_shcfile(filepath, leap_year=leap_year)

    if time.size == 1:  # static field

        nmin = params['nmin']
        nmax = params['nmax']
        coeffs_static = np.zeros((nmax*(nmax+2),))
        coeffs_static[int(nmin**2-1):] = coeffs  # pad zeros to coefficients
        coeffs_static = coeffs_static.reshape((1, 1, -1))
        model = CHAOS(breaks=np.array(time),
                      coeffs_static=coeffs_static,
                      name=name)

    else:  # time-dependent field

        coeffs = np.transpose(coeffs)
        order = params['order']
        step = params['step']
        breaks = time[::step]

        # interpolate with piecewise polynomial of given order
        coeffs_pp = np.empty((order, time.size // step, coeffs.shape[-1]))
        for m, left_break in enumerate(breaks[:-1]):
            left = m * step  # time index of left_break
            x = time[left:left+order] - left_break
            c = np.linalg.solve(np.vander(x, order), coeffs[left:left+order])

            for k in range(order):
                coeffs_pp[k, m] = c[k]

        model = CHAOS(breaks=breaks,
                      order=order,
                      coeffs_tdep=coeffs_pp,
                      name=name)

    return model


def load_CovObs_txtfile(filepath, name=None):
    """
    Load the ensemble mean of the internal model from the COV-OBS
    txt-file in spline format.

    Parameters
    ----------
    filepath : str
        Path to spline-formatted txt-file (not part of ChaosMagPy).
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

        model = cp.load_CovObs_txtfile('COV-OBS.x2-int')  # load model txt-file

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # sample model timespan a bit away from endpoints
        time = np.linspace(model.breaks[6], model.breaks[-6], 1000)
        coeffs = model.synth_coeffs(time, nmax=1, deriv=1)

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

    # add endpoint multiplicity to "trick" scipy's BSpline routine
    knots = mu.augment_breaks(breaks, order)

    data = np.fromstring(' '.join(lines[2:]), sep=' ')

    # add zeros at endpoints to match manually extended knots
    coeffs = np.zeros((knots.size - order, nmax*(nmax+2)))

    # insert actual coefficients, endpoint coefficients are now zero
    coeffs[degree:-degree, :] = data.reshape(
        (breaks.size - order, nmax*(nmax+2)))

    return BaseModel.from_bspline(name, knots, coeffs, order,
                                  source='internal', meta=None)


def load_gufm1_txtfile(filepath, name=None):
    """
    Load model parameter file of the gufm1 model.

    Parameters
    ----------
    filepath : str
        Path to txt-file (not part of ChaosMagPy).
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

        model = cp.load_gufm1_txtfile('gufm1')  # load model txt-file

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # sample model timespan a bit away from endpoints
        time = np.linspace(model.breaks[6], model.breaks[-6], 1000)
        coeffs = model.synth_coeffs(time, nmax=1, deriv=1)

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
    coeffs = np.zeros((knots.size - order, nmax*(nmax + 2)))

    # insert actual coefficients, endpoint coefficients are now zero
    coeffs[degree:-degree, :] = data[(nbreaks + 2):].reshape(
        (breaks.size - order, nmax*(nmax + 2)))

    return BaseModel.from_bspline(name, knots, coeffs, order,
                                  source='internal', meta=None)
