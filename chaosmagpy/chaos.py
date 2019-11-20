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

"""

import numpy as np
import os
import re
import warnings
import scipy.interpolate as sip
import hdf5storage as hdf
import h5py
import chaosmagpy.coordinate_utils as cu
import chaosmagpy.model_utils as mu
import chaosmagpy.data_utils as du
import chaosmagpy.plot_utils as pu
import matplotlib.pyplot as plt
from chaosmagpy.config_utils import basicConfig
from chaosmagpy.plot_utils import defaultkeys
from datetime import datetime
from timeit import default_timer as timer

ROOT = os.path.abspath(os.path.dirname(__file__))


class Base(object):
    """
    Piecewise polynomial base class.

    """

    def __init__(self, name, breaks=None, order=None, coeffs=None, meta=None):

        self.name = str(name)

        # ensure breaks is None or has 2 elements
        if breaks is not None:
            breaks = np.array(breaks, dtype=np.float)
            if breaks.size == 1:
                breaks = np.append(breaks, breaks)

        self.breaks = breaks

        self.pieces = None if breaks is None else int(breaks.size - 1)

        if coeffs is None:
            self.coeffs = coeffs
            self.order = None if order is None else int(order)
            self.dim = None

        else:
            coeffs = np.array(coeffs, dtype=np.float)

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
            Array containing the time in modified Julian dates.
        dim : int, positive, optional
            Truncation value of the number of coefficients (no truncation by
            default).
        deriv : int, positive, optional
            Derivative in time (None defaults to 0). For secular variation,
            choose ``deriv=1``.
        extrapolate : {'linear', 'quadratic', 'cubic', 'spline', 'constant', 'off'} or int, optional
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
        coeffs : ndarray, shape (..., ``nmax`` * (``nmax`` + 2))
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
                    string = '", "'.join([*dkey.keys()])
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
    Class for time-dependent (piecewise polynomial) model.

    Parameters
    ----------
    name : str
        User specified name of the model.
    breaks : ndarray, shape (m+1,)
        Break points for piecewise-polynomial representation of the
        time-dependent internal field in modified Julian date format.
    order : int, positive
        Order `k` of polynomial pieces (e.g. 4 = cubic) of the time-dependent
        field.
    coeffs : ndarray, shape (`k`, `m`, ``nmax`` * (``nmax`` + 2))
        Coefficients of the time-dependent field.
    source : {'internal', 'external'}
        Internal or external source (defaults to ``'internal'``)
    meta : dict, optional
        Dictionary containing additional information about the model.

    Attributes
    ----------
    breaks : ndarray, shape (m+1,)
        Break points for piecewise-polynomial representation of the
        magnetic field in modified Julian date format.
    pieces : int, positive
        Number `m` of intervals given by break points in ``breaks``.
    order : int, positive
        Order `k` of polynomial pieces (e.g. 4 = cubic) of the time-dependent
        field.
    nmax : int, positive
        Maximum spherical harmonic degree of the time-dependent field.
    dim : int, ``nmax`` * (``nmax`` + 2)
        Dimension of the model.
    coeffs : ndarray, shape (`k`, `m`, ``nmax`` * (``nmax`` + 2))
        Coefficients of the time-dependent field.
    source : {'internal', 'external'}
        Internal or external source (defaults to ``'internal'``)

    """

    def __init__(self, name, breaks=None, order=None, coeffs=None,
                 source=None, meta=None):
        """
        Initialize time-dependent spherical harmonic model as a piecewise
        polynomial.

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
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        deriv : int, positive, optional
            Derivative in time (None defaults to 0). For secular variation,
            choose ``deriv=1``.
        extrapolate : {'linear', 'quadratic', 'cubic', 'spline', 'constant', 'off'} or int, optional
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

        dim = None if nmax is None else nmax*(nmax+2)
        coeffs = super().synth_coeffs(time, dim=dim, deriv=deriv,
                                      extrapolate=extrapolate)

        return coeffs

    def synth_values(self, time, radius, theta, phi, *, nmax=None,
                     deriv=None, grid=None, extrapolate=None):
        """
        Compute magnetic field components.

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
        extrapolate : {'linear', 'quadratic', 'cubic', 'spline', 'constant', 'off'} or int, optional
            Extrapolate to times outside of the model bounds. Specify
            polynomial degree as string or any order as integer. Defaults to
            ``'linear'`` (equiv. to order 2 polynomials).

        Returns
        -------
        B_radius, B_theta, B_phi : ndarray, shape (...)
            Radial, colatitude and azimuthal field components.

        """

        coeffs = self.synth_coeffs(time, nmax=nmax, deriv=deriv,
                                   extrapolate=extrapolate)

        return mu.synth_values(coeffs, radius, theta, phi, nmax=nmax,
                               source=self.source, grid=grid)

    def power_spectrum(self, time, radius=None, **kwargs):
        """
        Compute the powerspectrum.

        Parameters
        ----------
        time : ndarray, shape (...)
            Time in modified Julian date.
        radius : float
            Radius in kilometers (defaults to mean Earth's surface defined in
            ``basicConfig['r_surf']``).

        Returns
        -------
        R_n : ndarray, shape (..., ``nmax``)
            Power spectrum of spherical harmonics up to degree ``nmax``.

        Other Parameters
        ----------------
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        deriv : int, positive, optional
            Derivative in time (default is 0). For secular variation, choose
            ``deriv=1``.
        **kwargs : keywords
            Other options to pass to :meth:`BaseModel.synth_coeffs` method.

        See Also
        --------
        model_utils.power_spectrum

        """

        radius = basicConfig['params.r_surf'] if radius is None else radius

        coeffs = self.synth_coeffs(time, **kwargs)

        return mu.power_spectrum(coeffs, radius)

    def plot_power_spectrum(self, time, **kwargs):
        """
        Plot the power spectrum.

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
                        titles='power spectrum')

        kwargs = defaultkeys(defaults, kwargs)

        radius = kwargs.pop('radius')
        nmax = kwargs.pop('nmax')
        deriv = kwargs.pop('deriv')

        units = f'({du.gauss_units(deriv)})$^2$'
        kwargs.setdefault('ylabel', units)

        R_n = self.power_spectrum(time, radius, nmax=nmax, deriv=deriv)

        pu.plot_power_spectrum(R_n, **kwargs)
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
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
        deriv : int, positive, optional
            Derivative in time (default is 0). For secular variation, choose
            ``deriv=1``.
        **kwargs : keywords
            Other options are passed to :func:`plot_utils.plot_maps`
            function.

        See Also
        --------
        plot_utils.plot_maps

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

        time = np.array(time, dtype=np.float)
        theta = np.linspace(1, 179, num=320)
        phi = np.linspace(-180, 180, num=721)

        B_radius, B_theta, B_phi = self.synth_values(
            time, radius, theta, phi, nmax=nmax, deriv=deriv,
            grid=True, extrapolate=None)

        pu.plot_maps(theta, phi, B_radius, B_theta, B_phi, **kwargs)
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
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).
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
        plot_utils.plot_timeseries

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

        pu.plot_timeseries(time, B_radius, B_theta, B_phi, **kwargs)
        plt.show()


class CHAOS(object):
    """
    Class for the time-dependent geomagnetic field model CHAOS.

    Parameters
    ----------
    breaks : ndarray, shape (m+1,)
        Break points for piecewise-polynomial representation of the
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
    coeffs_delta : dict with ndarrays, shape (:math:`m_q`,)
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
    version : str, optional
        Version specifier (e.g. ``'6.x9'``).
    name : str, optional
        User defined name of the model. Defaults to ``'CHAOS-<version>'``,
        where <version> is the default in ``basicConfig['params.version']``.
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
        as :class:`Base` class instance. (CHAOS-6: keys are ``'oersted'``,
        ``'champ'``, ``'sac_c'``, ``'swarm_a'``, ``'swarm_b'``, ``'swarm_c'``).
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
    coeffs_delta : dict with ndarrays, shape (:math:`m_q`,)
        Coefficients of baseline corrections of static external field in SM
        coordinates. The dictionary keys are ``'q10'``, ``'q11'``, ``'s11'``.
    version : str
        Version specifier (``None`` evaluates to
        ``basicConfig['params.version']`` by default).

    Examples
    --------
    Create a time-dependent internal field model as a piecewise polynomial of
    order 4 (i.e. cubic) having 10 pieces, spanning the first 50 days in the
    year 2000 (breaks in modified Julian date 2000). As example, choose random
    coefficients for the time-dependent field of spherical harmonic degree 1
    (= 3 coefficients).

    .. code-block:: python

      import chaosmagpy as cp
      import numpy as np

      # define model
      m = 10  # number of pieces
      breaks = np.linspace(0, 50, m+1)
      k = 4  # polynomial order
      coeffs = np.random.random(size=(k, m, 3))

      # create CHAOS class instance
      model = cp.CHAOS(breaks, k, coeffs_tdep=coeffs)

    Now, plot for example the field map on January 2, 2000 0:00 UTC

    .. code-block:: python

      model.plot_maps_tdep(time=1, radius=6371.2)

    Save the Gauss coefficients of the time-dependent field in shc-format to
    the current working directory.

    .. code-block:: python

      model.save_shcfile('CHAOS-6-x7_tdep.shc', model='tdep')


    """

    def __init__(self, breaks, order=None, *,
                 coeffs_tdep=None, coeffs_static=None,
                 coeffs_sm=None, coeffs_gsm=None,
                 breaks_delta=None, coeffs_delta=None,
                 breaks_euler=None, coeffs_euler=None,
                 version=None, name=None, meta=None):
        """
        Initialize the CHAOS model.

        """

        self.timestamp = str(datetime.utcnow())

        # time-dependent internal field
        self.model_tdep = BaseModel('model_tdep', breaks, order, coeffs_tdep)
        self.model_static = BaseModel('model_tdep', breaks[[0, -1]], 1,
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

                model = Base(satellite, order=1,
                             breaks=breaks_euler[satellite],
                             coeffs=coeffs_euler[satellite],
                             meta={'Euler_prerotation': Euler_prerotation})
                self.model_euler[satellite] = model

        # set version of CHAOS model
        if version is None:
            version = basicConfig['params.version']
            print(f'Setting default CHAOS version to {version}.')
            self.version = version
        else:
            self.version = str(version)

        # give the model a name: CHAOS-x.x or user input
        if name is None:
            self.name = f'CHAOS-{self.version}'
        else:
            self.name = name

        self.meta = meta

    def __call__(self, time, radius, theta, phi, source_list=None):
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
        source_list : list, ['tdep', 'static', 'gsm', 'sm']
            Specify sources in any order. Default is all sources.

        Returns
        -------
        B_radius, B_theta, B_phi : ndarray, shape (grid_shape)
            Radial, colatitude and azimuthal field components.

        """

        time = np.array(time, dtype=float)
        radius = np.array(radius, dtype=float)
        theta = np.array(theta, dtype=float)
        phi = np.array(phi, dtype=float)

        if source_list is None:
            source_list = ['tdep', 'static', 'gsm', 'sm']

        source_list = np.ravel(np.array(source_list))

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

            print(f'Computing time-dependent internal field'
                  f' up to degree {self.model_tdep.nmax}.')

            s = timer()
            B_radius_new, B_theta_new, B_phi_new = self.synth_values_tdep(
                time, radius, theta, phi)

            B_radius += B_radius_new
            B_theta += B_theta_new
            B_phi += B_phi_new
            e = timer()

            print('Finished in {:.6} seconds.'.format(e-s))

        if 'static' in source_list:

            nmax_static = 85

            print(f'Computing static internal (i.e. small-scale crustal) field'
                  f' up to degree {nmax_static}.')

            s = timer()
            B_radius_new, B_theta_new, B_phi_new = self.synth_values_static(
                radius, theta, phi, nmax=nmax_static)

            B_radius += B_radius_new
            B_theta += B_theta_new
            B_phi += B_phi_new
            e = timer()

            print('Finished in {:.6} seconds.'.format(e-s))

        if 'gsm' in source_list:

            print(f'Computing GSM field up to degree {self.n_gsm}.')

            s = timer()
            B_radius_new, B_theta_new, B_phi_new = self.synth_values_gsm(
                time, radius, theta, phi, source='all')

            B_radius += B_radius_new
            B_theta += B_theta_new
            B_phi += B_phi_new
            e = timer()

            print('Finished in {:.6} seconds.'.format(e-s))

        if 'sm' in source_list:

            print(f'Computing SM field up to degree {self.n_sm}.')

            s = timer()
            B_radius_new, B_theta_new, B_phi_new = self.synth_values_sm(
                time, radius, theta, phi, source='all')

            B_radius += B_radius_new
            B_theta += B_theta_new
            B_phi += B_phi_new
            e = timer()

            print('Finished in {:.6} seconds.'.format(e-s))

        return B_radius, B_theta, B_phi

    def __str__(self):
        """
        Print model version and initialization timestamp.
        """

        string = (f"This is {self.name} (v{self.version})"
                  f" initialized on {self.timestamp} UTC.")

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

        """

        return self.model_tdep.synth_coeffs(time, nmax=nmax, **kwargs)

    def synth_values_tdep(self, *args, **kwargs):
        """
        Compute magnetic components of the internal
        time-dependent field.

        Convenience method. See :meth:`BaseModel.synth_values`.

        """

        return self.model_tdep.synth_values(*args, **kwargs)

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
        time = np.array(time, dtype=np.float)

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
        time = np.array(time, dtype=np.float)

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

        pu.plot_maps(theta, phi, B_radius, B_theta, B_phi,
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
            times = np.array([], dtype=np.float)
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

        # write Euler angles to matfile for each satellite
        satellites = self.meta['satellites']

        t_break_Euler = []  # list of Euler angle breaks for satellites
        alpha = []  # list of alpha for each satellite
        beta = []  # list of beta for each satellite
        gamma = []  # list of gamma for each satellite
        for num, satellite in enumerate(satellites):

            # reduce breaks if start end end are equal
            breaks = self.model_euler[satellite].breaks
            if breaks[0] == breaks[-1]:
                breaks = breaks[0]

            t_break_Euler.append(breaks.reshape((-1, 1)).astype(float))
            alpha.append(self.model_euler[satellite].coeffs[0, :, 0].reshape(
                (-1, 1)).astype(float))
            beta.append(self.model_euler[satellite].coeffs[0, :, 1].reshape(
                (-1, 1)).astype(float))
            gamma.append(self.model_euler[satellite].coeffs[0, :, 2].reshape(
                (-1, 1)).astype(float))

        hdf.write(np.array(t_break_Euler), path='/model_Euler/t_break_Euler/',
                  filename=filepath, matlab_compatible=True)
        hdf.write(np.array(alpha), path='/model_Euler/alpha/',
                  filename=filepath, matlab_compatible=True)
        hdf.write(np.array(beta), path='/model_Euler/beta/',
                  filename=filepath, matlab_compatible=True)
        hdf.write(np.array(gamma), path='/model_Euler/gamma/',
                  filename=filepath, matlab_compatible=True)

        # write static internal field model to matfile
        g = np.ravel(self.model_static.coeffs).reshape((-1, 1))

        hdf.write(g, path='/g', filename=filepath, matlab_compatible=True)

        hdf.write(self.meta['params'], path='/params', filename=filepath,
                  matlab_compatible=True)

        print('CHAOS saved to {}.'.format(
            os.path.join(os.getcwd(), filepath)))

    @classmethod
    def from_mat(self, filepath, name=None, version=None):
        """
        Alternative constructor for creating a :class:`CHAOS` class instance.

        Parameters
        ----------
        filepath : str
            Path to mat-file containing the CHAOS model.
        name : str, optional
            User defined name of the model. Defaults to ``'CHAOS-<version>'``,
            where <version> is the default in
            ``basicConfig['params.version']``.
        version : str, optional
            Version specifier (e.g. ``'6.x9'``).

        Returns
        -------
        model : :class:`CHAOS` instance
            Fully initialized model.

        Examples
        --------
        Load for example the mat-file ``CHAOS-6-x7.mat`` in the current working
        directory like this:

        .. code-block:: python

           from chaosmagpy import CHAOS

           model = CHAOS.from_mat('CHAOS-6-x7.mat')
           print(model)

        See Also
        --------
        load_CHAOS_matfile

        """

        return load_CHAOS_matfile(filepath, name=name, version=version)

    @classmethod
    def from_shc(self, filepath, *, name=None, leap_year=None, version=None):
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
        version : str, optional
            Version specifier (e.g. ``'6.x9'``).

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

        .. code-block:: python

           import chaosmagpy as cp

           model = cp.CHAOS.from_shc('CHAOS-6-x7_tdep.shc')
           print(model)

        See Also
        --------
        load_CHAOS_shcfile

        """

        leap_year = True if leap_year is None else leap_year

        return load_CHAOS_shcfile(filepath, name=name, leap_year=leap_year,
                                  version=version)


def load_CHAOS_matfile(filepath, name=None, version=None):
    """
    Load CHAOS model from mat-file.

    Parameters
    ----------
    filepath : str
        Path to mat-file containing the CHAOS model.
    name : str, optional
        User defined name of the model. Defaults to ``'CHAOS-<version>'``,
        where <version> is the default in ``basicConfig['params.version']``.
    version : str, optional
        Version specifier (e.g. ``'6.x9'``).

    Returns
    -------
    model : :class:`CHAOS` instance
        Fully initialized model.

    Examples
    --------
    Load for example the mat-file ``CHAOS-6-x7.mat`` in the current working
    directory like this:

    .. code-block:: python

       import chaosmagpy as cp

       model = cp.load_CHAOS_matfile('CHAOS-6-x7.mat')
       print(model)

    See Also
    --------
    CHAOS, load_CHAOS_shcfile

    """

    filepath = str(filepath)

    if version is None:
        version = _guess_version(filepath)

    mat_contents = du.load_matfile(filepath, variable_names=[
        'g', 'pp', 'model_ext', 'model_Euler', 'params'])

    pp = mat_contents['pp']
    model_ext = mat_contents['model_ext']
    model_euler = mat_contents['model_Euler']
    params = mat_contents['params']

    order = int(pp['order'])
    pieces = int(pp['pieces'])
    dim = int(pp['dim'])
    breaks = pp['breaks']  # flatten 2-D array
    coefs = pp['coefs']

    coeffs_static = mat_contents['g'].reshape((1, 1, -1))

    # reshaping coeffs_tdep from 2-D to 3-D: (order, pieces, coefficients)
    coeffs_tdep = coefs.transpose().reshape((order, pieces, dim))

    # external field (SM): n=1, 2 (n=1 are sm offset time averages!)
    coeffs_sm = np.copy(model_ext['m_sm'])
    coeffs_sm[:3] = model_ext['m_Dst']  # replace with m_Dst

    # external field (GSM): n=1, 2
    n_gsm = int(2)
    coeffs_gsm = np.zeros((n_gsm*(n_gsm+2),))  # appropriate number of coeffs
    # only m=0 are non-zero
    coeffs_gsm[[0, 3]] = model_ext['m_gsm']

    # coefficients and breaks of external SM field offsets for q10, q11, s11
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

    # define satellite names
    satellites = ['oersted', 'champ', 'sac_c', 'swarm_a', 'swarm_b', 'swarm_c',
                  'cryosat-2_1', 'cryosat-2_2', 'cryosat-2_3']

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

    # reshape angles to be (1, N) then stack last axis to get (1, N, 3)
    # so first dimension: order, second: number of intervals, third: 3 angles
    coeffs_euler = dict()
    for num, satellite in enumerate(satellites):
        euler = [model_euler[angle][num].reshape((1, -1)) for
                 angle in ['alpha', 'beta', 'gamma']]
        euler = np.stack(euler, axis=-1).astype(np.float)  # because no sac-c

        coeffs_euler[satellite] = euler

    dict_params = dict(Euler_prerotation=params['Euler_prerotation'])
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
                  version=version,
                  name=name,
                  meta=meta)

    return model


def load_CHAOS_shcfile(filepath, name=None, leap_year=None, version=None):
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
        User defined name of the model. Defaults to ``'CHAOS-<version>'``,
        where <version> is the default in ``basicConfig['params.version']``.
    leap_year : {True, False}, optional
        Take leap year in time conversion into account (default). Otherwise,
        use conversion factor of 365.25 days per year.
    version : str, optional
        Version specifier (e.g. ``'6.x9'``).

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

    .. code-block:: python

       import chaosmagpy as cp

       model = cp.load_CHAOS_shcfile('CHAOS-6-x7_tdep.shc')
       print(model)

    See Also
    --------
    CHAOS, load_CHAOS_matfile

    """

    leap_year = True if leap_year is None else leap_year

    time, coeffs, params = du.load_shcfile(str(filepath), leap_year=leap_year)

    if version is None:
        version = _guess_version(filepath)

    if time.size == 1:  # static field

        nmin = params['nmin']
        nmax = params['nmax']
        coeffs_static = np.zeros((nmax*(nmax+2),))
        coeffs_static[int(nmin**2-1):] = coeffs  # pad zeros to coefficients
        coeffs_static = coeffs_static.reshape((1, 1, -1))
        model = CHAOS(breaks=np.array(time),
                      coeffs_static=coeffs_static,
                      version=version)

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
                      version=version,
                      name=name)

    return model


def _guess_version(filepath):
    """
    Extract version from filename.

    For example from the file 'CHAOS-6-x7.mat', it returns '6.x7'. If not
    successful, version is set to default in ``basicConfig['params.version']``.

    """

    # search for CHAOS-{numeral}-x{numeral}, not followed by path separator
    head, tail = os.path.split(filepath)
    match = re.search(r'CHAOS-(\d+)[-.]?(x\d+)?(\.?\w*)', tail)

    if match:
        first, second = match.group(1, 2)
        if second is None:
            version = first
        else:
            version = '.'.join([first, second])
    else:
        warnings.warn(
            'Unknown CHAOS model version. Setting version to {:}.'.format(
                basicConfig['params.version']))
        version = basicConfig['params.version']

    return version
