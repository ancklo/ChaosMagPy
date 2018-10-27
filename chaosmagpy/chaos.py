import numpy as np
import os
import warnings
import scipy.io as sio
import scipy.interpolate as sip
import chaosmagpy.coordinate_utils as cu
import chaosmagpy.model_utils as mu
import chaosmagpy.data_utils as du
from chaosmagpy.plot_utils import plot_timeseries, plot_maps
from datetime import datetime
from timeit import default_timer as timer

ROOT = os.path.abspath(os.path.dirname(__file__))


class CHAOS(object):
    """
    Class for the time-dependent geomagnetic field model CHAOS. Currently only
    CHAOS-6 is supported.

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

    Attributes
    ----------
    timestamp : str
        UTC timestamp at initialization.
    breaks : ndarray, shape (m+1,)
        Break points for piecewise-polynomial representation of the
        time-dependent internal field in modified Julian date format.
    pieces : int, positive
        Number `m` of intervals given by break points in ``breaks``.
    order : int, positive
        Order `k` of polynomial pieces (e.g. 4 = cubic) of the time-dependent
        internal field.
    n_tdep : int, positive
        Maximum spherical harmonic degree of the time-dependent internal field.
    coeffs_tdep : ndarray, shape (`k`, `m`, ``n_tdep`` * (``n_tdep`` + 2))
        Coefficients of the time-dependent internal field.
    n_static : int, positive
        Maximum spherical harmonic degree of the static internal (i.e.
        small-scale crustal) field.
    coeffs_static : ndarray,  shape (``n_static`` * (``n_static`` + 2),)
        Coefficients of the static internal field.
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
    breaks_euler : dict with ndarrays, shape (:math:`m_e` +1,)
        Dictionary containing satellite name as key and corresponding break
        vectors of Euler angles (keys are ``'oersted'``, ``'champ'``,
        ``'sac_c'``, ``'swarm_a'``, ``'swarm_b'``, ``'swarm_c'``).
    coeffs_euler : dict with ndarrays, shape (1, :math:`m_e`, 3)
        Dictionary containing satellite name as key and arrays of the Euler
        angles alpha, beta and gamma as trailing dimension (keys are
        ``'oersted'``, ``'champ'``, ``'sac_c'``, ``'swarm_a'``, ``'swarm_b'``,
        ``'swarm_c'``).
    version : str
        Version specifier (``None`` evaluates to '6.x7' by default).

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

      model.n_tdep  # check degree of time-dependent model, should be 1

    Now, plot for example the field map on January 2, 2000 0:00 UTC

    .. code-block:: python

      model.plot_tdep_map(time=1, radius=6371.2)

    Save the Gauss coefficients of the time-dependent field in shc-format to
    the current working directory.

    .. code-block:: python

      model.save_shcfile('CHAOS-6-x7_tdep.shc', source='tdep')


    """

    def __init__(self, breaks=None, order=None, *,
                 coeffs_tdep=None, coeffs_static=None,
                 coeffs_sm=None, coeffs_gsm=None,
                 breaks_delta=None, coeffs_delta=None,
                 breaks_euler=None, coeffs_euler=None,
                 version=None):
        """
        Initialize the CHAOS model.

        """

        self.timestamp = str(datetime.utcnow())

        # time-dependent internal field
        self.breaks = breaks
        self.pieces = None if breaks is None else int(breaks.size - 1)
        self.order = None if order is None else int(order)

        # helper for returning the degree of the provided coefficients
        def dimension(coeffs):
            if coeffs is None:
                return coeffs
            else:
                return int(np.sqrt(coeffs.shape[-1] + 1) - 1)

        self.coeffs_tdep = coeffs_tdep
        self.n_tdep = dimension(coeffs_tdep)

        # static field
        if coeffs_static is not None:
            assert coeffs_static.ndim == 1

        self.coeffs_static = coeffs_static
        self.n_static = dimension(coeffs_static)

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
        self.breaks_euler = breaks_euler
        self.coeffs_euler = coeffs_euler

        # set version of CHAOS model
        if version is None:
            print('Setting default CHAOS version to "6.x7".')
            self.version = '6.x7'
        else:
            self.version = str(version)

    def __call__(self, time, radius, theta, phi, source_list=None):
        """
        Calculate the magnetic field from the CHAOS model using all sources
        (time-dependent and static internal field, and external SM/GSM
        fields including induced parts).

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

        grid_shape = max(radius.shape, theta.shape, phi.shape)

        B_radius = np.zeros(grid_shape)
        B_theta = np.zeros(grid_shape)
        B_phi = np.zeros(grid_shape)

        if 'tdep' in source_list:

            print(f'Computing time-dependent internal field'
                  f' up to degree {self.n_tdep}.')

            s = timer()
            coeffs = self.synth_tdep_field(time)
            B_radius_new, B_theta_new, B_phi_new = mu.synth_values(
                coeffs, radius, theta, phi)

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
            coeffs = self.synth_static_field(nmax=nmax_static)
            B_radius_new, B_theta_new, B_phi_new = mu.synth_values(
                coeffs, radius, theta, phi)

            B_radius += B_radius_new
            B_theta += B_theta_new
            B_phi += B_phi_new
            e = timer()

            print('Finished in {:.6} seconds.'.format(e-s))

        if 'gsm' in source_list:

            print(f'Computing GSM field up to degree {self.n_gsm}.')

            s = timer()
            coeffs_ext = self.synth_gsm_field(time, source='external')
            coeffs_int = self.synth_gsm_field(time, source='internal')

            B_radius_ext, B_theta_ext, B_phi_ext = mu.synth_values(
                coeffs_ext, radius, theta, phi, source='external')
            B_radius_int, B_theta_int, B_phi_int = mu.synth_values(
                coeffs_int, radius, theta, phi, source='internal')

            B_radius += B_radius_ext + B_radius_int
            B_theta += B_theta_ext + B_theta_int
            B_phi += B_phi_ext + B_phi_int
            e = timer()

            print('Finished in {:.6} seconds.'.format(e-s))

        if 'sm' in source_list:

            print(f'Computing SM field up to degree {self.n_sm}.')

            s = timer()
            coeffs_ext = self.synth_sm_field(time, source='external')
            coeffs_int = self.synth_sm_field(time, source='internal')

            B_radius_ext, B_theta_ext, B_phi_ext = mu.synth_values(
                coeffs_ext, radius, theta, phi, source='external')
            B_radius_int, B_theta_int, B_phi_int = mu.synth_values(
                coeffs_int, radius, theta, phi, source='internal')

            B_radius += B_radius_ext + B_radius_int
            B_theta += B_theta_ext + B_theta_int
            B_phi += B_phi_ext + B_phi_int
            e = timer()

            print('Finished in {:.6} seconds.'.format(e-s))

        return B_radius, B_theta, B_phi

    def __str__(self):
        """
        Print model version and initialization timestamp.
        """

        string = ("This is CHAOS-{:} initialized on ".format(self.version)
                  + self.timestamp + ' UTC.')

        return string

    def synth_tdep_field(self, time, nmax=None, deriv=None):
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
        deriv : int, positive, optional
            Derivative in time (None defaults to 0). For secular variation,
            choose ``deriv=1``.

        Returns
        -------
        coeffs : ndarray, shape (..., ``nmax`` * (``nmax`` + 2))
            Coefficients of the time-dependent internal field.

        """

        if self.coeffs_tdep is None:
            raise ValueError("Time-dependent internal field coefficients "
                             "are missing.")

        # handle optional argument: nmax
        if nmax is None:
            nmax = self.n_tdep
        elif nmax > self.n_tdep:
            warnings.warn(
                'Supplied nmax = {0} is incompatible with number of model '
                'coefficients. Using nmax = {1} instead.'.format(
                    nmax, self.n_tdep))
            nmax = self.n_tdep

        if deriv is None:
            deriv = 0

        if np.amin(time) < self.breaks[0] or np.amax(time) > self.breaks[-1]:
            warnings.warn("Requested time-dependent internal coefficients are "
                          "outside of the modelled period from "
                          f"{self.breaks[0]} to {self.breaks[-1]}. "
                          "Returning nan's.")

        PP = sip.PPoly.construct_fast(
            self.coeffs_tdep[..., :nmax*(nmax+2)].astype(float),
            self.breaks.astype(float), extrapolate=False)

        PP = PP.derivative(nu=deriv)
        coeffs = PP(time) * 365.25**deriv

        return coeffs

    def plot_tdep_timeseries(self, radius, theta, phi,
                             nmax=None, deriv=None):
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

        Returns
        -------
        B_radius, B_theta, B_phi
            Time series plot of the radial, colatitude and azimuthal field
            components.

        """

        if deriv is None:
            deriv = 0

        time = np.linspace(self.breaks[0], self.breaks[-1], num=500)

        coeffs = self.synth_tdep_field(time, nmax=nmax, deriv=deriv)

        B_radius, B_theta, B_phi = mu.synth_values(
            coeffs, radius, theta, phi, nmax=nmax, source='internal')

        units = du.gauss_units(deriv)
        titles = ['$B_r$', '$B_\\theta$', '$B_\\phi$']

        plot_timeseries(time, B_radius, B_theta, B_phi,
                        figsize=None, titles=titles, label=units)

    def plot_tdep_map(self, time, radius, nmax=None, deriv=None):
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

        Returns
        -------
        B_radius, B_theta, B_phi
            Global map of the radial, colatitude and azimuthal field
            components.

        """

        deriv = 0 if deriv is None else deriv

        # handle optional argument: nmax
        if nmax is None:
            nmax = self.n_tdep
        elif nmax > self.n_tdep:
            warnings.warn(
                'Supplied nmax = {0} is incompatible with number of model '
                'coefficients. Using nmax = {1} instead.'.format(
                    nmax, self.n_tdep))
            nmax = self.n_tdep

        time = np.array(time, dtype=np.float)
        theta = np.linspace(1, 179, num=320)
        phi = np.linspace(-180, 180, num=721)

        coeffs = self.synth_tdep_field(time, nmax=nmax, deriv=deriv)

        B_radius, B_theta, B_phi = mu.synth_values(
            coeffs, radius, theta, phi,
            nmax=nmax, source='internal', grid=True)

        units = du.gauss_units(deriv)
        titles = [f'$B_r$ ($n\\leq{nmax}$, deriv={deriv})',
                  f'$B_\\theta$ ($n\\leq{nmax}$, deriv={deriv})',
                  f'$B_\\phi$ ($n\\leq{nmax}$, deriv={deriv})']

        plot_maps(theta, phi, B_radius, B_theta, B_phi,
                  titles=titles, label=units)

    def synth_static_field(self, nmax=None):
        """
        Compute the static internal field coefficients from the CHAOS model.

        Parameters
        ----------
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).

        Returns
        -------
        coeffs : ndarray, shape (``nmax`` * (``nmax`` + 2),)
            Coefficients of the static internal field.

        """

        if self.coeffs_static is None:
            raise ValueError("Static internal field coefficients are missing.")

        # handle optional argument: nmax
        if nmax is None:
            nmax = self.n_static
        elif nmax > self.n_static:
            warnings.warn(
                'Supplied nmax = {0} is incompatible with number of model '
                'coefficients. Using nmax = {1} instead.'.format(
                    nmax, self.n_static))
            nmax = self.n_static
        elif self.n_tdep and nmax <= self.n_tdep:
            raise ValueError("SH degree of static internal field is smaller "
                             "than the time-dependent internal field degree.")

        coeffs = self.coeffs_static[:nmax*(nmax+2)]

        return coeffs

    def plot_static_map(self, radius, nmax=None):
        """
        Plot global map of the static internal field from the CHAOS model.

        Parameters
        ----------
        radius : ndarray, shape (), (1,) or float
            Array containing the radius in kilometers.
        nmax : int, positive, optional
            Maximum degree harmonic expansion (default is given by the model
            coefficients, but can also be smaller, if specified).

        Returns
        -------
        B_radius, B_theta, B_phi
            Global map of the radial, colatitude and azimuthal field
            components.

        """

        # handle optional argument: nmax
        if nmax is None:
            nmax = self.n_static
        elif nmax > self.n_static:
            warnings.warn(
                'Supplied nmax = {0} is incompatible with number of model '
                'coefficients. Using nmax = {1} instead.'.format(
                    nmax, self.n_static))
            nmax = self.n_static

        theta = np.linspace(1, 179, num=360)
        phi = np.linspace(-180, 180, num=721)

        coeffs = self.synth_static_field(nmax=nmax)

        B_radius, B_theta, B_phi = mu.synth_values(
            coeffs, radius, theta, phi,
            nmax=nmax, source='internal', grid=True)

        units = du.gauss_units(0)
        titles = [f'$B_r$ ($n\\leq{nmax}$)',
                  f'$B_\\theta$ ($n\\leq{nmax}$)',
                  f'$B_\\phi$ ($n\\leq{nmax}$)']

        plot_maps(theta, phi, B_radius, B_theta, B_phi,
                  label=units, titles=titles, cmap='nio', vmax=200, vmin=-200)

    def synth_gsm_field(self, time, nmax=None, source=None):
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

        # build rotation matrix from file
        filepath = os.path.join(ROOT, 'lib', 'frequency_spectrum_gsm.npz')
        frequency_spectrum = np.load(filepath)

        if source == 'external':
            # unpack file: oscillations per day, complex spectrum
            frequency = frequency_spectrum['frequency']
            spectrum = frequency_spectrum['spectrum']

            # build rotation matrix for external field coefficients GSM -> GEO
            rotate_gauss = cu.synth_rotate_gauss(time, frequency, spectrum)

            # rotate external GSM coefficients to GEO reference
            coeffs = np.matmul(rotate_gauss, self.coeffs_gsm)

        elif source == 'internal':
            # unpack file: oscillations per day, complex spectrum
            frequency_ind = frequency_spectrum['frequency_ind']
            spectrum_ind = frequency_spectrum['spectrum_ind']

            # build rotation matrix for external field coefficients GSM -> GEO
            rotate_gauss_ind = cu.synth_rotate_gauss(time, frequency_ind,
                                                     spectrum_ind)

            # rotate internal GSM coefficients to GEO reference
            coeffs = np.matmul(rotate_gauss_ind, self.coeffs_gsm)

        else:
            raise ValueError("Wrong source parameter, use "
                             "'external' or 'internal'")

        return coeffs[..., :nmax*(nmax+2)]

    def synth_sm_field(self, time, nmax=None, source=None):
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
        if np.amin(time) < start or np.amax(time) > end:
            warnings.warn("Requested external coefficients in SM are "
                          "outside of the modelled period from "
                          f"{start} to {end}. Returning nan's.")

        # ensure ndarray input
        time = np.array(time, dtype=np.float)

        # build rotation matrix from file
        filepath = os.path.join(ROOT, 'lib', 'frequency_spectrum_sm.npz')
        frequency_spectrum = np.load(filepath)

        # load RC-index into date frame
        filepath = os.path.join(ROOT, 'lib', 'RC_1997-2018.dat')
        df_RC = du.load_RC_datfile(filepath)

        # check RC index time and input times
        start = df_RC['time'].iloc[0]
        end = df_RC['time'].iloc[-1]
        if np.amin(time) < start:
            raise ValueError(
                'Insufficient RC time series. Input times must be between '
                f'{start} and {end}, but found {np.amin(time)}')

        # check RC index time and inputs time
        if np.amax(time) > end:
            raise ValueError(
                'Insufficient RC time series. Input times must be between '
                f'{start} and {end}, but found {np.amax(time)}.')

        # use piecewise polynomials to evaluate baseline correction in bins
        delta_q10 = sip.PPoly.construct_fast(
            self.coeffs_delta['q10'].astype(float),
            self.breaks_delta['q10'].astype(float), extrapolate=False)

        delta_q11 = sip.PPoly.construct_fast(
            self.coeffs_delta['q11'].astype(float),
            self.breaks_delta['q11'].astype(float), extrapolate=False)

        delta_s11 = sip.PPoly.construct_fast(
            self.coeffs_delta['s11'].astype(float),
            self.breaks_delta['s11'].astype(float), extrapolate=False)

        # unpack file: oscillations per day, complex spectrum
        frequency = frequency_spectrum['frequency']
        spectrum = frequency_spectrum['spectrum']

        # build rotation matrix for external field coefficients SM -> GEO
        rotate_gauss = cu.synth_rotate_gauss(time, frequency, spectrum)

        if source == 'external':
            # interpolate RC (linear) at input times: RC_ext is callable
            RC_ext = sip.interp1d(
                df_RC['time'].values, df_RC['RC_e'].values, kind='linear')

            coeffs_sm = np.empty(time.shape + (self.n_sm*(self.n_sm+2),))

            coeffs_sm[..., 0] = (RC_ext(time)*self.coeffs_sm[0]
                                 + delta_q10(time))
            coeffs_sm[..., 1] = (RC_ext(time)*self.coeffs_sm[1]
                                 + delta_q11(time))
            coeffs_sm[..., 2] = (RC_ext(time)*self.coeffs_sm[2]
                                 + delta_s11(time))
            coeffs_sm[..., 3:] = self.coeffs_sm[3:]

            # insert singleton dimension before last dimension of
            # coeffs_sm_time since this is needed for correct broadcasting
            # and summing
            coeffs_sm = np.expand_dims(coeffs_sm, axis=-2)

            # rotate external SM coefficients to GEO reference
            coeffs = np.sum(rotate_gauss*coeffs_sm, axis=-1)

        elif source == 'internal':
            # interpolate RC (linear) at input times: RC_int is callable
            RC_int = sip.interp1d(
                df_RC['time'].values, df_RC['RC_i'].values, kind='linear')

            # unpack file: oscillations per day, complex spectrum
            frequency = frequency_spectrum['frequency_ind']
            spectrum = frequency_spectrum['spectrum_ind']

            # build rotation matrix for induced coefficients SM -> GEO
            rotate_gauss_ind = cu.synth_rotate_gauss(time, frequency, spectrum)

            # take degree 1 matrix elements of unmodified rotation matrix
            # since induction effect will be accounted for by RC_i
            rotate_gauss = rotate_gauss[..., :3, :3]

            coeffs_sm = np.empty(time.shape + (3,))

            coeffs_sm[..., 0] = RC_int(time)*self.coeffs_sm[0]
            coeffs_sm[..., 1] = RC_int(time)*self.coeffs_sm[1]
            coeffs_sm[..., 2] = RC_int(time)*self.coeffs_sm[2]

            coeffs_sm_ind = np.empty(
                time.shape + (self.n_sm*(self.n_sm+2),))

            coeffs_sm_ind[..., 0] = delta_q10(time)
            coeffs_sm_ind[..., 1] = delta_q11(time)
            coeffs_sm_ind[..., 2] = delta_s11(time)
            coeffs_sm_ind[..., 3:] = self.coeffs_sm[3:]

            # insert singleton dimension before last dimension of
            # coeffs_sm since this is needed for correct broadcasting
            # and summing
            coeffs_sm = np.expand_dims(coeffs_sm, axis=-2)
            coeffs_sm_ind = np.expand_dims(coeffs_sm_ind, axis=-2)

            # rotate internal SM coefficients to GEO reference
            coeffs = np.sum(rotate_gauss_ind*coeffs_sm_ind, axis=-1)
            coeffs[..., :3] += np.sum(rotate_gauss*coeffs_sm, axis=-1)

        else:
            raise ValueError("Wrong source parameter, use "
                             "'external' or 'internal'")

        return coeffs[..., :nmax*(nmax+2)]

    def plot_external_map(self, time, radius, nmax=None, reference=None,
                          source=None):
        """
        Plot global map of the external field from the CHAOS model.

        Parameters
        ----------
        time : ndarray, shape (), (1,) or float
            Time given as MJD2000 (modified Julian date).
        radius : ndarray, shape (), (1,) or float
            Array containing the radius in kilometers.
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

        if reference is None:
            reference = 'all'
        reference = reference.lower()

        if source is None:
            source = 'all'

        time = np.array(time, dtype=float)
        radius = np.array(radius, dtype=float)
        theta = np.linspace(1, 179, num=320)
        phi = np.linspace(-180, 180, num=721)

        # compute GSM contribution: external, internal
        if reference == 'all' or reference == 'gsm':

            if source == 'all' or source == 'external':
                coeffs_ext = self.synth_gsm_field(
                    time, nmax=nmax, source='external')

                # compute magnetic field given external GSM field coefficients
                B_radius_ext, B_theta_ext, B_phi_ext = mu.synth_values(
                    coeffs_ext, radius, theta, phi,
                    nmax=nmax, source='external', grid=True)

            if source == 'all' or source == 'internal':
                coeffs_int = self.synth_gsm_field(
                    time, nmax=nmax, source='internal')

                # compute magnetic field given external GSM field coefficients
                B_radius_int, B_theta_int, B_phi_int = mu.synth_values(
                    coeffs_int, radius, theta, phi,
                    nmax=nmax, source='internal', grid=True)

            if source == 'external':
                B_radius_gsm = B_radius_ext
                B_theta_gsm = B_theta_ext
                B_phi_gsm = B_phi_ext
            elif source == 'internal':
                B_radius_gsm = B_radius_int
                B_theta_gsm = B_theta_int
                B_phi_gsm = B_phi_int
            elif source == 'all':
                B_radius_gsm = B_radius_ext + B_radius_int
                B_theta_gsm = B_theta_ext + B_theta_int
                B_phi_gsm = B_phi_ext + B_phi_int
            else:
                raise ValueError('Use source "internal", "external" or "all" '
                                 '(for both added together)')

        # compute SM contribution: external, internal
        if reference == 'all' or reference == 'sm':

            if source == 'all' or source == 'external':
                coeffs_ext = self.synth_sm_field(
                    time, nmax=nmax, source='external')

                # compute magnetic field given external SM field coefficients
                B_radius_ext, B_theta_ext, B_phi_ext = mu.synth_values(
                    coeffs_ext, radius, theta, phi,
                    nmax=nmax, source='external', grid=True)

            if source == 'all' or source == 'internal':
                coeffs_int = self.synth_sm_field(
                    time, nmax=nmax, source='internal')

                # compute magnetic field given external SM field coefficients
                B_radius_int, B_theta_int, B_phi_int = mu.synth_values(
                    coeffs_int, radius, theta, phi,
                    nmax=nmax, source='internal', grid=True)

            if source == 'external':
                B_radius_sm = B_radius_ext
                B_theta_sm = B_theta_ext
                B_phi_sm = B_phi_ext
            elif source == 'internal':
                B_radius_sm = B_radius_int
                B_theta_sm = B_theta_int
                B_phi_sm = B_phi_int
            elif source == 'all':
                B_radius_sm = B_radius_ext + B_radius_int
                B_theta_sm = B_theta_ext + B_theta_int
                B_phi_sm = B_phi_ext + B_phi_int
            else:
                raise ValueError('Use source "internal", "external" or "all" '
                                 '(for both added together)')

        if reference == 'all':
            B_radius = B_radius_gsm + B_radius_sm
            B_theta = B_theta_gsm + B_theta_sm
            B_phi = B_phi_gsm + B_phi_sm

        if reference == 'gsm':
            B_radius = B_radius_gsm
            B_theta = B_theta_gsm
            B_phi = B_phi_gsm

        if reference == 'sm':
            B_radius = B_radius_sm
            B_theta = B_theta_sm
            B_phi = B_phi_sm

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

    def save_shcfile(self, filepath, source=None, deriv=None):
        """
        Save spherical harmonic coefficients to a file in `shc`-format.

        Parameters
        ----------
        filepath : str
            Path and name of output file `*.shc`.
        source : {'tdep', 'static'}, optional
            Choose part of the model to save (default is 'tdep').
        deriv : int, optional
            Derivative of the time-dependent field (default is 0, ignored for
            static source).

        """

        source = 'tdep' if source is None else source

        deriv = 0 if deriv is None else deriv

        if source == 'tdep':
            if self.coeffs_tdep is None:
                raise ValueError("Time-dependent internal field coefficients "
                                 "are missing.")

            nmin = 1
            nmax = self.n_tdep

            # compute times in mjd2000
            times = np.array([], dtype=np.float)
            for start, end in zip(self.breaks[:-1], self.breaks[1:]):
                step = (end - start)/(self.order-1)
                times = np.append(times, np.arange(start, end, step))
            times = np.append(times, self.breaks[-1])

            # write comment lines
            comment = (
                f"# {self}\n"
                f"# Spherical harmonic coefficients of the time-dependent"
                f" internal field model (derivative = {deriv})"
                f" from degree {nmin} to {nmax}.\n"
                f"# Coefficients (nT/yr^{deriv}) are given at"
                f" {(self.breaks.size-1) * (self.order-1) + 1} points in"
                f" time and were extracted from order-{self.order}"
                f" piecewise polynomial (i.e. break points are every"
                f" {self.order-1} steps).\n"
                f"# Created on {datetime.utcnow()} UTC.\n"
                f"{nmin} {nmax} {times.size} {self.order} {self.order-1}\n"
                )

            gauss_coeffs = self.synth_tdep_field(times, nmax=nmax, deriv=deriv)

        # output static field model coefficients
        if source == 'static':
            if self.coeffs_static is None:
                raise ValueError("Static internal field coefficients "
                                 "are missing.")

            nmin = self.n_tdep + 1
            nmax = self.n_static

            # compute times in mjd2000
            times = np.array([self.breaks[0]])

            # write comment lines
            comment = (
                f"# {self}\n"
                f"# Spherical harmonic coefficients of the static internal"
                f" field model from degree {nmin} to {nmax}.\n"
                f"# Created on {datetime.utcnow()} UTC.\n"
                f"{nmin} {nmax} {times.size} 1 0\n"
                )

            gauss_coeffs = self.synth_static_field(nmax=nmax)
            gauss_coeffs = gauss_coeffs[int(nmin**2-1):].reshape((1, -1))

        # compute all possible degree and orders
        degree = np.array([], dtype=np.int)
        order = np.array([], dtype=np.int)
        for n in range(nmin, nmax+1):
            degree = np.append(degree, np.repeat(n, 2*n+1))
            order = np.append(order, [0])
            for m in range(1, n+1):
                order = np.append(order, [m, -m])

        with open(filepath, 'w') as f:
            # write comment line
            f.write(comment)

            # write header lines to 8 significants
            f.write('  ')  # to represent two missing values
            for time in times:
                f.write(' {:9.4f}'.format(time / 365.25 + 2000.))
            f.write('\n')

            # write coefficient table to 8 significants
            for row, (n, m) in enumerate(zip(degree, order)):

                f.write('{:} {:}'.format(n, m))

                for col in range(times.size):
                    f.write(' {:.8e}'.format(gauss_coeffs[col, row]))

                f.write('\n')

        print('Coefficients saved to {}.'.format(
            os.path.join(os.getcwd(), filepath)))

    @classmethod
    def from_mat(self, filepath):
        """
        Alternative constructor for creating a :class:`CHAOS` class instance.

        Parameters
        ----------
        filepath : str
            Path to mat-file containing the CHAOS model.

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

           model = cp.CHAOS.from_mat('CHAOS-6-x7.mat')
           print(model)

        See Also
        --------
        load_CHAOS_matfile

        """

        return load_CHAOS_matfile(filepath)

    def to_mat(self):
        raise NotImplementedError

    @classmethod
    def from_shc(self, filepath):
        """
        Alternative constructor for creating a :class:`CHAOS` class instance.

        Parameters
        ----------
        filepath : str
            Path to shc-file.

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
        return load_CHAOS_shcfile(filepath)

    def to_shc(self):
        raise NotImplementedError


def load_CHAOS_matfile(filepath):
    """
    Load CHAOS model from mat-file, e.g. ``CHAOS-6-x7.mat``.

    Parameters
    ----------
    filepath : str
        Path to mat-file containing the CHAOS model.

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

    version = _guess_version(filepath)

    mat_contents = sio.loadmat(filepath)
    pp = mat_contents['pp']
    pp = pp[0, 0]
    model_ext = mat_contents['model_ext']
    model_ext = model_ext[0, 0]
    model_euler = mat_contents['model_Euler']
    model_euler = model_euler[0, 0]

    order = int(pp['order'])
    pieces = int(pp['pieces'])
    dim = int(pp['dim'])
    breaks = np.ravel(pp['breaks'])  # flatten 2-D array
    coefs = pp['coefs']
    coeffs_static = np.ravel(mat_contents['g'])

    # reshaping coeffs_tdep from 2-D to 3-D: (order, pieces, coefficients)
    n_tdep = int(np.sqrt(dim+1)-1)
    coeffs_tdep = np.empty([order, pieces, n_tdep * (n_tdep + 2)])
    for k in range(order):
        for l in range(pieces):
            for m in range(n_tdep * (n_tdep + 2)):
                coeffs_tdep[k, l, m] = coefs[l * n_tdep * (n_tdep + 2) + m, k]

    # external field (SM): n=1, 2
    coeffs_sm = np.ravel(model_ext['m_sm'])  # degree 1 are time averages!
    coeffs_sm[:3] = np.ravel(model_ext['m_Dst'])  # replace with m_Dst

    # external field (GSM): n=1, 2
    n_gsm = int(2)
    coeffs_gsm = np.zeros((n_gsm*(n_gsm+2),))  # appropriate number of coeffs
    coeffs_gsm[[0, 3]] = np.ravel(model_ext['m_gsm'])  # only m=0 are non-zero

    # coefficients and breaks of external SM field offsets for q10, q11, s11
    breaks_delta = {}
    breaks_delta['q10'] = np.ravel(model_ext['t_break_q10'])
    breaks_delta['q11'] = np.ravel(model_ext['t_break_qs11'])
    breaks_delta['s11'] = np.ravel(model_ext['t_break_qs11'])

    # reshape to comply with scipy PPoly coefficients
    coeffs_delta = {}
    coeffs_delta['q10'] = np.ravel(model_ext['q10']).reshape((1, -1))
    coeffs_delta['q11'] = np.ravel(model_ext['qs11'][:, 0]).reshape((1, -1))
    coeffs_delta['s11'] = np.ravel(model_ext['qs11'][:, 1]).reshape((1, -1))

    # define satellite names
    satellites = ['oersted', 'champ', 'sac_c', 'swarm_a', 'swarm_b', 'swarm_c']

    # coefficients and breaks of euler angles
    breaks_euler = {}
    for num, satellite in enumerate(satellites):
        breaks_euler[satellite] = np.ravel(
            model_euler['t_break_Euler'][0, num])

    # reshape angles to be (1, N) then stack last axis to get (1, N, 3)
    # so first dimension: order, second: number of intervals, third: 3 angles
    def compose_array(num):
        return np.stack([model_euler[angle][0, num].reshape(
                (1, -1)) for angle in ['alpha', 'beta', 'gamma']], axis=-1)

    coeffs_euler = {}
    for num, satellite in enumerate(satellites):
        coeffs_euler[satellite] = compose_array(num)

    return CHAOS(
        breaks=breaks,
        order=order,
        coeffs_tdep=coeffs_tdep,
        coeffs_static=coeffs_static,
        coeffs_sm=coeffs_sm,
        coeffs_gsm=coeffs_gsm,
        breaks_delta=breaks_delta,
        coeffs_delta=coeffs_delta,
        breaks_euler=breaks_euler,
        coeffs_euler=coeffs_euler,
        version=version)


def load_CHAOS_shcfile(filepath):
    """
    Load CHAOS model from shc-file, e.g. ``CHAOS-6-x7_tdep.shc``. The file
    should contain the coefficients or the time-dependent or static internal
    part of the CHAOS model. In case of the time-dependent part, a
    reconstruction of the piecewise polynomial is performed (only accurate
    to 0.01 nT).

    Parameters
    ----------
    filepath : str
        Path to shc-file.

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

    Notes
    -----
    The piecewise polynomial of the time-dependent internal part of the CHAOS
    model is contructed from the snapshots of the model, accurately handling
    the break-points of the model. Note however that the original coefficients
    can only be retrieved to an absolute error of around 0.01 nT, i.e.
    small-scale field and higher derivates (>1) are not recommended if
    accuracy is important. Use the ``load_CHAOS_matfile`` function instead.

    See Also
    --------
    CHAOS, load_CHAOS_matfile

    """

    time, coeffs, params = du.load_shcfile(str(filepath))

    if time.size == 1:  # static field

        nmin = params['nmin']
        nmax = params['nmax']
        coeffs_static = np.zeros((nmax*(nmax+2),))
        coeffs_static[int(nmin**2-1):] = coeffs  # pad zeros to coefficients
        model = CHAOS(coeffs_static=coeffs_static,
                      version=_guess_version(filepath))

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
                      version=_guess_version(filepath))

    return model


def _guess_version(filepath):
    """
    Extract version from filename. For example from the file 'CHAOS-6-x7.mat',
    it returns '6.x7'. If not successful, user input is required, otherwise
    default '6.x7' is returned.
    """

    # quick fix, not very stable: consider ntpath, imho
    head, tail = os.path.split(filepath)
    tail = os.path.splitext(tail or os.path.basename(head))[0]
    version = '.'.join(tail.split('_')[0].split('-')[-2:])
    while len(version) != 4 or version[1] != '.':
        version = input('Type in version [6.x7]: ')
        if not version:  # default behaviour
            version = '6.x7'

    return version
