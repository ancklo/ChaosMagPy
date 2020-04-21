"""
This module provides functions related to coordinate transformations. Typical
coordinate reference frames and corresponding abbbreviations are listed in the
following.

**Abbreviations**

GEO : geocentric, orthogonal coordinate system
    With z-axis along Earth's rotation axis, x-axis pointing to Greenwich and
    y-axis completing right-handed coordinate system.
GG : geographic or geodetic, orthogonal coordinate system.
    Earth is approximated by a spheroid (ellipsoid of revolution) with
    equatorial radius `a` and polar radius `b`, `b < a`.
USE : local orthogonal coordinate system on spherical surface
    Axes directions are defined as Up-South-East (e.g. B_radius, B_theta,
    B_phi in GEO).
GSM : geocentric solar magnetic, orthogonal coordinate system
    With x-axis pointing towards the sun, y-axis perpendicular to plane spanned
    by Eart-Sun line and dipole axis and z-axis completing right-handed system.
SM : solar magnetic, orthogonal coordinate system
    With z-axis along dipole axis pointing to the geomagnetic North pole,
    y-axis perpendicular to plane containing the dipole axis and the Earth-Sun
    line, and x-axis completing the right-handed system.
MAG : magnetic orthogonal coordinate system (centered dipole)
    With z-axis pointing to the geomagnetic North pole, x-axis in the plane
    spanned by the dipole axis and Earth's rotation axis, and y-axis completing
    the right-handed system.

.. autosummary::
    :toctree: functions

    igrf_dipole
    synth_rotate_gauss
    rotate_gauss_fft
    rotate_gauss
    sun_position
    zenith_angle
    spherical_to_cartesian
    cartesian_to_spherical
    basevectors_gsm
    basevectors_sm
    basevectors_use
    basevectors_mag
    geo_to_gg
    gg_to_geo
    geo_to_base
    matrix_geo_to_base
    transform_points
    transform_vectors
    center_azimuth
    local_time
    q_response
    q_response_1D

"""

import numpy as np
import os
from numpy import degrees, radians
from math import pi, ceil, factorial
from chaosmagpy.model_utils import legendre_poly
from chaosmagpy.config_utils import basicConfig

ROOT = os.path.abspath(os.path.dirname(__file__))


def igrf_dipole(epoch=None):
    """
    Compute unit vector that is anti-parallel to the IGRF dipole.

    The vector points towards the geomagnetic North pole (located in the
    Northern Hemisphere). Epoch 2015 of IGRF-12 is used by default.

    Parameters
    ----------
    epoch : {'2015', '2010'}, optional
        Epoch of IGRF-12 (2015) and IGRF-11 (2010, default).

    Returns
    -------
    dipole : ndarray, shape (3,)
        Unit vector pointing to geomagnetic North pole (located in Northern
        Hemisphere).

    """

    # default IGRF dipole
    epoch = '2015' if epoch is None else str(epoch)

    if epoch == '2015':
        # IGRF-12 dipole coefficients, epoch 2015: theta = 9.69, phi = 287.37
        dipole = _dipole_to_unit(np.array([-29442.0, -1501.0, 4797.1]))

    elif epoch == '2010':
        # dipole as used in original chaos software (IGRF-11), epoch 2010
        dipole = _dipole_to_unit(11.32, 289.59)

    else:
        raise ValueError('Only epoch "2010" (IGRF-11) and'
                         '"2015" (IGRF-12) supported.')

    return dipole


def _dipole_to_unit(*args):
    """
    Convert degree-1 SH coefficients or geomagnetic North pole position to
    unit vector.

    Parameters
    ----------
    *args
        Takes ``theta``, ``phi`` in degrees, ``[g10, g11, h11]`` or
        ``g10``, ``g11``, ``h11`` as input.

    Returns
    -------
    vector : ndarray, shape (3,)
        Unit vector pointing to geomagnetic North pole.
    """

    if len(args) == 1:
        vector = np.roll(args[0], -1)  # g11, h11, g10: dipole
        # unit vector, opposite to dipole
        vector = -vector/np.linalg.norm(vector)
    elif len(args) == 2:
        theta = radians(args[0])
        phi = radians(args[1])
        vector = np.array([np.sin(theta)*np.cos(phi),
                           np.sin(theta)*np.sin(phi),
                           np.cos(theta)])
    elif len(args) == 3:
        vector = np.array([args[1], args[2], args[0]])  # g11, h11, g10: dipole
        # unit vector, opposite to dipole
        vector = -vector/np.linalg.norm(vector)
    else:
        raise ValueError('Only 1, 2 or 3 inputs accepted '
                         f'but {len(args)} given.')

    return vector


def synth_rotate_gauss(time, frequency, spectrum, scaled=None):
    """
    Compute matrices to transform sphercial harmonic expansion from Fourier
    coefficients.

    The function computes matrices that couple the spherical harmonic expansion
    of a time-dependent reference system (e.g. GSM, SM) and GEO using Fourier
    coefficients.

    Parameters
    ----------
    time : ndarray, shape (...)
        Time given as modified Julian date, i.e. with respect to the date 0h00
        January 1, 2000 (mjd2000).
    frequency : ndarray, shape (k,) or (k, m, n)
        Vector of positive frequencies given in oscillations per day.
    spectrum : ndarray, shape (k, m, n)
        Fourier components of the matrices (reside in the last two dimensions).
    scaled : bool, optional (defaults to ``False``)
        If ``True``, the function expects `scaled` Fourier coefficients, i.e.
        the non-bias term (all non-zero frequency terms) have been multiplied
        by a factor of 2. Hence, taking the real part of the spectrum
        multiplied with the complex exponentials results in the correctly
        scaled and time-shifted real-valued harmonics.

    Returns
    -------
    matrix : ndarray, shape (..., m, n)

    """

    if scaled is None:
        scaled = False

    time = np.array(time[..., None, None, None], dtype=np.float)
    frequency = 2*pi*np.array(frequency, dtype=np.float)
    if frequency.ndim == 1:
        frequency.reshape(-1, 1, 1)
    spectrum = np.array(spectrum, dtype=np.complex)

    # output of shape (..., k, n, m)
    freq_t = frequency*time

    # compute complex exponentials
    harmonics = np.empty(freq_t.shape, dtype=np.complex)
    harmonics = np.cos(freq_t) + 1j*np.sin(freq_t)

    if scaled is False:
        # scale non-offset coefficients by 2 before synthesizing matrices
        harmonics = np.where(frequency > 0.0, 2*harmonics, harmonics)

    matrix = np.sum(spectrum*harmonics, axis=-3)

    return np.real(matrix)


def rotate_gauss_fft(nmax, kmax, *, qfunc=None, step=None, N=None, filter=None,
                     save_to=None, reference=None, scaled=None,
                     start_date=None):
    """
    Compute Fourier coefficients of the timeseries of matrices that transform
    spherical harmonic expansions (degree ``kmax``) from a time-dependent
    reference system (GSM, SM) to GEO (degree ``nmax``).

    Parameters
    ----------
    nmax : int
        Maximum degree of spherical harmonic expansion with respect to
        geographic reference (target reference system).
    kmax : int
        Maximum degree of spherical harmonic expansion with respect to rotated
        reference system.
    qfunc : callable
        Callable ``q = qfunc(freq, k)`` that returns the complex q-response
        ``q`` (ndarray, shape (``N``,)) given a frequency vector ``freq``
        (ndarray, shape (``N``,)) in (1/sec) and the index ``k`` (int) counting
        the Gauss coefficients in natural order, i.e. ``k = 0`` is
        :math:`g_1^0`, ``k = 1`` is :math:`g_1^1`, ``k = 2`` is :math:`h_1^1`
        and so on.
    step : float
        Sample spacing given in hours (default is 1.0 hour).
    N : int, optional
        Number of samples for which to evaluate the FFT (default is
        N = 8*365.25*24 equiv. to 8 years using default sample spacing).
    filter : int, optional
        Set filter length, i.e. number of Fourier coefficients to be saved
        (default is ``int(N/2+1)``).
    save_to : str, optional
        Path and file name to store output in npz-format. Defaults to
        ``False``, i.e. no file is written.
    reference : {'gsm', 'sm'}, optional
        Time-dependent reference system (default is GSM).
    scaled : bool, optional (default is ``False``)
        If ``True``, the function returns `scaled` Fourier coefficients, i.e.
        the non-bias terms (all non-zero frequency terms) are multiplied by a
        factor of 2. Hence, taking the real part of the first half of the
        spectrum multiplied with the complex exponentials results in the
        correctly scaled and time-shifted real-valued harmonics.
    start_date : float, optional (defaults to ``0.0``, i.e. Jan 1, 2000)
        Time point from which to compute the time series of coefficient
        matrices in modified Julian date.

    Returns
    -------
    frequency, spectrum, frequency_ind, spectrum_ind : ndarray, \
shape (``filter``, ``nmax`` (``nmax`` + 2), ``kmax`` (``kmax`` + 2))
        Unsorted vector of positive frequencies in 1/days and complex fourier
        spectrum of rotation matrices to transform spherical
        harmonic expansions.

    Notes
    -----
    If ``save_to=<filepath>``, then an ``*.npz``-file is written with the
    keywords {'frequency', 'spectrum', 'frequency_ind', 'spectrum_ind', ...}
    and all the possible keywords. Among them, ``'dipole'`` means the three
    spherical harmonic coefficients of the dipole set in
    ``basicConfig['params.dipole']``.

    About the discrete Fourier transform (DFT) used here:

    A discrete periodic signal :math:`x[n]` with :math:`n\\in [0, N-1]`
    (period of `N`) is represented in terms of complex-exponentials as

    .. math::

       x[n] = \\sum_{k=0}^{N-1}X[k]w_N^{kn}, \\qquad w_N = \\exp(i 2\\pi/N)

    Here, :math:`X[k]`, :math:`k\\in [0, N-1]` is the Fourier transform of
    :math:`x[n]`. The DFT is defined as:

    .. math::

       X[k] = \\frac{1}{N}\\sum_{n=0}^{N-1}x[n]w_N^{-kn}

    In ``numpy``, this operation is implemented with

    .. code-block:: python

       import numpy as np

       X = np.fft.fft(x) / N

    Finally, if ``save_to`` is given, only half of the Fourier coefficients
    ``X[0:int(N/2)+1]`` (right-exclusive) are saved (or less if ``filter`` is
    specified).

    """

    if reference is None:
        reference = 'gsm'

    if step is None:
        step = 1.0  # sample spacing of one hour

    if N is None:
        N = int(8*365.25*24)  # number of samples
    N = int(N)

    if filter is None:  # number of significant Fourier components to be saved
        filter = int(N/2 + 1)
    filter = int(filter)

    if save_to is None:
        save_to = False  # do not write output file

    if scaled is None:
        scaled = False

    if start_date is None:
        start_date = 0.0

    time = np.arange(N) * step / 24. + start_date  # time in days

    # compute base vectors of time-dependent reference system
    if str(reference).lower() == 'gsm':
        base_1, base_2, base_3 = basevectors_gsm(time)
    elif str(reference).lower() == 'sm':
        base_1, base_2, base_3 = basevectors_sm(time)
    else:
        raise ValueError('Reference system must be either "GSM" or "SM".')

    # predefine output matrices, last dimension runs through time
    matrix_time = np.empty((N, nmax*(nmax+2), kmax*(kmax+2)))

    print("Calculating Gauss rotation matrices for {:}".format(
        reference.upper()))

    for k in range(N):
        # compute transformation matrix: reference to geographic system
        matrix_time[k] = rotate_gauss(
            nmax, kmax, base_1[k], base_2[k], base_3[k])
        print("Finished {:.1f}%".format((k+1)/N*100), end='\r')

    print("")

    # DFT and proper scaling
    spectrum_full = np.fft.fft(matrix_time, axis=0) / N
    spectrum_full = spectrum_full[:int(N/2+1)]  # remove aliases

    # oscillations per second
    frequency_full = (np.arange(int(N/2+1)) / N) / step / 3600

    if qfunc is None:
        # compute q-response and keep in memory
        q = q_response(frequency_full, nmax)

        # now define qfunc here
        def qfunc(freq, k):
            # index of degree in response
            n = np.floor(np.sqrt(k+1)-1).astype(int)
            return q[n]

    # predefine output arrays
    frequency = np.empty((filter, nmax*(nmax+2), kmax*(kmax+2)))
    frequency_ind = np.empty((filter, nmax*(nmax+2), kmax*(kmax+2)))
    spectrum = np.empty((filter, nmax*(nmax+2), kmax*(kmax+2)),
                        dtype=np.complex)
    spectrum_ind = np.empty((filter, nmax*(nmax+2), kmax*(kmax+2)),
                            dtype=np.complex)

    for k in range(nmax*(nmax+2)):

        # compute Q-response for freqencies and given Gauss coefficient
        response = qfunc(frequency_full, k)

        for l in range(kmax*(kmax+2)):
            # select specific Fourier coefficients from the rotation matrix
            element = spectrum_full[:, k, l]

            # modify Fourier components with Q-response
            element_ind = response*element

            # index of sorted element spectrum (descending order)
            sort = np.argsort(np.abs(element))[::-1]
            sort = sort[:filter]  # only keep small number of components

            # index of sorted element spectrum (descending order)
            sort_ind = np.argsort(np.abs(element_ind))[::-1]
            sort_ind = sort_ind[:filter]  # only keep small number

            # write sorted frequency (per day) and fourier components to array
            frequency[:, k, l] = frequency_full[sort] * (24*3600)
            frequency_ind[:, k, l] = frequency_full[sort_ind] * (24*3600)
            spectrum[:, k, l] = element[sort]
            spectrum_ind[:, k, l] = element_ind[sort_ind]

    if scaled:
        # scale non-offset coefficients by 2
        spectrum = np.where(frequency == 0.0, spectrum, 2*spectrum)
        spectrum_ind = np.where(frequency_ind == 0.0, spectrum_ind,
                                2*spectrum_ind)

    # save several arrays to binary
    if save_to:
        np.savez(str(save_to),
                 frequency=frequency, spectrum=spectrum,
                 frequency_ind=frequency_ind, spectrum_ind=spectrum_ind,
                 step=step, N=N, filter=filter, reference=reference,
                 scaled=scaled, dipole=basicConfig['params.dipole'],
                 start_date=start_date)
        print("Output saved to {:}".format(save_to))

    return frequency, spectrum, frequency_ind, spectrum_ind


def rotate_gauss(nmax, kmax, base_1, base_2, base_3):
    """
    Compute the matrix to transform the spherical harmonic expansion from a
    rotated coordinate system to the standard geographic reference (GEO).

    The rotated coordinate system is described by 3 orthogonal
    base vectors with components in GEO coordinates.

    Parameters
    ----------
    nmax : int
        Maximum degree of spherical harmonic expansion with respect to
        geographic reference (target reference system).
    kmax : int
        Maximum degree of spherical harmonic expansion with respect to rotated
        reference system.
    base_1, base_2, base_3 : ndarray, shape (..., 3)
        Base vectors of rotated reference system given in terms of the
        target reference system. Vectors reside in the last dimension.

    Returns
    -------
    matrix : ndarray, shape (..., ``nmax`` (``nmax`` + 2), ``kmax`` \
(``kmax`` + 2))
        Matrices reside in last two dimensions. They transform spherical
        harmonic coefficients of rotated reference (e.g. GSM) to standard
        geographic reference (GEO):

        [g10 g11 h11 ...]_geo = M * [g10 g11 h11 ...]_gsm

    """

    assert (base_1.shape == base_2.shape) and (base_1.shape == base_3.shape)
    time_shape = base_1.shape[:-1]  # retain original shape of grid

    # predefine output array
    matrix_time = np.empty(time_shape + (nmax**2+2*nmax, kmax**2+2*kmax))

    # define Gauss-Legendre grid for surface integration
    n_theta = ceil((nmax + kmax + 1)/2)  # number of points in colatitude
    n_phi = 2*n_theta  # number of points in azimuth

    # integrates polynomials of degree 2*n_theta-1 exactly
    x, weights = np.polynomial.legendre.leggauss(n_theta)
    theta = degrees(np.arccos(x))
    phi = np.arange(n_phi) * degrees(2*pi)/n_phi

    # compute Schmidt quasi-normalized associated Legendre functions and
    # corresponding normalization
    Pnm = legendre_poly(nmax, theta)
    n_Pnm = int((nmax**2+3*nmax)/2)
    norm = np.empty((n_Pnm,))
    for n in range(1, nmax+1):
        lower = int((n**2+n)/2-1)
        upper = int((n**2+3*n)/2)
        norm[lower] = 2/(2*n+1)  # inner product of Pn0
        norm[lower+1:upper] = 4/(2*n+1)  # inner product of Pnm m>0

    # generate grid of rotated reference system
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # run over time index and produce matrix for every point in time
    for index in np.ndindex(time_shape):

        # predefine array size for each point in time
        matrix = np.empty((nmax*(nmax+2), kmax*(kmax+2)))

        theta_ref, phi_ref = geo_to_base(
            theta_grid, phi_grid, base_1[index], base_2[index], base_3[index])

        # compute Schmidt quasi-normalized associated Legendre functions on
        # grid in rotated reference system: theta_ref, phi_ref
        Pnm_ref = legendre_poly(kmax, theta_ref)

        # compute powers of complex exponentials
        nphi_ref = radians(np.multiply.outer(np.arange(kmax+1), phi_ref))
        exp_ref = np.cos(nphi_ref) + 1j*np.sin(nphi_ref)

        # loop over columns of matrix
        col = 0  # index of column
        for k in range(1, kmax+1):

            # l = 0
            sh_ref = Pnm_ref[k, 0]*exp_ref[0]  # cplx spherical harmonic
            fft_c = np.fft.fft(sh_ref.real) / n_phi  # only real part non-zero

            # SH analysis: write column of matrix, row by row
            row = 0  # index of row
            for n in range(1, nmax+1):

                lower = int((n**2+n)/2-1)  # index for Pnm norm

                #  m = 0: colatitude integration using Gauss weights
                coeff = np.sum(fft_c[:, 0]*Pnm[n, 0]*weights) / norm[lower]
                matrix[row, col] = coeff.real
                row += 1

                # m > 0
                for m in range(1, n+1):
                    coeff = (np.sum(2*fft_c[:, m]*Pnm[n, m]*weights) /
                             norm[lower+m])
                    matrix[row, col] = coeff.real
                    matrix[row+1, col] = -coeff.imag
                    row += 2

            col += 1  # update index of column

            # l > 0
            for l in range(1, k+1):
                sh_ref = Pnm_ref[k, l]*exp_ref[l]
                fft_c = np.fft.fft(sh_ref.real) / n_phi
                fft_s = np.fft.fft(sh_ref.imag) / n_phi

                # SH analysis: write column of R, row by row
                row = 0  # index of row
                for n in range(1, nmax+1):

                    lower = int((n**2+n) / 2-1)  # index for Pnm norm

                    # cosine part
                    coeff = np.sum(fft_c[:, 0]*Pnm[n, 0]*weights)/norm[lower]
                    matrix[row, col] = coeff.real

                    # sine part
                    coeff = np.sum(fft_s[:, 0]*Pnm[n, 0]*weights)/norm[lower]
                    matrix[row, col+1] = coeff.real

                    row += 1  # update row index

                    # m > 0
                    for m in range(1, n+1):
                        # cosine part
                        coeff = (np.sum(2*fft_c[:, m]*Pnm[n, m]*weights) /
                                 norm[lower+m])
                        matrix[row, col] = coeff.real
                        matrix[row+1, col] = -coeff.imag

                        # sine part
                        coeff = (np.sum(2*fft_s[:, m]*Pnm[n, m]*weights) /
                                 norm[lower+m])
                        matrix[row, col+1] = coeff.real
                        matrix[row+1, col+1] = -coeff.imag

                        row += 2  # update row index

                col += 2  # update column index

        matrix_time[index] = matrix  # write rotation matrix into output

    return matrix_time


def sun_position(time):
    """
    Computes the sun's position in longitude and colatitude given time
    (mjd2000).

    It is accurate for years 1901 through 2099, to within 0.006 deg.
    Input shape is preserved.

    Parameters
    ----------
    time : ndarray, shape (...)
        Time given as modified Julian date, i.e. with respect to the date 0h00
        January 1, 2000 (mjd2000).

    Returns
    -------
    theta : ndarray, shape (...)
        Geocentric colatitude of sun's position in degrees
        :math:`[0^\\circ, 180^\\circ]`.
    phi : ndarray, shape (...)
        Geocentric east longtiude of sun's position in degrees
        :math:`(-180^\\circ, 180^\\circ]`.

    References
    ----------
    Taken from `here <http://jsoc.stanford.edu/doc/keywords/Chris_Russel/
    Geophysical%20Coordinate%20Transformations.htm#appendix2>`_

    """
    rad = pi / 180
    year = 2000  # reference year for mjd2000
    assert np.all((year + time // 365.25) < 2099) \
        and np.all((year - time // 365.25) > 1901), \
        ("Time must be between 1901 and 2099.")

    frac_day = np.remainder(time, 1)  # decimal fraction of a day
    julian_date = 365 * (year-1900) + (year-1901)//4 + time + 0.5

    t = julian_date/36525
    v = np.remainder(279.696678 + 0.9856473354*julian_date, 360.)
    g = np.remainder(358.475845 + 0.985600267*julian_date, 360.)

    slong = v + (1.91946 - 0.004789*t)*np.sin(g*rad) + 0.020094*np.sin(2*g*rad)
    obliq = (23.45229 - 0.0130125*t)
    slp = (slong - 0.005686)

    sind = np.sin(obliq*rad)*np.sin(slp*rad)
    cosd = np.sqrt(1.-sind**2)

    #  sun's declination in radians
    declination = np.arctan(sind/cosd)
    # sun's right right ascension in radians (0, 2*pi)
    right_ascension = pi - np.arctan2(sind/(cosd * np.tan(obliq*rad)),
                                      -np.cos(slp*rad)/cosd)
    # Greenwich mean siderial time in radians (0, 2*pi)
    gmst = np.remainder(279.690983 + 0.9856473354*julian_date
                        + 360.*frac_day + 180., 360.) * rad

    theta = degrees(pi/2 - declination)  # convert to colatitude
    phi = center_azimuth(degrees(right_ascension - gmst))

    return theta, phi


def zenith_angle(time, theta, phi):
    """
    Compute the solar zenith angle.

    Parameters
    ----------
    time : ndarray, shape (...)
        Time in modified Julian date.
    theta : ndarray, shape (...)
        Colatitude in degrees.
    phi : ndarray, shape (...)
        Longitude in degrees.

    Returns
    -------
    zeta : ndarray, shape (...)
        Zenith angle in degrees :math:`[0^\\circ, 180^\\circ]` (angle between
        the local zenith and the center of the solar disc). Solar elevation
        angle is then computed by :math:`90^\\circ - \\theta_\\mathrm{zenith}`.

    """

    theta_sun, phi_sun = sun_position(time)

    colat = radians(theta_sun)
    azim = radians(phi_sun)
    theta = radians(theta)
    phi = radians(phi)

    cos_zeta = (np.cos(theta)*np.cos(colat) +
                np.sin(theta)*np.sin(colat)*np.cos(azim - phi))

    return degrees(np.arccos(cos_zeta))


def spherical_to_cartesian(radius, theta, phi):
    """
    Convert geocentric spherical to cartesian coordinates.

    Parameters
    ----------
    radius : float or ndarray, shape (...)
        Geocentric radius.
    theta : float or ndarray, shape (...)
        Geocentric colatitude in degrees.
    phi : float or ndarray, shape (...)
        Geocentric longitude in degrees.

    Returns
    -------
    x, y, z : float or ndarray, shape(...)
        Cartesian coordinates.

    """

    theta, phi = radians(theta), radians(phi)

    x = np.array(radius) * np.cos(phi) * np.sin(theta)
    y = np.array(radius) * np.sin(phi) * np.sin(theta)
    z = np.array(radius) * np.cos(theta)

    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Convert geocentric cartesian to spherical coordinates.

    Parameters
    ----------
    x, y, z : float or ndarray, shape (...)

    Returns
    -------
    radius : float or ndarray, shape (...)
        Geocentric radius.
    theta : float or ndarray, shape (...)
        Geocentric colatitude in degrees :math:`[0^\\circ, 180^\\circ]`.
    phi : float or ndarray, shape (...)
        Geocentric longitude in degrees :math:`(-180^\\circ,180^\\circ]`.
    """

    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)

    return radius, degrees(theta), degrees(phi)


def gg_to_geo(height, beta):
    """
    Compute geocentric colatitude and radius from geodetic colatitude and
    height.

    Parameters
    ----------
    height : ndarray, shape (...)
        Altitude in kilometers.
    beta : ndarray, shape (...)
        Geodetic colatitude

    Returns
    -------
    radius : ndarray, shape (...)
        Geocentric radius in kilometers.
    theta : ndarray, shape (...)
        Geocentric colatitude in degrees.

    References
    ----------
    Equations (51)-(53) from "The main field" (chapter 4) by Langel, R. A. in:

    "Geomagnetism", Volume 1, Jacobs, J. A., Academic Press, 1987.

    """

    a = basicConfig['params.ellipsoid'][0]  # equatorial radius
    b = basicConfig['params.ellipsoid'][1]  # polar radius

    # convert geodetic colatitude to latitude
    alpha = radians(90. - beta)

    sin_alpha_2 = np.sin(alpha)**2
    cos_alpha_2 = np.cos(alpha)**2

    factor = height*np.sqrt(a**2*cos_alpha_2 + b**2*sin_alpha_2)
    gamma = np.arctan2((factor + b**2)*np.tan(alpha), (factor + a**2))

    theta = 90. - degrees(gamma)
    radius = np.sqrt(height**2 + 2*factor +
                     a**2*(1. - (1. - (b/a)**4)*sin_alpha_2) /
                          (1. - (1. - (b/a)**2)*sin_alpha_2))

    return radius, theta


def geo_to_gg(radius, theta):
    """
    Compute geodetic colatitude and vertical height above the ellipsoid from
    geocentric radius and colatitude.

    Parameters
    ----------
    radius : ndarray, shape (...)
        Geocentric radius in kilometers.
    theta : ndarray, shape (...)
        Geocentric colatitude in degrees.

    Returns
    -------
    height : ndarray, shape (...)
        Altitude in kilometers.
    beta : ndarray, shape (...)
        Geodetic colatitude

    Notes
    -----
    Round-off errors might lead to a failure of the algorithm especially but
    not exclusively for points close to the geographic poles. Corresponding
    geodetic coordinates are returned as NaN.

    References
    ----------
    Function uses Heikkinen's algorithm taken from:

    Zhu, J., "Conversion of Earth-centered Earth-fixed coordinates to geodetic
    coordinates", IEEE Transactions on Aerospace and Electronic Systems, 1994,
    vol. 30, num. 3, pp. 957-961

    """

    a = basicConfig['params.ellipsoid'][0]  # equatorial radius
    b = basicConfig['params.ellipsoid'][1]  # polar radius

    a2 = a**2
    b2 = b**2

    e2 = (a2 - b2) / a2  # squared eccentricity
    e4 = e2*e2
    ep2 = (a2 - b2) / b2  # squared primed eccentricity

    r = radius * np.sin(radians(theta))
    z = radius * np.cos(radians(theta))

    r2 = r**2
    z2 = z**2

    F = 54*b2*z2

    G = r2 + (1. - e2)*z2 - e2*(a2 - b2)

    c = e4*F*r2 / G**3

    s = (1. + c + np.sqrt(c**2 + 2*c))**(1./3)

    P = F / (3*(s + 1./s + 1.)**2 * G**2)

    Q = np.sqrt(1. + 2*e4*P)

    r0 = -P*e2*r / (1. + Q) + np.sqrt(
        0.5*a2*(1. + 1./Q) - P*(1. - e2)*z2 / (Q*(1. + Q)) - 0.5*P*r2)

    U = np.sqrt((r - e2*r0)**2 + z2)

    V = np.sqrt((r - e2*r0)**2 + (1. - e2)*z2)

    z0 = b2*z/(a*V)

    height = U*(1. - b2 / (a*V))

    beta = 90. - degrees(np.arctan2(z + ep2*z0, r))

    return height, beta


def basevectors_gsm(time, dipole=None):
    """
    Computes the unit base vectors of the gsm coordinate system with respect to
    the geocentric coordinate system.

    Parameters
    ----------
    time : float or ndarray, shape (...)
        Time given as modified Julian date, i.e. with respect to the date 0h00
        January 1, 2000 (mjd2000).
    dipole : ndarray, shape (3,), optional
        Dipole spherical harmonics :math:`g_1^0`, :math:`g_1^1` and
        :math:`h_1^1`. Defaults to ``basicConfig['params.dipole']``.

    Returns
    -------
    gsm_1, gsm_2, gsm_3 : ndarray, shape (..., 3)
        GSM unit base vectors. The leading dimension agrees with the shape of
        ``time``, while the last dimension contains the unit vector
        components in terms of the geocentric coordinate system.
    """

    if dipole is None:
        dipole = basicConfig['params.dipole']

    vec = _dipole_to_unit(dipole)

    # get sun's position at specified times
    theta_sun, phi_sun = sun_position(time)

    # compute sun's position
    x_sun, y_sun, z_sun = spherical_to_cartesian(1, theta_sun, phi_sun)

    # create array in which the first unit vector resides in last dimension
    gsm_1 = np.empty(x_sun.shape + (3,))
    gsm_1[..., 0] = x_sun
    gsm_1[..., 1] = y_sun
    gsm_1[..., 2] = z_sun

    # compute second base vector of GSM using the cross product of the
    # dipole unit vector with the first unit base vector

    gsm_2 = np.cross(vec, gsm_1)  # over last dimension by default
    norm_gsm_2 = np.linalg.norm(gsm_2, axis=-1, keepdims=True)
    gsm_2 = gsm_2 / norm_gsm_2

    # compute third unit base vector using the cross product of first and
    # second unit base vector
    gsm_3 = np.cross(gsm_1, gsm_2)

    return gsm_1, gsm_2, gsm_3


def basevectors_sm(time, dipole=None):
    """
    Computes the unit base vectors of the sm coordinate system with respect to
    the geocentric coordinate system.

    Parameters
    ----------
    time : float or ndarray, shape (...)
        Time given as modified Julian date, i.e. with respect to the date 0h00
        January 1, 2000 (mjd2000).
    dipole : ndarray, shape (3,), optional
        Dipole spherical harmonics :math:`g_1^0`, :math:`g_1^1` and
        :math:`h_1^1`. Defaults to ``basicConfig['params.dipole']``.

    Returns
    -------
    sm_1, sm_2, sm_3 : ndarray, shape (..., 3)
        SM unit base vectors. The leading dimension agrees with the shape of
        ``time``, while the last dimension contains the unit vector
        components in terms of the geocentric coordinate system.

    """

    if dipole is None:
        dipole = basicConfig['params.dipole']

    vec = _dipole_to_unit(dipole)

    # get sun's position at specified times and convert to cartesian
    theta_sun, phi_sun = sun_position(time)
    x_sun, y_sun, z_sun = spherical_to_cartesian(1, theta_sun, phi_sun)

    # create array in which the sun's vector resides in last dimension
    s = np.empty(x_sun.shape + (3,))
    s[..., 0] = x_sun
    s[..., 1] = y_sun
    s[..., 2] = z_sun

    # set third unit base vector of SM to dipole unit vector
    sm_3 = np.empty(x_sun.shape + (3,))
    sm_3[..., 0] = vec[0]
    sm_3[..., 1] = vec[1]
    sm_3[..., 2] = vec[2]

    # compute second base vector of SM using the cross product of the IGRF
    # dipole unit vector and the sun direction vector
    sm_2 = np.cross(sm_3, s)
    norm_sm_2 = np.linalg.norm(sm_2, axis=-1, keepdims=True)
    sm_2 = sm_2 / norm_sm_2

    # compute third unit base vector using the cross product of second and
    # third unit base vector
    sm_1 = np.cross(sm_2, sm_3)

    return sm_1, sm_2, sm_3


def basevectors_mag(dipole=None):
    """
    Computes the unit base vectors of the central-dipole coordinate system
    (sometimes referred to as MAG) with respect to the geocentric coordinate
    system.

    Parameters
    ----------
    dipole : ndarray, shape (3,), optional
        Dipole spherical harmonics :math:`g_1^0`, :math:`g_1^1` and
        :math:`h_1^1`. Defaults to ``basicConfig['params.dipole']``.

    Returns
    -------
    mag_1, mag_2, mag_3 : ndarray, shape (3,)
        MAG unit base vectors.

    """

    if dipole is None:
        dipole = basicConfig['params.dipole']

    mag_3 = _dipole_to_unit(dipole)

    mag_2 = np.cross(np.array([0., 0., 1.]), mag_3)
    mag_2 = mag_2 / np.linalg.norm(mag_2)

    mag_1 = np.cross(mag_2, mag_3)

    return mag_1, mag_2, mag_3


def basevectors_use(theta, phi):
    """
    Computes the unit base vectors of local orthogonal coordinate system USE
    (Up-South-East) on spherical surface (theta, phi) with respect to the
    geocentric coordinate system.

    Parameters
    ----------
    theta : ndarray, shape (...)
        Geocentric colatitude in degrees :math:`(0^\\circ, 180^\\circ)`, i.e.
        exclude poles.
    phi : ndarray, shape (...)
        Geocentric longitude in degrees.

    Returns
    -------
    use_1, use_2, use_3 : ndarray, shape (..., 3)
        USE unit base vectors. The leading dimension agrees with the shape of
        ``theta`` or ``phi``, while the last dimension contains the unit
        vector components in terms of the geocentric coordinate system.

    """

    theta = np.array(radians(theta))
    phi = np.array(radians(phi))

    assert np.amin(theta) > 0 and np.amax(theta) < pi, "Not defined at poles."

    grid_shape = max(theta.shape, phi.shape)

    # predefine output, the components of the base vectors in the last
    # dimensions: shape (..., 3, 3)
    use_1 = np.empty(grid_shape + (3,))
    use_2 = np.empty(grid_shape + (3,))
    use_3 = np.empty(grid_shape + (3,))

    # calculate and save sin/cos of angles
    sin_phi = np.sin(phi)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)

    # first base vector (Up)
    use_1[..., 0] = sin_theta*cos_phi
    use_1[..., 1] = sin_theta*sin_phi
    use_1[..., 2] = cos_theta

    # second base vector (South)
    use_2[..., 0] = cos_theta*cos_phi
    use_2[..., 1] = cos_theta*sin_phi
    use_2[..., 2] = -sin_theta

    # third base vector (East)
    use_3[..., 0] = -sin_phi
    use_3[..., 1] = cos_phi
    use_3[..., 2] = 0

    return use_1, use_2, use_3


def geo_to_base(theta, phi, base_1, base_2, base_3, inverse=None):
    """
    Transforms spherical geographic coordinates into spherical coordinates of a
    rotated reference system as given by three base vectors.

    Parameters
    ----------
    theta : float or ndarray, shape (...)
        Geocentric colatitude in degrees.
    phi : float or ndarray, shape (...)
        Geocentric longitude in degrees.
    base_1, base_2, base_3 : ndarray, shape (3,) or (..., 3)
        Base vector 1 through 3 having components with respect to GEO.
    inverse : bool
        Use inverse transformation instead, i.e. transform from rotated to
        geographic (default is False).

    Returns
    -------
    theta : ndarray, shape (...)
        Reference colatitude in degrees :math:`[0^\\circ, 180^\\circ]`.
    phi : ndarray, shape (...)
        Reference longitude in degrees :math:`(-180^\\circ, 180^\\circ]`.

    See Also
    --------
    transform_points

    """

    inverse = False if inverse is None else inverse

    # convert spherical to cartesian (radius = 1) coordinates
    x, y, z = spherical_to_cartesian(1, theta, phi)

    if inverse:
        # components of unit base vectors are the columns of inverse matrix
        x_ref = base_1[..., 0]*x + base_2[..., 0]*y + base_3[..., 0]*z
        y_ref = base_1[..., 1]*x + base_2[..., 1]*y + base_3[..., 1]*z
        z_ref = base_1[..., 2]*x + base_2[..., 2]*y + base_3[..., 2]*z

    else:
        # components of unit base vectors are the rows of the rotation matrix
        x_ref = base_1[..., 0]*x + base_1[..., 1]*y + base_1[..., 2]*z
        y_ref = base_2[..., 0]*x + base_2[..., 1]*y + base_2[..., 2]*z
        z_ref = base_3[..., 0]*x + base_3[..., 1]*y + base_3[..., 2]*z

    # convert to spherical coordinates, discard radius as it is 1.
    r_ref, theta_ref, phi_ref = cartesian_to_spherical(x_ref, y_ref, z_ref)

    # assert that radius left unchanged
    np.testing.assert_allclose(r_ref, np.broadcast_to(1, r_ref.shape))

    return theta_ref, phi_ref


def transform_points(theta, phi, time=None, *, reference=None, inverse=None,
                     dipole=None):
    """
    Transforms spherical geographic coordinates into spherical coordinates of
    the target coordinate system.

    Parameters
    ----------
    theta : float or ndarray, shape (...)
        Geocentric colatitude in degrees.
    phi : float or ndarray, shape (...)
        Geocentric longtiude in degrees.
    time : float or ndarray, shape (...)
        Time given as modified Julian date, i.e. with respect to the date 0h00
        January 1, 2000 (mjd2000). Ignored for ``reference='mag'``.
    reference : {'gsm', 'sm', 'mag'}
        Target coordinate system for points provided in GEO
    inverse : bool
        Use inverse transformation instead, i.e. transform from rotated
        coordinates to geographic (default is False).
    dipole : ndarray, shape (3,), optional
        Dipole spherical harmonics :math:`g_1^0`, :math:`g_1^1` and
        :math:`h_1^1`. Defaults to ``basicConfig['params.dipole']``.

    Returns
    -------
    theta : ndarray, shape (...)
        Target reference colatitude in degrees :math:`[0^\\circ, 180^\\circ]`.
    phi : ndarray, shape (...)
        Target reference longitude in degrees
        :math:`(-180^\\circ, 180^\\circ]`.

    See Also
    --------
    geo_to_base

    """

    reference = str(reference).lower()
    inverse = False if inverse is None else inverse

    if dipole is None:
        dipole = basicConfig['params.dipole']

    if reference == 'gsm':
        # compute GSM base vectors
        base_1, base_2, base_3 = basevectors_gsm(time, dipole=dipole)

    elif reference == 'sm':
        # compute SM base vectors
        base_1, base_2, base_3 = basevectors_sm(time, dipole=dipole)

    elif reference == 'mag':
        # compute centered dipole base vectors
        base_1, base_2, base_3 = basevectors_mag()

    else:
        raise ValueError('Wrong reference system. Use one of '
                         '{"gsm", "sm", "mag"}.')

    if inverse:
        theta_base, phi_base = geo_to_base(
            theta, phi, base_1, base_2, base_3, inverse=True)

    else:
        theta_base, phi_base = geo_to_base(theta, phi, base_1, base_2, base_3)

    return theta_base, phi_base


def matrix_geo_to_base(theta, phi, base_1, base_2, base_3, inverse=None):
    """
    Computes matrix to rotate vectors from USE frame at spherical geographic
    coordinates (theta, phi) to USE frame at spherical reference coordinates
    defined by a set of base vectors.

    Parameters
    ----------
    theta : float or ndarray, shape (...)
        Geocentric colatitude in degrees.
    phi : float or ndarray, shape (...)
        Geocentric longtiude in degrees.
    base_1, base_2, base_3 : ndarray, shape (..., 3)
        Base vectors 1 through 3 as columns with respect to GEO.
    inverse : bool
        Use inverse transformation instead, i.e. transform from rotated
        coordinates to geographic (default is False).

    Returns
    -------
    theta : ndarray, shape (...)
        Reference colatitude in degrees :math:`[0^\\circ, 180^\\circ]`.
    phi : ndarray, shape (...)
        Reference longitude in degrees :math:`(-180^\\circ, 180^\\circ]`.
    R : ndarray, shape (..., 3, 3), optional
        Array of matrices that rotates vectors B in spherical GEO to the target
        spherical reference. The matrices (3x3) reside in the last two
        dimensions, while the leading dimensions are identical to the input
        grid.

        | B_radius_ref = B_radius
        | B_theta_ref  = R[1, 1]*B_theta + R[1, 2]*B_phi
        | B_phi_ref    = R[2, 1]*B_theta + R[2, 2]*B_phi

    See Also
    --------
    transform_vectors

    """

    inverse = False if inverse is None else inverse

    if inverse:
        theta_ref, phi_ref = theta, phi
        theta, phi = geo_to_base(theta_ref, phi_ref, base_1, base_2,
                                 base_3, inverse=True)
    else:
        theta_ref, phi_ref = geo_to_base(theta, phi, base_1, base_2, base_3)

    # matrix to rotate vector from USE at (theta, phi) to GEO
    R_use_to_geo = np.stack(basevectors_use(theta, phi), axis=-1)

    # rotate vector according to reference system defined by base vectors
    R_geo_to_ref = np.stack((base_1, base_2, base_3), axis=-2)

    # matrix to rotate vector from original USE to reference system
    R_use_to_ref = np.matmul(R_geo_to_ref, R_use_to_geo)

    # matrix to rotate reference to new USE using the transpose
    R_ref_to_use = np.stack(basevectors_use(theta_ref, phi_ref), axis=-2)

    # complete rotation matrix: spherical GEO to spherical reference
    R = np.matmul(R_ref_to_use, R_use_to_ref)

    # assert that radial component is unchanged
    test = R[..., 0, 0]
    np.testing.assert_allclose(test, np.ones(test.shape))

    if inverse:
        R = np.swapaxes(R, -2, -1)  # transpose matrices
        theta_ref, phi_ref = theta, phi  # overwrite for correct output

    return theta_ref, phi_ref, R


def transform_vectors(theta, phi, B_theta, B_phi, time=None, reference=None,
                      inverse=None, dipole=None):
    """
    Transforms vectors with components in USE (Up-South-East) at
    spherical geographic coordinates (theta, phi) to components in USE at the
    sphercial coordinates of the rotated target coordinate system.

    Parameters
    ----------
    theta : float or ndarray, shape (...)
        Geocentric colatitude in degrees.
    phi : float or ndarray, shape (...)
        Geocentric longitude in degrees.
    B_theta : float or ndarray, shape (...)
        Colatitude vector components.
    B_phi : float or ndarray, shape (...)
        Azimuthal vector components.
    time : float or ndarray, shape (...)
        Time given as modified Julian date, i.e. with respect to the date 0h00
        January 1, 2000 (mjd2000). Ignored for ``reference='mag'``.
    reference : {'gsm', 'sm', 'mag'}
        Target coordinate system.
    inverse : bool
        Use inverse transformation instead, i.e. transform from rotated
        coordinates to geographic (default is False).
    dipole : ndarray, shape (3,), optional
        Dipole spherical harmonics :math:`g_1^0`, :math:`g_1^1` and
        :math:`h_1^1`. Defaults to ``basicConfig['params.dipole']``.

    Returns
    -------
    theta : ndarray, shape (...)
        Target reference colatitude in degrees :math:`[0^\\circ, 180^\\circ]`.
    phi : ndarray, shape (...)
        Target reference longitude in degrees
        :math:`(-180^\\circ, 180^\\circ]`.
    B_theta : float or ndarray, shape (...)
        Colatitude vector components in the target reference.
    B_phi : float or ndarray, shape (...)
        Azimuthal vector components in the target reference.

    See Also
    --------
    matrix_geo_to_base

    """

    inverse = False if inverse is None else inverse

    reference = str(reference).lower()

    if dipole is None:
        dipole = basicConfig['params.dipole']

    if reference == 'gsm':
        # compute GSM base vectors
        base_1, base_2, base_3 = basevectors_gsm(time, dipole=dipole)

    elif reference == 'sm':
        # compute SM base vectors
        base_1, base_2, base_3 = basevectors_sm(time, dipole=dipole)

    elif reference == 'mag':
        # compute centered dipole base vectors
        base_1, base_2, base_3 = basevectors_mag()

    else:
        raise ValueError('Wrong reference system. Use one of '
                         '{"gsm", "sm", "mag"}.')

    theta_ref, phi_ref, R = matrix_geo_to_base(
        theta, phi, base_1, base_2, base_3, inverse=inverse)

    B_theta_ref = R[..., 1, 1]*B_theta + R[..., 1, 2]*B_phi
    B_phi_ref = R[..., 2, 1]*B_theta + R[..., 2, 2]*B_phi

    return theta_ref, phi_ref, B_theta_ref, B_phi_ref


def center_azimuth(phi):
    """
    Project arbitrary azimuth angles in degrees to semi-open interval
    :math:`(-180^\\circ, 180^\\circ]`.

    Parameters
    ----------
    phi : ndarray, float
        Azimuth in degrees.

    Returns
    -------
    phi : ndarray, float
        Prime meridian centered azimuth.

    """

    phi = phi % 360.
    try:  # works for ndarray
        phi = np.where(phi > 180., phi - 360., phi)  # centered around prime
    except TypeError:  # catch error if float
        phi += -360. if phi > 180. else 0.

    return phi


def local_time(time, phi):
    """
    Compute local time from the azimuthal distance to prime meridian.

    Parameters
    ----------
    time : float, ndarray
        Time given as modified Julian date.
    phi : float, ndarray
        Azimuth in degrees.

    Returns
    -------
    local : ndarray
        Local time [0, 24).

    """

    return np.remainder(time + phi/360, 1)*24


def q_response_1D(periods, sigma, radius, n, kind=None):
    """
    Computes the response for a spherically layered conductor in an
    inducing external field of a single spherical harmonic degree.

    Parameters
    ----------
    periods : ndarray or float, shape (m,)
        Oscillation period of the inducing field in seconds.
    sigma : ndarray, shape (k,)
        Conductivity of spherical shells, starting with the outermost in (S/m).
    radius : ndarray, shape (k,)
        Radius of the interfaces in between the layers, starting with outermost
        layer in kilometers (i.e. conductor surface, see Notes).
    n : int
        Spherical degree of inducing external field.
    kind : {'quadratic', 'constant'}, optional
        Approximation for "quadratic" layers (layers of sigma with inverse
        quadratic dependence on radius) or "constant" layers (layers of
        constant sigma, last layer will be set to infinity irrespective of its
        value in ``sigma[-1]``).

    Returns
    -------
    C : ndarray, shape (m,)
        C-response in (km), complex.
    rho_a : ndarray, shape (m,)
        Electrical surface resistance in (:math:`\\Omega m`).
    phi : ndarray, shape (m,)
        Proportional to phase angle of C-response in degrees.
    Q : ndarray, shape (m,)
        Q-response, complex.

    Notes
    -----

    Option ``kind='quadratic'``:
        The following shows how the conductivity is defined in the sphercial
        shells:

        | ``radius[0]`` >= `r` > ``radius[1]``: \
            ``sigma[0]`` * ( ``radius[0]`` / `r` ) * * 2
        | ``radius[1]`` >= `r` > ``radius[2]``: \
            ``sigma[1]`` * ( ``radius[1]`` / `r` ) * * 2
        | ...
        | ``radius[k-1]`` >= `r` > 0 : \
            ``sigma[k-1]`` * ( ``radius[k-1]`` / `r` ) * * 2

        Courtesy of A. Grayver. Code based on Kuvshinov & Semenov (2012).

    Option ``kind='constant'``:
        The following shows how the conductivity is defined in the sphercial
        shells:

        | ``radius[0]`` >= `r` > ``radius[1]``: ``sigma[0]``
        | ``radius[1]`` >= `r` > ``radius[2]``: ``sigma[1]``
        | ...
        | ``radius[k-1]`` >= `r` > 0 : ``sigma[k-1]`` = ``np.inf`` \
            (:math:`\\sigma` = `\\inf`)

        There are ``k`` sphercial shells of uniform conductivity with
        radius in (km) and conductivity :math:`\\sigma` in (S/m).

        The last shell corresponds to the sphercial core whose conductivity is
        set to infinity regardless of the provided ``sigma[-1]``.

        The program should work also for very small periods, where it
        models the response of a layered plane conductor

        | Python version: August 2018, Clemens Kloss
        | Matlab version: November 2000, Nils Olsen
        | Original Fortran program: Peter Weidelt

    """

    if kind is None:
        kind = 'quadratic'

    periods = np.array(periods)  # ensure numpy array
    if periods.ndim > 1:
        raise ValueError("Input ``periods`` must be a vector.")

    sigma = np.array(sigma)  # ensure numpy array
    if sigma.ndim > 1:
        raise ValueError("Conductivity ``sigma`` must be a vector.")

    if kind == 'constant':

        nl = radius.size-2  # index of last layer, there are nl+1 layers

        eps = 1.0e-10
        zlimit = 3

        fac1 = factorial(n)
        fac2 = (-1)**n * fac1/(2*n+1)

        # initialze helpers variables and output
        C = np.empty(periods.shape, dtype=np.complex)
        z = np.empty((2,), dtype=np.complex)
        p = np.empty((2,), dtype=np.complex)
        q = np.empty((2,), dtype=np.complex)
        pd = np.empty((2,), dtype=np.complex)
        qd = np.empty((2,), dtype=np.complex)

        for counter, period in enumerate(periods):
            for il in range(nl, -1, -1):  # runs over nl...0
                k = np.sqrt(8.0e-7 * 1.0j * pi**2 * sigma[il] / period)
                z[0] = k*radius[il]*1000
                z[1] = k*radius[il+1]*1000

                # calculate spherical bessel functions with small argument
                # by power series (abramowitz & Stegun 10.2.5, 10.2.6
                # and 10.2.4):

                if abs(z[0]) < zlimit:
                    for m in range(2):
                        p[m] = 1+0j
                        q[m] = 1+0j
                        pd[m] = n
                        qd[m] = -(n+1)
                        zz = z[m]**2 / 2

                        j = 1
                        dp = 1+0j
                        dq = 1+0j
                        while (abs(dp) > eps or abs(dq) > eps):
                            dp = dp * zz / j / (2*j+1+2*n)
                            dq = dq * zz / j / (2*j-1-2*n)
                            p[m] = p[m] + dp
                            q[m] = q[m] + dq
                            pd[m] = pd[m] + dp*(2*j+n)
                            qd[m] = qd[m] + dq*(2*j-n-1)
                            j += 1

                        p[m] = p[m] * z[m]**n / fac1
                        q[m] = q[m] * z[m]**(-n-1) * fac2
                        q[m] = (-1)**(n+1) * pi/2 * (p[m]-q[m])
                        pd[m] = pd[m] * z[m]**(n-1) / fac1
                        qd[m] = qd[m] * z[m]**(-n-2) * fac2
                        qd[m] = (-1)**(n+1) * pi/2 * (pd[m]-qd[m])

                    v1 = p[1] / p[0]
                    v2 = pd[0] / p[0]
                    v3 = pd[1] / p[0]
                    v4 = q[0] / q[1]
                    v5 = qd[0] / q[1]
                    v6 = qd[1] / q[1]
                else:
                    # calculate spherical bessel functions with large argument
                    # the exponential behaviour is split off and treated
                    # separately (abramowitz & stegun 10.2.9 and 10.2.15)
                    for m in range(2):
                        zz = 2*z[m]
                        rm = 1+0j
                        rp = 1+0j
                        rmd = 1+0j
                        rpd = 1+0j
                        d = 1+0j
                        sg = 1+0j
                        for j in range(1, n+1):
                            d = d * (n+1-j)*(n+j) / j / zz
                            sg = -sg
                            rp = rp + d
                            rm = rm + sg*d
                            rmd = rmd + sg*d*(j+1)
                            rpd = rpd + d*(j+1)

                        e = np.exp(-2*z[m])
                        p[m] = (rm - sg*rp*e) / zz
                        q[m] = (pi/zz) * rp
                        pd[m] = ((rm + sg*rp*e) /
                                 zz - 2*(rmd - sg*rpd*e) / zz**2)
                        qd[m] = -q[m] - 2*pi*rpd / zz**2

                    e = np.exp(-(z[0] - z[1]))
                    v1 = p[1] / p[0] * e
                    v2 = pd[0] / p[0]
                    v3 = pd[1] / p[0] * e
                    v4 = q[0] / q[1] * e
                    v5 = qd[0] / q[1] * e
                    v6 = qd[1] / q[1]

                if (il == nl):
                    b = k*(v2 - v5*v1) / (1 - v4*v1)
                else:
                    b = (k*((v2 - v5*v1)*b + k*(v5*v3-v2*v6)) /
                         ((1 - v4*v1)*b + k*(v4*v3 - v6)))

            C[counter] = radius[0] / (1+1000*radius[0]*b)  # C in km
            print("Finished {:.1f}%".format(
                (counter+1)/periods.size*100), end='\r')

        print('')

        # if nargout > 1
        rho_a = 1e-7*8*pi**2 / periods * np.abs(C*1000)**2
        phi = 90 + 57.3*np.angle(C)
        Q = n/(n+1) * (1 - (n+1)*C/radius[0]) / (1 + n*C/radius[0])

    elif kind == 'quadratic':

        radius = 1e3*np.array(radius)

        # constants
        mu = 4*np.pi*1e-7

        omega = 2*pi*1.0j/periods

        # Number of layers
        N = sigma.size
        M = omega.size

        # Preallocate
        Y = np.zeros((M, N), dtype=np.complex)

        # values for inner sphere, r = N (core)
        qk = -omega*mu*radius[-1]
        bk = np.sqrt((n+0.5)**2 - qk*sigma[-1]*radius[-1])
        bkp = bk + 0.5
        Y[:, -1] = -bkp/qk

        # Loop over all layers above core (from core to surface)
        # from before last (N-2) to first (0)
        for k in range(N-2, -1, -1):
            # Compute temporary scalars
            qk = -omega*mu*radius[k]
            bk = np.sqrt((n+0.5)**2 - qk*sigma[k]*radius[k])
            bkp = bk + 0.5
            bkm = bk - 0.5

            etak = radius[k]/radius[k+1]
            zetak = etak**(2*bk)

            tauk = (1. - zetak) / (1. + zetak)
            # handling of precision overflow due to high frequencies
            tauk[np.isnan(tauk)] = -1.

            qk = -omega*mu*radius[k]
            qk1 = -omega*mu*radius[k+1]
            qY = qk1*Y[:, k+1]

            # Admittance for this layer
            Y[:, k] = 1/qk*(qY*(bk-0.5*tauk)+bkp*bkm*tauk)/(bk+tauk*(0.5+qY))

        # Compute the C-response (in km)
        C = 1/(omega*mu*Y[:, 0])/1e3

        rho_a = mu*omega*np.abs(C*1000)**2  # rho_a in (Ohm*m)
        phi = 90 + 57.3*np.angle(C)  # phase phi in degrees

        # Q-response
        Q = n/(n+1)*(1-(n+1)*C/(radius[0]/1e3))/(1+n*C/(radius[0]/1e3))

    else:
        raise ValueError(f'Unknown option "kind={kind}".')

    return C, rho_a, phi, Q


def q_response(frequency, nmax):
    """
    Computes the Q-response given a conductivity model of Earth.

    The conductivity model is loaded during the computation from
    ``basicConfig['file.Earth_conductivity']``.

    Parameters
    ----------
    frequency : ndarray, shape (N,)
        Vector of N frequencies (1/sec) at which to compute the Q-response.
    nmax : int
        Maximum spherical harmonic degree of inducing field.

    Returns
    -------
    q_response : ndarray, shape (nmax, N)
        Q-response for every frequency and harmonic degree of inducing
        field. Index 0 corresponds to degree 1, index 1 corresponds to degree
        2, and so on.

    """

    # load conductivity model
    filepath = basicConfig['file.Earth_conductivity']
    sigma_model = np.loadtxt(filepath)

    radius_ref = 6371.2  # reference radius in km

    # unpack file: depth and layer conductivity
    # convert depth to radius
    sigma_radius = radius_ref - sigma_model[:, 0]

    # conductivity profile
    sigma = sigma_model[:, 1]

    # find all harmonic terms
    index = frequency > 0.0

    periods = 1 / frequency[index]

    q_response = np.zeros((nmax, frequency.size), dtype=np.complex)
    for n in range(nmax):
        print('Calculating Q-response for degree {:}'.format(n+1))
        # compute Q-response for conductivity model and given degree n
        C_n, rho_n, phi_n, Q_n = q_response_1D(
            periods, sigma, sigma_radius, n+1, kind='quadratic')
        q_response[n, index] = Q_n  # index 0: degree 1, index 1: degree 2, ...

    return q_response
