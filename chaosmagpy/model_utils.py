"""
Functions for building the CHAOS model.

"""

import numpy as np
import warnings
from numpy import radians, degrees
from math import pi
from scipy.interpolate import BSpline, PPoly

R_REF = 6371.2  # reference radius in kilometers


def design_matrix(knots, order, n_tdep, time, radius, theta, phi,
                  n_static=None, source=None):
    """
    Returns matrices that connect radial, colatitude and azimuthal field
    components on a grid with `radius`, `theta` (colatitude) and `phi`
    with the spherical harmonics expansion of a potential. The potential is
    time-dependent on large length-scales (`n` <= `n_tdep`) and static on small
    length-scales (`n_tdep` < `n` <= `n_static`). The time-dependent part uses
    a B-spline representation of order `k`.

    Parameters
    ----------
    knots : ndarray, shape >= (k+1,)
        Knot vector with adequate endpoint multiplicity.
    order : int, positive
        Order `k` of B-spline basis functions (4 = cubic).
    n_tdep : int, positive
        Maximum degree of the time-dependent field.
    time : ndarray, shape (N,)
        Time vector of the `N` data points in days.
    radius : ndarray, shape (N,)
        Radius vector of the `N` data points in kilometers.
    theta : ndarray, shape (N,)
        Data colatitude vector in degrees :math:`[0^\\circ,180^\\circ]`.
    phi : ndarray, shape (N,)
        Data longitude vector in degrees.
    n_static : int, positive, optional
        Maximum degree of static field (default is ``None``,
        `n_static` > `n_tdep`).
    source : {'internal', 'external'}, optional
        Magnetic field source (default is internal).

    Returns
    -------
    G_radius, G_theta, G_phi : ndarray, shape (N, ...)
        Forward matrix for :math:`B_r`, :math:`B_{\\theta}` and
        :math:`B_{\\phi}`.

    Raises
    ------
    ValueError
        If degree of static field is smaller than the degree of the time-dep.
        field. In order to exclude the static field and obtain a purely
        time-dependent potential, use optional argument ``n_static=None`` or
        leave out completely.

    See Also
    --------
    design_gauss

    """

    assert time.shape == radius.shape == theta.shape == phi.shape
    assert time.ndim == 1
    assert np.amin(theta) > 0 and np.amax(theta) < degrees(pi)

    n_data = time.size  # number of data points

    if n_static:
        if n_static < n_tdep:
            raise ValueError("Degree of static field must be greater than "
                             "degree of time-dependent field.")
        elif n_static == n_tdep:
            warnings.warn("Static and time-dependent field are of same "
                          "degree. Ignoring static field.")

    if n_static is None:
        n_static = n_tdep

    if source is None:
        source = 'internal'

    n_coeff_tdep = n_tdep * (n_tdep + 2)  # number of t-dep field coefficients
    n_coeff_static = n_static * (n_static + 2)  # number of static field coeff

    # compute matrices that connect harmonics expansion with field values
    A_radius, A_theta, A_phi = design_gauss(radius, theta, phi,
                                            nmax=n_static, source=source)

    # compute collocation matrix of B-spline segments
    collmat = colloc_matrix(time, knots, order)
    n_segment = collmat.shape[1]  # number of B-spline basis functions

    # allocate memory for output matrices
    G_radius = np.empty((n_data, (n_segment - 1)*n_coeff_tdep
                        + n_coeff_static))
    G_theta = np.empty((n_data, (n_segment - 1)*n_coeff_tdep + n_coeff_static))
    G_phi = np.empty((n_data, (n_segment - 1)*n_coeff_tdep + n_coeff_static))

    n_col = 0  # counter for columns
    for coeff in range(n_coeff_tdep):
        # for each time-dependent expansion coefficient, multiply with
        # collocation matrix use broadcasting to match shapes
        G_radius[:, n_col:n_col+n_segment] = A_radius[:, coeff, None] * collmat
        G_theta[:, n_col:n_col+n_segment] = A_theta[:, coeff, None] * collmat
        G_phi[:, n_col:n_col+n_segment] = A_phi[:, coeff, None] * collmat

        n_col += n_segment

    # add static background field
    G_radius[:, n_col:] = A_radius[:, n_coeff_tdep:n_coeff_static]
    G_theta[:, n_col:] = A_theta[:, n_coeff_tdep:n_coeff_static]
    G_phi[:, n_col:] = A_phi[:, n_coeff_tdep:n_coeff_static]

    return G_radius, G_theta, G_phi


def colloc_matrix(x, knots, order):
    """
    Create collocation matrix of a univariate function on `x` in terms a
    B-spline representation of order `k`. The computation of the splines is
    based on the scipy-package.

    Parameters
    ----------
    x : ndarray, shape (N,)
        `N` points to evaluate the B-spline at.
    knots : ndarray, shape >= (k+1,)
        Vector of knots derived from breaks (with appropriate endpoint
        multiplicity).
    order : int, positive
        Order `k` of the B-spline (4 = cubic).

    Returns
    -------
    collmat : ndarray, shape (N, n-k)
        Collocation matrix, `n` is the size of `knots`.

    See Also
    --------
    augment_breaks
    """

    # create spline using scipy.interpolate
    coll = np.empty((x.size, knots.size - order))

    if order == 1:  # end-points are non-zero for order=1 (bug?)
        for shift in range(knots.size - order):

            index = np.logical_and(x >= knots[shift], x < knots[shift+1])
            coll[:, shift] = np.where(index, 1.0, 0.0)  # put 1.0 else 0.0

    else:
        for shift in range(knots.size - order):
            b = BSpline.basis_element(knots[shift:shift+order+1],
                                      extrapolate=False)

            coll[:, shift] = b(x)

    coll[-1, -1] = 1.0  # include last point
    np.nan_to_num(coll, copy=False)

    # assert that collocation is a partition of one
    np.testing.assert_allclose(coll.sum(axis=-1), np.ones(x.shape))

    return coll


def augment_breaks(breaks, order):
    """
    Augments a vector of break points and returns the knot vector for a
    B-spline representation of order `k`.

    Parameters
    ----------
    breaks: ndarray, shape (n,)
        1-D array, containing `n` break points (without endpoint repeats).
    order: int, positive
        Order `k` of B-spline (4 = cubic).

    Returns
    -------
    knots: ndarray, shape (n+2k-2,)
        1-D array with `k`-times repeated copies of the `breaks`-vector
        endpoints ``breaks[0]`` and ``breaks[-1]``.

    """

    if isinstance(breaks, np.ndarray) and breaks.ndim > 1:
        raise ValueError("Breaks must be a 1-D array.")

    degree = order - 1
    return np.array([breaks[0]]*degree + list(breaks) + [breaks[-1]]*degree)


def synth_from_pp(breaks, order, coeffs, time, radius, theta, phi, *,
                  nmax=None, source=None, deriv=None, grid=None):
    """
    Compute radial, colatitude and azimuthal field components from the magnetic
    potential in terms of a spherical harmonic expansion in form of a
    piecewise polynomial.

    Parameters
    ----------
    breaks : ndarray, shape (m+1,)
        1-D array, containing `m+1` break points (without endpoint repeats) for
        `m` intervals.
    order : int, positive
        Order `k` of piecewise polynomials (4 = cubic).
    coeffs : ndarray, shape (k, m, nmax*(nmax+2))
        Coefficients of the piecewise polynomials, where `m` is the number of
        polynomial pieces. The trailing dimension is equal to the number of
        expansion coefficients for each interval.
    time : ndarray, shape (...)
        Array containing the time in days.
    radius : ndarray, shape (...) or float
        Array containing the radius in kilometers.
    theta : ndarray, shape (...) or float
        Array containing the colatitude in degrees
        :math:`[0^\\circ,180^\\circ]`.
    phi : ndarray, shape (...) or float
        Array containing the longitude in degrees.
    nmax : int, positive, optional
        Maximum degree harmonic expansion (default is given by ``coeffs``,
        but can also be smaller, if specified).
    source : {'internal', 'external'}, optional
        Magnetic field source (default is an internal source).
    deriv : int, positive, optional
        Derivative to be taken (default is 0).
    grid : bool, optional
        If ``True``, field components are computed on a regular grid. Arrays
        ``theta``and ``phi`` must have one dimension less than the output grid
        since the grid will be created as their outer product.

    Returns
    -------
    B_radius, B_theta, B_phi : ndarray, shape (...)
        Radial, colatitude and azimuthal field components.

    See Also
    --------
    synth_values

    """

    # handle optional argument: nmax
    nmax_coeffs = int(np.sqrt(coeffs.shape[-1] + 1) - 1)  # degree for coeffs
    if nmax is None:
        nmax = nmax_coeffs
    elif nmax > nmax_coeffs:
        warnings.warn('Supplied nmax = {0} is incompatible with number of '
                      'model coefficients. Using nmax = {1} instead.'.format(
                        nmax, nmax_coeffs))
        nmax = nmax_coeffs

    # handle optional argument: source
    source = 'internal' if source is None else source

    # compute SH coefficients from pp-form and take derivatives if needed
    deriv = 0 if deriv is None else deriv

    # set grid option to false
    grid = False if grid is None else grid

    PP = PPoly.construct_fast(coeffs[..., :nmax*(nmax+2)], breaks,
                              extrapolate=False)

    PP = PP.derivative(nu=deriv)
    gauss_coeffs = PP(time) * 365.25**deriv

    B_radius, B_theta, B_phi = synth_values(
        gauss_coeffs, radius, theta, phi, nmax=nmax, source=source)

    return B_radius, B_theta, B_phi


def synth_values(coeffs, radius, theta, phi, *,
                 nmax=None, source=None, grid=None):
    """
    Computes radial, colatitude and azimuthal field components from the
    magnetic potential field in terms of spherical harmonic coefficients.

    Parameters
    ----------

    coeffs : ndarray, shape (grid_shape, ...) or (...,)
        Coefficients of the spherical harmonic expansion. The last dimension is
        equal to the number of coefficients at each grid point.
    radius : float or ndarray, shape (grid_shape)
        Array containing the radius in kilometers.
    theta : float or ndarray, shape (grid_shape)
        Array containing the colatitude in degrees
        :math:`[0^\\circ,180^\\circ]`.
    phi : float or ndarray, shape (grid_shape)
        Array containing the longitude in degrees.
    nmax : int, positive, optional
        Maximum degree up to which expansion is to be used (default is given by
        the `coeffs`, but can also be smaller if specified)
    source : {'internal', 'external'}, optional
        Magnetic field source (default is an internal source).
    grid : bool, optional
        If ``True``, field components are computed on a regular grid. Arrays
        ``theta``and ``phi`` must have one dimension less than the output grid
        since the grid will be created as their outer product.

    Returns
    -------
    B_radius, B_theta, B_phi : ndarray, shape (grid_shape)
        Radial, colatitude and azimuthal field components.

    Notes
    -----
    The function can work with different grid shapes, the inputs are
    broadcasted accordingly (``grid=False``, default).

    (1) Single point: grid coordinates are single element arrays, ``coeffs``
        must be a 1-D array of shape (# of coeffs,).
    (2) Several points: grid coordinates are N-D arrays of shape `grid_shape`,
        ``coeffs`` is either an (`grid_shape`, # of coeffs)-D array (each point
        comes with own field expansion) or a 1-D array (points have the same
        field expansion).

    """

    # handle optional argument: nmax
    nmax_coeffs = int(np.sqrt(coeffs.shape[-1] + 1) - 1)  # degree for coeffs
    if nmax is None:
        nmax = nmax_coeffs
    elif nmax > nmax_coeffs:
        warnings.warn('Supplied nmax = {0} is incompatible with number of '
                      'model coefficients. Using nmax = {1} instead.'.format(
                        nmax, nmax_coeffs))
        nmax = nmax_coeffs

    # handle optional argument: source
    if source is None:
        source = 'internal'

    # ensure ndarray inputs
    radius = np.array(radius, dtype=np.float) / R_REF
    theta = np.array(theta, dtype=np.float)
    phi = np.array(phi, dtype=np.float)

    # handle grid option
    grid = False if grid is None else grid

    # check grid shape and compare to supplied coefficients
    if grid is True:
        assert theta.size > 1 and phi.size > 1, 'At least two points.'
        theta_shape = theta.shape + phi.shape
        phi_shape = theta_shape
    else:
        theta_shape = theta.shape
        phi_shape = phi.shape

    grid_shape = max(radius.shape, theta_shape, phi_shape, coeffs.shape[:-1])

    assert (theta_shape == () or
            theta_shape == (1,) or
            theta_shape == grid_shape)
    assert (phi_shape == () or
            phi_shape == (1,) or
            phi_shape == grid_shape)
    assert (radius.shape == () or  # numpy float
            radius.shape == (1,) or  # ndarray with one element
            radius.shape == grid_shape)  # ndarray
    assert (coeffs.shape[:-1] == () or
            coeffs.shape[:-1] == grid_shape)

    assert np.amin(theta) >= 0 and np.amax(theta) <= degrees(pi)

    # initialize radial dependence given the source
    if source == 'internal':
        r_n = radius**(-3)
    elif source == 'external':
        r_n = np.ones(radius.shape)
    else:
        raise ValueError("Source must be either 'internal' or 'external'.")

    # compute associated Legendre polynomials as (n, m, theta-points)-array
    Pnm = legendre_poly(nmax, theta)

    if grid is True:
        Pnm = Pnm[..., None]
        theta = theta[..., None]
        phi = phi[None, ...]

    # save sinth for fast access
    sinth = Pnm[1, 1]

    # calculate cos(m*phi) and sin(m*phi) as (m, phi-points)-array
    phi = radians(phi)
    cmp = np.cos(np.multiply.outer(np.arange(nmax+1), phi))
    smp = np.sin(np.multiply.outer(np.arange(nmax+1), phi))

    # allocate arrays in memory
    B_radius = np.zeros(grid_shape)
    B_theta = np.zeros(grid_shape)
    B_phi = np.zeros(grid_shape)

    num = 0
    for n in range(1, nmax+1):
        if source == 'internal':
            B_radius += (n+1) * Pnm[n, 0] * r_n * coeffs[..., num]
        elif source == 'external':
            B_radius += -n * Pnm[n, 0] * r_n * coeffs[..., num]

        B_theta += -Pnm[0, n+1] * r_n * coeffs[..., num]

        num += 1

        for m in range(1, n+1):
            if source == 'internal':
                B_radius += ((n+1) * Pnm[n, m] * r_n
                             * (coeffs[..., num] * cmp[m]
                                + coeffs[..., num+1] * smp[m]))
            elif source == 'external':
                B_radius += -(n * Pnm[n, m] * r_n
                              * (coeffs[..., num] * cmp[m]
                                 + coeffs[..., num+1] * smp[m]))

            B_theta += (-Pnm[m, n+1] * r_n
                        * (coeffs[..., num] * cmp[m]
                           + coeffs[..., num+1] * smp[m]))

            with np.errstate(divide='ignore', invalid='ignore'):
                # handle poles using L'Hopital's rule
                div_Pnm = np.where(theta == 0., Pnm[m, n+1], Pnm[n, m] / sinth)
                div_Pnm = np.where(theta == degrees(pi), -Pnm[m, n+1], div_Pnm)

            B_phi += (m * div_Pnm * r_n
                      * (coeffs[..., num] * smp[m]
                         - coeffs[..., num+1] * cmp[m]))

            num += 2

        if source == 'internal':
            r_n = r_n / radius  # equivalent to r_n = radius**(-(n+2))
        elif source == 'external':
            r_n = r_n * radius  # equivalent to r_n = radius**(n-1)

    return B_radius, B_theta, B_phi


def design_gauss(radius, theta, phi, nmax, source=None):
    """
    Computes matrices to connect the radial, colatitude and azimuthal field
    components to the magnetic potential field in terms of spherical harmonic
    coefficients (Schmidt quasi-normalized).

    Parameters
    ----------

    radius : ndarray, shape (N,)
        Array containing the radius of `N` data points in kilometers.
    theta : ndarray, shape (N,)
        Array containing the colatitude in degrees
        :math:`[0^\\circ,180^\\circ]`.
    phi : ndarray, shape (N,)
        Array containing the longitude in degrees.
    nmax : int, positive
        Maximum degree of the sphercial harmonic expansion.
    source : {'internal', 'external'}, optional
        Magnetic field source (default is an internal source).

    Returns
    -------
    A_radius, A_theta, A_phi : ndarray, shape (N, ``nmax`` * (``nmax`` + 2))
        Matrices for radial, colatitude and azimuthal field components.

    """

    # ensure ndarray inputs
    radius = np.array(radius, dtype=np.float) / R_REF
    theta = np.array(theta, dtype=np.float)
    phi = np.array(phi, dtype=np.float)

    assert radius.shape == theta.shape == phi.shape
    assert radius.ndim == 1
    assert np.amin(theta) >= 0. and np.amax(theta) <= degrees(pi)

    # set internal source as default
    if source is None:
        source = 'internal'

    # initialize radial dependence of expansion
    if source == 'internal':
        r_n = radius**(-3)
    elif source == 'external':
        r_n = np.ones(radius.shape)
    else:
        raise ValueError("Source must be either 'internal' or 'external'.")

    # compute associated Legendre polynomials as (n, m, theta-points)-array
    Pnm = legendre_poly(nmax, theta)
    sinth = Pnm[1, 1]

    phi = radians(phi)

    # calculate cos(m*phi) and sin(m*phi) as (m, phi-points)-array
    cmp = np.cos(np.outer(np.arange(nmax+1), phi))
    smp = np.sin(np.outer(np.arange(nmax+1), phi))

    # allocate A_radius, A_theta, A_phi in memeory
    A_radius = np.zeros((theta.size, (nmax*(nmax+2))))
    A_theta = np.zeros((theta.size, (nmax*(nmax+2))))
    A_phi = np.zeros((theta.size, (nmax*(nmax+2))))

    num = 0
    for n in np.arange(1, nmax+1):
        if source == 'internal':
            A_radius[:, num] = (n+1) * Pnm[n, 0] * r_n
        elif source == 'external':
            A_radius[:, num] = -n * Pnm[n, 0] * r_n

        A_theta[:, num] = -Pnm[0, n+1] * r_n

        num += 1

        for m in range(1, n+1):
            if source == 'internal':
                A_radius[:, num] = (n+1) * Pnm[n, m] * r_n * cmp[m]
                A_radius[:, num+1] = (n+1) * Pnm[n, m] * r_n * smp[m]
            elif source == 'external':
                A_radius[:, num] = -n * Pnm[n, m] * r_n * cmp[m]
                A_radius[:, num+1] = -n * Pnm[n, m] * r_n * smp[m]

            A_theta[:, num] = -Pnm[m, n+1] * r_n * cmp[m]
            A_theta[:, num+1] = -Pnm[m, n+1] * r_n * smp[m]

            with np.errstate(divide='ignore', invalid='ignore'):
                # handle poles using L'Hopital's rule
                div_Pnm = np.where(theta == 0., Pnm[m, n+1], Pnm[n, m] / sinth)
                div_Pnm = np.where(theta == degrees(pi), -Pnm[m, n+1], div_Pnm)

            A_phi[:, num] = m * div_Pnm * r_n * smp[m]
            A_phi[:, num+1] = -m * div_Pnm * r_n * cmp[m]

            num += 2

        # update radial dependence given the source
        if source == 'internal':
            r_n = r_n / radius
        else:
            r_n = r_n * radius

    return A_radius, A_theta, A_phi


def legendre_poly(nmax, theta):
    """
    Returns associated Legendre polynomials `P(n,m)` (Schmidt quasi-normalized)
    and the derivative `dP(n,m)` vrt. `theta` evaluated at `theta`.

    Parameters
    ----------
    nmax : int, positive
        Maximum degree of the spherical expansion.
    theta : ndarray, shape (...)
        Colatitude in degrees :math:`[0^\\circ, 180^\\circ]`
        of arbitrary shape.

    Returns
    -------
    Pnm : ndarray, shape (n, m, ...)
          Evaluated values and derivatives, grid shape is appended as trailing
          dimensions. `P(n,m)` := ``Pnm[n, m, ...]`` and `dP(n,m)` :=
          ``Pnm[m, n+1, ...]``

    """

    costh = np.cos(radians(theta))
    sinth = np.sqrt(1-costh**2)

    Pnm = np.zeros((nmax+1, nmax+2) + costh.shape)
    Pnm[0, 0] = 1  # is copied into trailing dimenions
    Pnm[1, 1] = sinth  # write theta into trailing dimenions via broadcasting

    rootn = np.sqrt(np.arange(2 * nmax**2 + 1))

    # Recursion relations after Langel "The Main Field" (1987),
    # eq. (27) and Table 2 (p. 256)
    for m in range(nmax):
        Pnm_tmp = rootn[m+m+1] * Pnm[m, m]
        Pnm[m+1, m] = costh * Pnm_tmp

        if m > 0:
            Pnm[m+1, m+1] = sinth*Pnm_tmp / rootn[m+m+2]

        for n in np.arange(m+2, nmax+1):
            d = n * n - m * m
            e = n + n - 1
            Pnm[n, m] = ((e * costh * Pnm[n-1, m] - rootn[d-e] * Pnm[n-2, m])
                         / rootn[d])

    # dP(n,m) = Pnm(m,n+1) is the derivative of P(n,m) vrt. theta
    Pnm[0, 2] = -Pnm[1, 1]
    Pnm[1, 2] = Pnm[1, 0]
    for n in range(2, nmax+1):
        Pnm[0, n+1] = -np.sqrt((n*n + n) / 2) * Pnm[n, 1]
        Pnm[1, n+1] = ((np.sqrt(2 * (n*n + n)) * Pnm[n, 0]
                       - np.sqrt((n*n + n - 2)) * Pnm[n, 2]) / 2)

        for m in np.arange(2, n):
            Pnm[m, n+1] = (0.5*(np.sqrt((n + m) * (n - m + 1)) * Pnm[n, m-1]
                           - np.sqrt((n + m + 1) * (n - m)) * Pnm[n, m+1]))

        Pnm[n, n+1] = np.sqrt(2 * n) * Pnm[n, n-1] / 2

    return Pnm
