"""
This module provide functions for building the CHAOS model and geomagnetic
field models in general.

.. autosummary::
    :toctree: functions

    design_matrix
    design_gauss
    colloc_matrix
    augment_breaks
    pp_from_bspline
    synth_from_pp
    synth_values
    legendre_poly
    power_spectrum
    degree_correlation

"""

import numpy as np
import warnings
from chaosmagpy.config_utils import basicConfig
from scipy.interpolate import BSpline, PPoly


def design_matrix(knots, order, n_tdep, time, radius, theta, phi,
                  n_static=None, source=None):
    """
    Returns matrices that connect radial, colatitude and azimuthal field
    components on a grid with `radius`, `theta` (colatitude) and `phi`
    to the spherical harmonics expansion of a potential.

    The potential is time-dependent on large length-scales (`n` <= `n_tdep`)
    and static on small length-scales (`n_tdep` < `n` <= `n_static`). The
    time-dependent part uses a B-spline representation of order `k`.

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
    assert np.amin(theta) > 0. and np.amax(theta) < 180.

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


def colloc_matrix(x, knots, order, deriv=None):
    """
    Create collocation matrix of a univariate function on `x` in terms  of a
    B-spline representation of order `k`.

    The computation of the splines is based on the scipy-package.

    Parameters
    ----------
    x : ndarray, shape (N,)
        `N` points to evaluate the B-spline at.
    knots : ndarray, shape >= (k+1,)
        Vector of knots derived from breaks (with appropriate endpoint
        multiplicity).
    order : int, positive
        Order `k` of the B-spline (4 = cubic).
    deriv: int, positive, optional
        Derivative of the B-spline partition (defaults to 0).

    Returns
    -------
    collmat : ndarray, shape (N, n-k)
        Collocation matrix, `n` is the size of `knots`.

    See Also
    --------
    augment_breaks

    """

    if deriv is None:
        deriv = 0

    if deriv >= order:
        return np.zeros((max(x.size, 1), knots.size - order))

    else:
        # create spline using scipy.interpolate
        coll = np.empty((max(x.size, 1), knots.size - order))

        for n in range(knots.size - order):
            c = np.zeros(knots.size - order)
            c[n] = 1.

            b = BSpline.construct_fast(knots, c, order-1, extrapolate=False)

            coll[:, n] = b(x, nu=deriv)

        return coll


def augment_breaks(breaks, order):
    """
    Augment a vector of break points and return the knot vector for a
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


def pp_from_bspline(coeffs, knots, order):
    """
    Return a piecewise polynomial from a BSpline representation.

    Parameters
    ----------
    coeffs : ndarray, shape (M, D)
        Bspline coefficients for the `M` B-splines parameterizing
        `D` dimensions.
    knots : ndarray, shape (N,)
        B-spline knots. The knots must have the full endpoint multiplicity.
        Zero-pad spline coefficients if needed.
    order : int
        Order of the B-spline.

    Returns
    -------
    coeffs_pp : ndarray, shape (K, P, D)
        Coefficients of the piecewise polynomial where `K` is the order
        (``order``), `P` is the number of pieces and `D` is the dimension.
    breaks : ndarray, shape (P+1,)
        Break points of the piecewise polynomial.

    """

    degree = order - 1
    breaks = np.unique(knots)
    pieces = breaks.size - 1
    dim = coeffs.shape[-1]

    coeffs_pp = np.empty((order, pieces, dim))

    # have to do it manually for each dimension
    for d in range(dim):

        bs = BSpline(knots, coeffs[:, d], degree)
        pp = PPoly.from_spline(bs, extrapolate=False)

        # remove endpoint multiplicities
        coeffs_pp[:, :, d] = pp.c[:, degree:(degree+pieces)]

    return coeffs_pp, breaks


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
        ``theta`` and ``phi`` must have one dimension less than the output grid
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
        gauss_coeffs, radius, theta, phi, nmax=nmax, source=source, grid=grid)

    return B_radius, B_theta, B_phi


def synth_values(coeffs, radius, theta, phi, *, nmax=None, nmin=None,
                 mmax=None, source=None, grid=None):
    """
    Computes radial, colatitude and azimuthal field components from the
    magnetic potential field in terms of spherical harmonic coefficients.

    Parameters
    ----------

    coeffs : ndarray, shape (..., M)
        Coefficients of the spherical harmonic expansion. The last dimension is
        equal to the number of coefficients.
    radius : float or ndarray, shape (...)
        Array containing the radius in kilometers.
    theta : float or ndarray, shape (...)
        Array containing the colatitude in degrees
        :math:`[0^\\circ,180^\\circ]`.
    phi : float or ndarray, shape (...)
        Array containing the longitude in degrees.
    nmax : int, positive, optional
        Maximum degree up to which expansion is to be used (default is given by
        the last dimension of ``coeffs``, that is, ``M = nmax(nmax+2)``). This
        value can also be smaller if only coefficients at low degree should
        contribute.
    nmin : int, positive, optional
        Minimum degree of the expansion (defaults to 1). This value must
        correspond to the minimum degree of the coefficients and cannot be
        larger.
    mmax : int, positive, optional
        Maximum order of the spherical harmonic expansion (defaults to
        ``nmax``). This value can also be smaller. For example, if only the
        zonal part of the expansion should be used, set ``mmax = 0``.
    source : {'internal', 'external'}, optional
        Magnetic field source (default is an internal source).
    grid : bool, optional
        If ``True``, field components are computed on a regular grid. Arrays
        ``theta`` and ``phi`` must have one dimension less than the output grid
        since the grid will be created as their outer product (defaults to
        ``False``).

    Returns
    -------
    B_radius, B_theta, B_phi : ndarray, shape (...)
        Radial, colatitude and azimuthal field components.

    Warnings
    --------
    The function can also evaluate the field components at the geographic
    poles, i.e. where ``theta == 0.`` or ``theta == 180.``. However, users
    should be careful when doing this because the vector basis for spherical
    geocentric coordinates,
    :math:`{{\\mathbf{e}_r, \\mathbf{e}_\\theta, \\mathbf{e}_\\phi}}`,
    depends on longitude, which is not well defined at the poles. That is,
    at the poles, any value for the longitude maps to the same location in
    euclidean coordinates but gives a different vector basis in spherical
    geocentric coordinates. Nonetheless, by choosing a specific value for the
    longitude at the poles, users can define the vector basis, which then
    establishes the meaning of the spherical geocentric components returned by
    this function. The vector basis for the horizontal components is defined as

    .. math::

        \\mathbf{e}_\\theta &= \\cos\\theta\\cos\\phi\\mathbf{e}_x -
            \\cos\\theta\\sin\\phi\\mathbf{e}_y - \\sin\\theta\\mathbf{e}_z\\\\
        \\mathbf{e}_\\phi &= -\\sin\\phi\\mathbf{e}_x +
            \\cos\\phi\\mathbf{e}_y

    Hence, at the geographic north pole as given by ``theta = 0.`` and
    ``phi = 0.`` (chosen by the user), the returned component ``B_theta``
    will be along the direction :math:`\\mathbf{e}_\\theta = \\mathbf{e}_x` and
    ``B_phi`` along :math:`\\mathbf{e}_\\phi = \\mathbf{e}_y`. However,
    if ``phi = 180.`` is chosen, ``B_theta``
    will be along :math:`\\mathbf{e}_\\theta = -\\mathbf{e}_x` and ``B_phi``
    along :math:`\\mathbf{e}_\\phi = -\\mathbf{e}_y`.

    Notes
    -----
    The function can work with different grid shapes, but the inputs have to
    satisfy NumPy's `broadcasting rules \\
    <https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html>`_
    (``grid=False``, default). This also applies to the dimension of the
    coefficients ``coeffs`` excluding the last dimension.

    The optional parameter ``grid`` is for convenience. If set to ``True``,
    a singleton dimension is appended (prepended) to ``theta`` (``phi``)
    for broadcasting to a regular grid. The other inputs ``radius`` and
    ``coeffs`` must then be broadcastable as before but now with the resulting
    regular grid.

    Examples
    --------
    The most straight forward computation uses a fully specified grid. For
    example, compute the magnetic field at :math:`N=50` grid points on the
    surface.

    .. code-block:: python

      import chaosmagpy.model_utils as cpm
      import numpy as np

      N = 50
      coeffs = np.ones((3,))  # degree 1 coefficients for all points
      radius = 6371.2 * np.ones((N,))  # radius of 50 points in km
      phi = np.linspace(-180., 180., num=N)  # azimuth of 50 points in deg.
      theta = np.linspace(0., 180., num=N)  # colatitude of 50 points in deg.

      B = cpm.synth_values(coeffs, radius, theta, phi)
      print([B[num].shape for num in range(3)])  # output shape (N,)

    Instead of `N` points, compute the field on a regular
    :math:`N\\times N`-grid in azimuth and colatitude (slow).

    .. code-block:: python

      radius_grid = 6371.2 * np.ones((N, N))
      phi_grid, theta_grid = np.meshgrid(phi, theta)  # grid of shape (N, N)

      B = cpm.synth_values(coeffs, radius_grid, theta_grid, phi_grid)
      print([B[num].shape for num in range(3)])  # output shape (N, N)

    But this is slow since some computations on the grid are repeatedly done.
    The preferred method is to use NumPy's broadcasting rules (fast).

    .. code-block:: python

      radius_grid = 6371.2  # float, () or (1,)-shaped array broadcasted to NxN
      theta_grid = theta[:, None]  # append singleton: (N, 1)
      phi_grid = phi[None, :]  # prepend singleton: (1, N)

      B = cpm.synth_values(coeffs, radius_grid, theta_grid, phi_grid)
      print([B[num].shape for num in range(3)])  # output shape (N, N)

    For convenience, you can do the same by using ``grid=True`` option, which
    appends a singleton dimension to ``theta`` and inserts a singleton
    dimension at the second to last position in the shape of ``phi``.

    .. code-block:: python

      B = cpm.synth_values(coeffs, radius_grid, theta, phi, grid=True)
      print([B[num].shape for num in range(3)])  # output shape (N, N)

    Remember that ``grid=False`` (default) will result in
    (N,)-shaped outputs as in the first example.

    """

    # ensure ndarray inputs
    coeffs = np.asarray(coeffs, dtype=float)
    radius = np.asarray(radius, dtype=float) / basicConfig['params.r_surf']
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    theta_min = np.amin(theta)
    theta_max = np.amax(theta)

    if (theta_min <= 0.0) or (theta_max >= 180.0):
        if (theta_min == 0.0) or (theta_max == 180.0):
            warnings.warn('Supplied coordinates include the poles.')
            poles = True
        else:
            raise ValueError('Colatitude outside bounds [0, 180].')
    else:
        poles = False

    nmin = 1 if nmin is None else int(nmin)
    assert nmin > 0, 'The value of "nmin" must be greater than zero.'

    dim = coeffs.shape[-1]

    # handle nmax and mmax
    if (nmax is None) and (mmax is None):
        nmax = int(np.sqrt(dim + nmin**2) - 1)
        mmax = nmax

    elif mmax is None:  # but nmax given
        mmax = nmax

    elif nmax is None:  # but mmax given

        if mmax >= (nmin-1):
            nmax = int((dim - mmax*(mmax+2) + nmin**2 - 1) / (2*mmax+1) + mmax)

        else:
            nmax = int(dim / (2*mmax+1) + nmin - 1)

    if nmax < nmin:
        raise ValueError(f'Nothing to compute: nmax ({nmax}) < nmin ({nmin}).')

    # handle optional argument: source
    if source is None:
        source = 'internal'

    # handle grid option
    grid = False if grid is None else grid

    # manually broadcast input grid on surface
    if grid:
        theta = np.expand_dims(theta, axis=-1)  # append singleton dimension
        phi = np.expand_dims(phi, axis=-2)  # insert singleton before last

    # get shape of broadcasted result
    try:
        b = np.broadcast(
            radius, theta, phi, np.broadcast_to(0, coeffs.shape[:-1]))

    except ValueError:
        print('Cannot broadcast grid shapes (excl. last dimension of coeffs):')
        print(f'radius: {radius.shape}')
        print(f'theta:  {theta.shape}')
        print(f'phi:    {phi.shape}')
        print(f'coeffs: {coeffs.shape[:-1]}')
        raise

    grid_shape = b.shape

    # initialize radial dependence given the source
    if source == 'internal':
        r_n = radius**(-(nmin+2))
    elif source == 'external':
        r_n = radius**(nmin-1)
    else:
        raise ValueError("Source must be either 'internal' or 'external'.")

    # compute associated Legendre polynomials as (n, m, theta-points)-array
    Pnm = legendre_poly(nmax, theta)

    # save sinth for fast access
    sinth = Pnm[1, 1]

    # find the poles
    if poles:
        where_poles = (theta == 0.) | (theta == 180.)

    # convert phi to radians
    phi = np.radians(phi)

    # allocate arrays in memory
    B_radius = np.zeros(grid_shape)
    B_theta = np.zeros(grid_shape)
    B_phi = np.zeros(grid_shape)

    num = 0
    for n in range(nmin, nmax+1):

        if source == 'internal':
            B_radius += (n+1) * Pnm[n, 0] * r_n * coeffs[..., num]
        else:
            B_radius += -n * Pnm[n, 0] * r_n * coeffs[..., num]

        B_theta += -Pnm[0, n+1] * r_n * coeffs[..., num]

        num += 1
        for m in range(1, min(n, mmax)+1):

            cmp = np.cos(m*phi)
            smp = np.sin(m*phi)

            if source == 'internal':
                B_radius += ((n+1) * Pnm[n, m] * r_n
                             * (coeffs[..., num] * cmp
                                + coeffs[..., num+1] * smp))
            else:
                B_radius += -(n * Pnm[n, m] * r_n
                              * (coeffs[..., num] * cmp
                                 + coeffs[..., num+1] * smp))

            B_theta += (-Pnm[m, n+1] * r_n
                        * (coeffs[..., num] * cmp
                           + coeffs[..., num+1] * smp))

            # need special treatment at the poles because Pnm/sinth = 0/0 for
            # theta in {0., 180.},
            # use L'Hopital's rule to take the limit:
            # lim Pnm/sinth = k*dPnm | theta in {0., 180}
            # (evaluated at poles, where k=1 for theta=0 and k=-1 for
            # theta=180.); k = costh = P[1, 0] at poles for convenience
            if poles:
                div_Pnm = np.where(where_poles, Pnm[m, n+1], Pnm[n, m])
                div_Pnm[where_poles] *= Pnm[1, 0][where_poles]
                div_Pnm[~where_poles] /= sinth[~where_poles]
            else:
                div_Pnm = Pnm[n, m] / sinth

            B_phi += (m * div_Pnm * r_n
                      * (coeffs[..., num] * smp
                         - coeffs[..., num+1] * cmp))

            num += 2

        if source == 'internal':
            r_n = r_n / radius  # equivalent to r_n = radius**(-(n+2))
        else:
            r_n = r_n * radius  # equivalent to r_n = radius**(n-1)

    return B_radius, B_theta, B_phi


def design_gauss(radius, theta, phi, nmax, *, nmin=None, mmax=None,
                 source=None):
    """
    Computes matrices to connect the radial, colatitude and azimuthal field
    components to the magnetic potential field in terms of spherical harmonic
    coefficients (Schmidt quasi-normalized).

    Parameters
    ----------

    radius : ndarray, shape (...)
        Array containing the radius in kilometers.
    theta : ndarray, shape (...)
        Array containing the colatitude in degrees
        :math:`[0^\\circ,180^\\circ]`.
    phi : ndarray, shape (...)
        Array containing the longitude in degrees.
    nmax : int, positive
        Maximum degree of the sphercial harmonic expansion.
    nmin : int, positive, optional
        Minimum degree of the sphercial harmonic expansion (defaults to 1).
    mmax : int, positive, optional
        Maximum order of the spherical harmonic expansion (defaults to
        ``nmax``). For ``mmax = 0``, for example, only the zonal terms are
        returned.
    source : {'internal', 'external'}, optional
        Magnetic field source (default is an internal source).

    Returns
    -------
    A_radius, A_theta, A_phi : ndarray, shape (..., M)
        Matrices for radial, colatitude and azimuthal field components. The
        second dimension ``M`` varies depending on ``nmax``, ``nmin`` and
        ``mmax``.

    Warnings
    --------
    The function can also return the design matrix for the field components at
    the geographic poles, i.e. where ``theta == 0.`` or ``theta == 180.``.
    However, users should be careful when doing this because the vector basis
    for spherical geocentric coordinates,
    :math:`{{\\mathbf{e}_r, \\mathbf{e}_\\theta, \\mathbf{e}_\\phi}}`,
    depends on longitude, which is not well defined at the poles. That is,
    at the poles, any value for the longitude maps to the same location in
    euclidean coordinates but gives a different vector basis in spherical
    geocentric coordinates. Nonetheless, by choosing a specific value for the
    longitude at the poles, users can define the vector basis, which then
    establishes the meaning of the spherical geocentric components. The vector
    basis for the horizontal components is defined as

    .. math::

        \\mathbf{e}_\\theta &= \\cos\\theta\\cos\\phi\\mathbf{e}_x -
            \\cos\\theta\\sin\\phi\\mathbf{e}_y - \\sin\\theta\\mathbf{e}_z\\\\
        \\mathbf{e}_\\phi &= -\\sin\\phi\\mathbf{e}_x +
            \\cos\\phi\\mathbf{e}_y

    Hence, at the geographic north pole as given by ``theta = 0.`` and
    ``phi = 0.`` (chosen by the user), the returned design matrix ``A_theta``
    will be for components along the direction
    :math:`\\mathbf{e}_\\theta = \\mathbf{e}_x` and ``A_phi`` for components
    along :math:`\\mathbf{e}_\\phi = \\mathbf{e}_y`. However,
    if ``phi = 180.`` is chosen, ``A_theta`` will be for components along
    :math:`\\mathbf{e}_\\theta = -\\mathbf{e}_x` and ``A_phi``
    along :math:`\\mathbf{e}_\\phi = -\\mathbf{e}_y`.

    """

    # ensure ndarray inputs
    radius = np.asarray(radius, dtype=float) / basicConfig['params.r_surf']
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # get shape of broadcasted result
    try:
        b = np.broadcast(radius, theta, phi)

    except ValueError:
        print('Cannot broadcast grid shapes:')
        print(f'radius: {radius.shape}')
        print(f'theta:  {theta.shape}')
        print(f'phi:    {phi.shape}')
        raise

    shape = b.shape

    theta_min = np.amin(theta)
    theta_max = np.amax(theta)

    # check if poles are included
    if (theta_min <= 0.0) or (theta_max >= 180.0):
        if (theta_min == 0.0) or (theta_max == 180.0):
            warnings.warn('Supplied coordinates include the poles.')
            poles = True
        else:
            raise ValueError('Colatitude outside bounds [0, 180].')
    else:
        poles = False

    # set internal source as default
    if source is None:
        source = 'internal'

    assert nmax > 0, '"nmax" must be greater than zero.'

    nmin = 1 if nmin is None else int(nmin)
    assert nmin <= nmax, '"nmin" must be smaller than "nmax".'
    assert nmin > 0, '"nmin" must be greater than zero.'

    mmax = nmax if mmax is None else int(mmax)
    assert mmax <= nmax, '"mmax" must be smaller than "nmax".'
    assert mmax >= 0, '"mmax" must be greater than or equal to zero.'

    # initialize radial dependence given the source
    if source == 'internal':
        r_n = radius**(-(nmin+2))
    elif source == 'external':
        r_n = radius**(nmin-1)
    else:
        raise ValueError("Source must be either 'internal' or 'external'.")

    # compute associated Legendre polynomials as (n, m, theta-points)-array
    Pnm = legendre_poly(nmax, theta)
    sinth = Pnm[1, 1]

    phi = np.radians(phi)

    # find the poles
    if poles:
        where_poles = (theta == 0.) | (theta == 180.)

    # compute the number of dimensions based on nmax, nmin, mmax
    if mmax >= (nmin-1):
        dim = int(mmax*(mmax+2) + (nmax-mmax)*(2*mmax+1) - nmin**2 + 1)
    else:
        dim = int((nmax-nmin+1)*(2*mmax+1))

    # allocate A_radius, A_theta, A_phi in memeory
    A_radius = np.zeros(shape + (dim,))
    A_theta = np.zeros(shape + (dim,))
    A_phi = np.zeros(shape + (dim,))

    num = 0
    for n in range(nmin, nmax+1):

        if source == 'internal':
            A_radius[..., num] = (n+1) * Pnm[n, 0] * r_n
        else:
            A_radius[..., num] = -n * Pnm[n, 0] * r_n

        A_theta[..., num] = -Pnm[0, n+1] * r_n

        num += 1
        for m in range(1, min(n, mmax)+1):

            cmp = np.cos(m*phi)
            smp = np.sin(m*phi)

            if source == 'internal':
                A_radius[..., num] = (n+1) * Pnm[n, m] * r_n * cmp
                A_radius[..., num+1] = (n+1) * Pnm[n, m] * r_n * smp
            else:
                A_radius[..., num] = -n * Pnm[n, m] * r_n * cmp
                A_radius[..., num+1] = -n * Pnm[n, m] * r_n * smp

            A_theta[..., num] = -Pnm[m, n+1] * r_n * cmp
            A_theta[..., num+1] = -Pnm[m, n+1] * r_n * smp

            # need special treatment at the poles because Pnm/sinth = 0/0 for
            # theta in {0., 180.},
            # use L'Hopital's rule to take the limit:
            # lim Pnm/sinth = k*dPnm | theta in {0., 180}
            # (evaluated at poles, where k=1 for theta=0 and k=-1 for
            # theta=180.); k = costh = P[1, 0] at poles for convenience
            if poles:
                div_Pnm = np.where(where_poles, Pnm[m, n+1], Pnm[n, m])
                div_Pnm[where_poles] *= Pnm[1, 0][where_poles]
                div_Pnm[~where_poles] /= sinth[~where_poles]
            else:
                div_Pnm = Pnm[n, m] / sinth

            A_phi[..., num] = m * div_Pnm * r_n * smp
            A_phi[..., num+1] = -m * div_Pnm * r_n * cmp

            num += 2

        # update radial dependence given the source
        if source == 'internal':
            r_n = r_n / radius
        else:
            r_n = r_n * radius

    return A_radius, A_theta, A_phi


def legendre_poly(nmax, theta):
    """
    Returns associated Legendre polynomials :math:`P_n^m(\\cos\\theta)`
    (Schmidt quasi-normalized) and the derivative
    :math:`dP_n^m(\\cos\\theta)/d\\theta` evaluated at :math:`\\theta`.

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
          dimensions. :math:`P_n^m(\\cos\\theta)` := ``Pnm[n, m, ...]`` and
          :math:`dP_n^m(\\cos\\theta)/d\\theta` := ``Pnm[m, n+1, ...]``

    References
    ----------
    Based on Equations 26-29 and Table 2 in:

    Langel, R. A., "Geomagnetism - The main field", Academic Press, 1987,
    chapter 4

    """

    costh = np.cos(np.radians(theta))
    sinth = np.sqrt(1-costh**2)

    Pnm = np.zeros((nmax+1, nmax+2) + costh.shape)
    Pnm[0, 0] = 1.  # is copied into trailing dimensions
    Pnm[1, 1] = sinth  # write theta into trailing dimenions via broadcasting

    rootn = np.sqrt(np.arange(2 * nmax**2 + 1))

    # Recursion relations after Langel "The Main Field" (1987),
    # eq. (27) and Table 2 (p. 256)
    for m in range(nmax):
        Pnm_tmp = rootn[m+m+1] * Pnm[m, m]
        Pnm[m+1, m] = costh * Pnm_tmp

        if m > 0:
            Pnm[m+1, m+1] = sinth*Pnm_tmp / rootn[m+m+2]

        for n in range(m+2, nmax+1):
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

        for m in range(2, n):
            Pnm[m, n+1] = (0.5*(np.sqrt((n + m) * (n - m + 1)) * Pnm[n, m-1]
                           - np.sqrt((n + m + 1) * (n - m)) * Pnm[n, m+1]))

        Pnm[n, n+1] = np.sqrt(2 * n) * Pnm[n, n-1] / 2

    return Pnm


def power_spectrum(coeffs, radius=None, *, nmax=None, source=None, axis=None):
    """
    Compute the spatial power spectrum.

    Parameters
    ----------
    coeffs : ndarray, shape (..., N*(N+2))
        Spherical harmonic coefficients for degree `N`.
    radius : float, optional
        Radius in kilometers (defaults to mean Earth's surface radius).
        It has no effect for ``source='toroidal'``.
    nmax : int, optional
        Maximum spherical degree (defaults to `N`).
    source : {'internal', 'external', 'toroidal'}
        Source of the field model (defaults to internal).
    axis : int, optional
        Axis of ``coeffs`` along which to compute the spatial power spectrum
        (defaults to -1, last dimension of ``coeffs``).

    Returns
    -------
    W_n : ndarray, shape (..., ``nmax``)
        Power spectrum for degrees up to degree ``nmax``

    Notes
    -----
    The spatial power spectrum for a potential field is defined as

    .. math::

        W_n(r) &= \\langle|\\mathbf{B}|^2\\rangle
             = \\frac{1}{4\\pi}\\iint_\\Omega |\\mathbf{B}|^2 \\mathrm{d}
               \\Omega \\\\
             &= W_n^\\mathrm{i}(r) + W_n^\\mathrm{e}(r) + W_n^\\mathrm{T}

    where the internal :math:`W_n^\\mathrm{i}`, external
    :math:`W_n^\\mathrm{e}` and the non-potential toroidal
    :math:`W_n^\\mathrm{T}` spatial power spectra are

    .. math::

        W_n^\\mathrm{i}(r) &= (n+1)\\left(\\frac{a}{r}\\right)^{2n+4}
                              \\sum_{m=0}^n [(g_n^m)^2 + (h_n^m)^2] \\\\
        W_n^\\mathrm{e}(r) &= n\\left(\\frac{r}{a}\\right)^{2n-2}\\sum_{m=0}^n
                              [(q_n^m)^2 + (s_n^m)^2] \\\\
        W_n^\\mathrm{T}    &= \\frac{n(n+1)}{2n+1}\\sum_{m=0}^n
                              [(T_n^{m,c})^2 + (T_n^{m,s})^2]

    References
    ----------
    Sabaka, T. J.; Hulot, G. & Olsen, N.,
    "Mathematical properties relevant to geomagnetic field modeling",
    Handbook of geomathematics, Springer, 2010, 503-538

    """

    axis = -1 if axis is None else int(axis)
    r = 1 if radius is None else basicConfig['params.r_surf'] / radius

    N = int(np.sqrt(coeffs.shape[axis] + 1) - 1)  # maximum degree

    if nmax is None:
        nmax = N
    elif nmax > N:
        print(f'Incompatible maximum degree nmax = {nmax}, '
              f'setting nmax to {N}.')
        nmax = N

    source = 'internal' if source is None else source

    if source == 'internal':
        def factor(n, r):
            return (n+1) * r**(2*n+4)
    elif source == 'external':
        def factor(n, r):
            return n * r**(-(2*n-2))
    elif source == 'toroidal':
        def factor(n, _):
            return n*(n+1) / (2*n+1)
    else:
        raise ValueError(
            'Unknown source. Use `internal`, `external` or `toroidal`.')

    shape = list(coeffs.shape)
    shape[axis] = nmax  # replace dim of axis with nmax
    W_n = np.empty(shape)

    for n in range(1, nmax+1):
        min = n**2 - 1
        max = min + (2*n + 1)

        slc1 = coeffs.ndim*[slice(None),]  # index for summation
        slc1[axis] = slice(min, max)

        slc2 = coeffs.ndim*[slice(None),]  # index for insertion into output
        slc2[axis] = n-1

        W_n[tuple(slc2)] = factor(n, r)*np.sum(coeffs[tuple(slc1)]**2,
                                               axis=axis)

    return W_n


def degree_correlation(coeffs_1, coeffs_2):
    """
    Correlation per spherical harmonic degree between two models 1 and 2.

    Parameters
    ----------
    coeffs_1, coeffs_2 : ndarray, shape (N,)
        Two sets of coefficients of equal length `N`.

    Returns
    -------
    C_n : ndarray, shape (nmax,)
        Degree correlation of the two models. There are `N = nmax(nmax+2)`
        coefficients.

    """

    if coeffs_1.ndim != 1:
        raise ValueError(f'Only 1-D input allowed {coeffs_1.ndim} != 1')

    if coeffs_2.ndim != 1:
        raise ValueError(f'Only 1-D input allowed {coeffs_2.ndim} != 1')

    if coeffs_1.size != coeffs_2.size:
        raise ValueError(
            'Number of coefficients is '
            'not equal ({0} != {1}).'.format(coeffs_1.size, coeffs_2.size))

    nmax = int(np.sqrt(coeffs_1.size + 1) - 1)

    C_n = np.zeros((nmax,))
    R_n = np.zeros((nmax,))  # elements are prop. to power spectrum of coeffs_1
    S_n = np.zeros((nmax,))  # elements are prop. to power spectrum of coeffs_2

    coeffs_12 = coeffs_1*coeffs_2

    for n in range(1, nmax+1):
        min = n**2 - 1
        max = min + (2*n + 1)
        R_n[n-1] = np.sum(coeffs_1[min:max]**2)
        S_n[n-1] = np.sum(coeffs_2[min:max]**2)
        C_n[n-1] = (np.sum(coeffs_12[min:max]) / np.sqrt(R_n[n-1]*S_n[n-1]))

    return C_n
