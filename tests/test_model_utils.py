# Copyright (C) 2023 Technical University of Denmark
#
# This file is part of ChaosMagPy.
#
# ChaosMagPy is released under the MIT license. See LICENSE in the root of the
# repository for full licensing details.

import numpy as np
import os
import textwrap
from unittest import TestCase, main
from chaosmagpy import model_utils as m
from timeit import default_timer as timer
from math import pi

try:
    from tests.helpers import load_matfile
except ImportError:
    from helpers import load_matfile


RAD = pi / 180
R_REF = 6371.2  # reference radius in kilometers

ROOT = os.path.abspath(os.path.dirname(__file__))
MATFILE_PATH = os.path.join(ROOT, 'data/CHAOS_test.mat')

# check if mat-file exists in tests directory
if os.path.isfile(MATFILE_PATH) is False:
    MATFILE_PATH = str(input('Matfile path for coordinate_utils test?: '))


class ModelUtils(TestCase):
    def setUp(self):

        print(textwrap.dedent(f"""\

            {"":-^70}
            Running {self._testMethodName}:
            """))

    def test_design_gauss(self):

        radius = 6371*np.ones((15,))
        theta = np.arange(15)
        phi = np.arange(15)

        # order equal to zero
        A = np.array(m.design_gauss(radius, theta, phi, nmax=3))
        A2 = np.array(m.design_gauss(radius, theta, phi, nmax=3, mmax=0))

        index = [0, 3, 8]
        np.testing.assert_equal(A[:, :, index], A2)

        # order equal or less than 1
        A = np.array(m.design_gauss(radius, theta, phi, nmax=3))
        A2 = np.array(m.design_gauss(radius, theta, phi, nmax=3, mmax=1))

        index = [0, 1, 2, 3, 4, 5, 8, 9, 10]
        np.testing.assert_equal(A[:, :, index], A2)

    def test_design_gauss_multidim(self):

        radius = 6371.
        theta = np.arange(30).reshape((3, 1, 10))
        phi = np.arange(30).reshape((3, 10, 1))
        coeffs = np.arange(8) + 1.

        Br, Bt, Bp = m.synth_values(coeffs, radius, theta, phi, nmax=2)
        Ar, At, Ap = m.design_gauss(radius, theta, phi, nmax=2)

        np.testing.assert_allclose(Br, Ar@coeffs)
        np.testing.assert_allclose(Bt, At@coeffs)
        np.testing.assert_allclose(Bp, Ap@coeffs)

    def test_degree_correlation(self):

        nmax = 4
        coeffs = np.random.random((int(nmax*(nmax+2)),))

        np.testing.assert_equal(
            m.degree_correlation(coeffs, coeffs), np.ones((nmax,)))

    def test_power_spectrum(self):

        shape = (15, 24, 3)  # nmax = (3, 4, 1)
        coeffs = np.ones(shape)

        R_n = m.power_spectrum(coeffs, source='internal')  # axis=-1
        R_n_desired = 6. * np.ones((15, 24, 1))
        np.testing.assert_equal(R_n, R_n_desired)

        R_n = m.power_spectrum(coeffs, source='internal', axis=0)
        R_n_desired = np.array([6., 15., 28.])[:, None, None] * np.ones(
            (3, 24, 3))
        np.testing.assert_equal(R_n, R_n_desired)

        R_n = m.power_spectrum(coeffs, source='external', axis=1)
        R_n_desired = np.array([3., 10., 21., 36])[None, :, None] * np.ones(
            (15, 4, 3))
        np.testing.assert_equal(R_n, R_n_desired)

        R_n = m.power_spectrum(coeffs, source='toroidal')
        R_n_desired = 2. * np.ones((15, 24, 1))
        np.testing.assert_equal(R_n, R_n_desired)

    def test_synth_values_mmax(self):

        theta = 79.
        phi = 13.
        radius = R_REF

        # function for quick testing with "true" solution
        np.testing.assert_allclose(
            m.synth_values([1., 2., 3., 4., 5., 6., 7., 8.],
                           radius, theta, phi),
            m.synth_values([1., 2., 3., 4., 5., 6., 7., 8.],
                           radius, theta, phi, nmax=None, mmax=None))

        np.testing.assert_allclose(
            m.synth_values([1., 2., 3., 0., 0., 0., 0., 0.],
                           radius, theta, phi),
            m.synth_values([1., 2., 3., 4., 5., 6., 7., 8.],
                           radius, theta, phi, nmax=1, mmax=None))

        np.testing.assert_allclose(
            m.synth_values([1., 2., 3., 4., 5., 6., 0., 0.],
                           radius, theta, phi),
            m.synth_values([1., 2., 3., 4., 5., 6.],
                           radius, theta, phi, nmax=2, mmax=1))

        np.testing.assert_allclose(
            m.synth_values([1., 0., 0., 0., 0., 0., 0., 0.],
                           radius, theta, phi),
            m.synth_values([1.],
                           radius, theta, phi, nmax=1, mmax=0))

        np.testing.assert_allclose(
            m.synth_values([1., 2., 3., 4., 5., 6., 0., 0.],
                           radius, theta, phi),
            m.synth_values([1., 2., 3., 4., 5., 6.],
                           radius, theta, phi, nmax=None, mmax=1))

    def test_synth_values_nmin(self):

        theta = 79.
        phi = 13.
        radius = R_REF

        theta_ls = theta*np.ones((15,))
        phi_ls = phi*np.ones((27,))
        phi_grid, theta_grid = np.meshgrid(phi_ls, theta_ls)  # 2-D
        radius_grid = R_REF * np.ones(phi_grid.shape)  # 2-D
        coeffs_grid = np.random.random()*np.ones(phi_grid.shape + (24,))  # 3-D

        coeffs_grid_min = np.copy(coeffs_grid)
        coeffs_grid_min[..., :8] = 0.  # nmax=4

        # exhaustive inputs
        B = m.synth_values(coeffs_grid_min, radius_grid, theta_grid, phi_grid)

        # function for quick testing with "true" solution
        def test(field):
            self.assertIsNone(np.testing.assert_allclose(B, field))

        test(m.synth_values(coeffs_grid[..., 8:], radius, theta_grid, phi_grid,
                            nmin=3, nmax=4))
        test(m.synth_values(coeffs_grid_min[..., 8:], radius, theta_grid,
                            phi_grid, nmin=3, nmax=4))

    def test_synth_values_inputs(self):

        theta = 79.
        phi = 13.
        radius = R_REF

        theta_ls = theta*np.ones((15,))
        phi_ls = phi*np.ones((27,))
        phi_grid, theta_grid = np.meshgrid(phi_ls, theta_ls)  # 2-D
        radius_grid = R_REF * np.ones(phi_grid.shape)  # 2-D
        coeffs_grid = np.random.random()*np.ones(phi_grid.shape + (24,))  # 3-D

        # exhaustive inputs
        B = m.synth_values(coeffs_grid, radius_grid, theta_grid, phi_grid)

        # function for quick testing with "true" solution
        def test(field):
            self.assertIsNone(np.testing.assert_allclose(B, field))

        # one input () dimension: float
        test(m.synth_values(coeffs_grid, radius, theta_grid, phi_grid))
        test(m.synth_values(coeffs_grid, radius_grid, theta, phi_grid))
        test(m.synth_values(coeffs_grid, radius_grid, theta_grid, phi))

        # one input (1,) dimension: one element array
        test(m.synth_values(
            coeffs_grid[0, 0], radius_grid, theta_grid, phi_grid))
        test(m.synth_values(
            coeffs_grid, radius_grid[0, 0], theta_grid, phi_grid))
        test(m.synth_values(
            coeffs_grid, radius_grid, theta_grid[0, 0], phi_grid))
        test(m.synth_values(
            coeffs_grid, radius_grid, theta_grid, phi_grid[0, 0]))

        # GRID MODE
        test(m.synth_values(
            coeffs_grid, radius_grid, theta_ls, phi_ls, grid=True))
        test(m.synth_values(coeffs_grid, radius, theta_ls, phi_ls, grid=True))
        test(m.synth_values(
            coeffs_grid[0, 0], radius_grid, theta_ls, phi_ls, grid=True))
        test(m.synth_values(
            coeffs_grid[0, 0], radius_grid[0, 0], theta_ls, phi_ls, grid=True))

        # test multi-dimensional grid

        # full grid
        coeffs_grid = np.random.random((24,))
        radius_grid = radius * np.ones((2, 5, 6, 4))
        phi_grid = phi * np.ones((2, 5, 6, 4))
        theta_grid = theta * np.ones((2, 5, 6, 4))

        B = m.synth_values(coeffs_grid, radius_grid, theta_grid, phi_grid)

        # reduced grid
        radius_grid2 = radius * np.ones((2, 1, 6, 4))
        phi_grid2 = phi * np.ones((2, 5, 1, 1))
        theta_grid2 = theta * np.ones((1, 5, 6, 4))

        test(m.synth_values(coeffs_grid, radius_grid2, theta_grid2, phi_grid2))

    def test_synth_values_grid(self):

        radius = R_REF
        theta = np.linspace(0., 180., num=4000)
        phi = np.linspace(-180., 180., num=3000)
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        coeffs = np.random.random((24,))

        s = timer()
        B_grid = m.synth_values(
            coeffs, radius, theta_grid, phi_grid, grid=False)
        e = timer()
        print("  Time for 'grid=False' computation: ", e - s)

        s = timer()
        B_grid2 = m.synth_values(
            coeffs, radius, theta[..., None], phi[None, ...], grid=False)
        e = timer()
        print("  Time for broadcasted 'grid=False' computation: ", e - s)

        s = timer()
        B = m.synth_values(
            coeffs, radius, theta, phi, grid=True)
        e = timer()
        print("  Time for 'grid=True' computation: ", e - s)

        for comp, comp_grid in zip(B_grid, B_grid2):
            np.testing.assert_allclose(comp, comp_grid)

        for comp, comp_grid in zip(B_grid2, B):
            np.testing.assert_allclose(comp, comp_grid)

    def test_synth_values_poles(self):

        radius = 6371.2
        theta = np.array([0., 180.])
        phi = np.linspace(0., 360., num=10, endpoint=False)

        # dipole aligned with x-axis, B = -grad(V)
        # North pole: Bx = -1, Bt = -1
        # South pole: Bx = -1, Bt = 1
        Br, Bt, Bp = m.synth_values([0., 1., 0], radius, theta, phi, grid=True)

        # north pole (theta, phi) = (0., 0.)
        npole = (0, 0)
        np.testing.assert_allclose(Br[npole], 0.)
        np.testing.assert_allclose(Bt[npole], -1.)
        np.testing.assert_allclose(Bp[npole], 0.)

        Bx = (np.cos(np.radians(0.))*np.cos(np.radians(phi))*Bt[0, :]
              - np.sin(np.radians(phi))*Bp[0, :])
        np.testing.assert_allclose(Bx, -np.ones((10,)))

        # south pole (theta, phi) = (180., 0.)
        spole = (1, 0)
        np.testing.assert_allclose(Br[spole], 0.)
        np.testing.assert_allclose(Bt[spole], 1.)
        np.testing.assert_allclose(Bp[spole], 0.)

        Bx = (np.cos(np.radians(180.))*np.cos(np.radians(phi))*Bt[1, :]
              - np.sin(np.radians(phi))*Bp[1, :])
        np.testing.assert_allclose(Bx, -np.ones((10,)))

    def test_design_matrix(self):
        """
        Test matrices for time-dependent field model using B-spline basis.

        """

        n_data = int(300)
        t_start = 1997.1
        t_end = 2018.1
        n_breaks = int((t_end - t_start) / 0.5 + 1)
        time = np.linspace(t_start, t_end, num=n_data)
        radius = R_REF * np.ones(time.shape)
        theta = np.linspace(1, 179, num=n_data)
        phi = np.linspace(-180, 179, num=n_data)
        n_static = int(80)
        n_tdep = int(20)
        order = int(6)  # order of spline basis functions (4 = cubic)

        # create a knot vector without endpoint repeats and # add endpoint
        # repeats as appropriate for spline degree p
        knots = np.linspace(t_start, t_end, num=n_breaks)
        knots = m.augment_breaks(knots, order)

        s = timer()
        G_radius, G_theta, G_phi = m.design_matrix(knots, order, n_tdep, time,
                                                   radius, theta, phi,
                                                   n_static=n_static)
        e = timer()

        # load matfile
        test = load_matfile(MATFILE_PATH, 'test_design_matrix')

        G_radius_mat = test['G_radius']
        G_theta_mat = test['G_theta']
        G_phi_mat = test['G_phi']
        runtime = test['runtime']

        print("  Time for design_matrix computation (Python): ", e - s)
        print("  Time for design_matrix computation (Matlab):  {:}".format(
            runtime[0, 0]))

        self.assertIsNone(
            np.testing.assert_allclose(G_radius, G_radius_mat, atol=1e-5))
        self.assertIsNone(
            np.testing.assert_allclose(G_theta, G_theta_mat, atol=1e-5))
        self.assertIsNone(
            np.testing.assert_allclose(G_phi, G_phi_mat, atol=1e-5))

    def test_colloc_matrix(self):

        breaks = np.linspace(0., 300., 20)
        order = 6
        knots = m.augment_breaks(breaks, order)

        x = np.linspace(0., 300., 1000)

        test = load_matfile(MATFILE_PATH, 'test_colloc_matrix')

        colloc_m = test['colloc']

        for deriv in range(order):
            print(f"Checking deriv = {deriv} of collocation matrix.")

            colloc = m.colloc_matrix(x, knots, order, deriv=deriv)

            self.assertIsNone(np.testing.assert_allclose(
                colloc, colloc_m[deriv::order], atol=1e-5))


if __name__ == '__main__':
    main()
