import numpy as np
import os
from numpy import degrees
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
MATFILE_PATH = os.path.join(ROOT, 'CHAOS_test.mat')

# check if mat-file exists in tests directory
if os.path.isfile(MATFILE_PATH) is False:
    MATFILE_PATH = str(input('Matfile path for coordinate_utils test?: '))


class ModelUtilsTestCase(TestCase):
    def setUp(self):

        print(f'\nRunning {self._testMethodName}:')

    def test_degree_correlation(self):

        nmax = 4
        coeffs = np.random.random((int(nmax*(nmax+2)),))

        self.assertIsNone(np.testing.assert_equal(
            m.degree_correlation(coeffs, coeffs), np.ones((nmax,))))

    def test_power_spectrum(self):

        N = 3
        shape = (10, 2, int(N*(N+2)))
        coeffs = np.ones(shape)

        R_n = m.power_spectrum(coeffs)

        self.assertIsNone(np.testing.assert_equal(R_n.shape, (10, 2, N)))

    def test_synth_values_min(self):

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

        test(m.synth_values(coeffs_grid_min, radius, theta_grid, phi_grid,
                            nmin=1, nmax=4))
        test(m.synth_values(coeffs_grid_min, radius, theta_grid, phi_grid,
                            nmin=3, nmax=4))
        test(m.synth_values(coeffs_grid, radius, theta_grid, phi_grid,
                            nmin=3, nmax=4))

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
            self.assertIsNone(np.testing.assert_allclose(comp, comp_grid))

        for comp, comp_grid in zip(B_grid2, B):
            self.assertIsNone(np.testing.assert_allclose(comp, comp_grid))

    def test_design_matrix(self):
        """
        Test matrices for time-dependent field model using B-spline basis.
        Compare with output of identically named data code in ../data/.
        """

        n_data = int(300)
        t_start = 1997.1
        t_end = 2018.1
        n_breaks = int((t_end - t_start) / 0.5 + 1)
        time = np.linspace(t_start, t_end, num=n_data)
        radius = R_REF * np.ones(time.shape)
        theta = degrees(RAD * np.linspace(1, 179, num=n_data))
        phi = degrees(RAD * np.linspace(-180, 179, num=n_data))
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


if __name__ == '__main__':
    main()
