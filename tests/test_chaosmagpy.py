import os
import numpy as np
import textwrap
from chaosmagpy import load_CHAOS_matfile, load_CHAOS_shcfile
from chaosmagpy import coordinate_utils as c
from chaosmagpy import model_utils as m
from chaosmagpy import data_utils as du
from unittest import TestCase, main, skip
try:
    from tests.helpers import load_matfile
except ImportError:
    from helpers import load_matfile

R_REF = 6371.2  # reference radius in km
ROOT = os.path.abspath(os.path.dirname(__file__))
MATFILE_PATH = os.path.join(ROOT, 'CHAOS_test.mat')
CHAOS_PATH = os.path.join(ROOT, 'CHAOS-6-x7.mat')

# check if mat-file exists in tests directory
if os.path.isfile(MATFILE_PATH) is False:
    MATFILE_PATH = str(input('Matfile path for chaosmagpy test?: '))


class ChaosMagPy(TestCase):
    def setUp(self):

        print(textwrap.dedent(f"""\

            {"":-^70}
            Running {self._testMethodName}:
            """))

    def test_synth_euler_angles(self):

        model = load_CHAOS_matfile(CHAOS_PATH)

        test = load_matfile(MATFILE_PATH, 'test_synth_euler_angles')
        time = np.squeeze(test['time'])

        swarm_c = model.synth_euler_angles(time, 'swarm_c')

        self.assertIsNone(np.testing.assert_allclose(
            swarm_c, test['swarm_c']))

    def test_save_matfile(self):

        seq = np.random.randint(0, 10, size=(5,))
        filename = 'CHAOS-tmp_' + ''.join([str(a) for a in seq]) + '.mat'
        filepath = os.path.join(ROOT, filename)

        model = load_CHAOS_matfile(CHAOS_PATH)
        print('  ', end='')  # indent line by two withespaces
        model.save_matfile(filepath)

        def test(x, y):
            if isinstance(x, str) or isinstance(y, str):
                # convert unicode to str
                np.testing.assert_string_equal(str(x), str(y))
            else:
                np.testing.assert_allclose(x, y, atol=1e-10)

        chaos = du.load_matfile(CHAOS_PATH, variable_names=[
            'pp', 'model_ext', 'model_Euler', 'g'])
        chaos_out = du.load_matfile(filepath, variable_names=[
            'pp', 'model_ext', 'model_Euler', 'g'])

        pp = chaos['pp']
        pp_out = chaos_out['pp']

        for key in ['order', 'dim', 'pieces', 'form', 'coefs', 'breaks']:
            test(pp[key], pp_out[key])

        test(chaos['g'], chaos_out['g'])

        model_ext = chaos['model_ext']
        model_ext_out = chaos_out['model_ext']

        for key in ['m_Dst', 'm_gsm', 'm_sm', 'q10', 'qs11', 't_break_q10',
                    't_break_qs11']:
            test(model_ext[key], model_ext_out[key])

        model_Euler = chaos['model_Euler']
        model_Euler_out = chaos_out['model_Euler']

        for key in ['alpha', 'beta', 'gamma', 't_break_Euler']:
            var = model_Euler[key]
            var_out = model_Euler_out[key]
            test(var.shape, var_out.shape)

            for value, value_out in zip(np.ravel(var), np.ravel(var_out)):
                test(value, value_out)

        print(f"  Removing file {filepath}")
        os.remove(filepath)

    def test_save_shcfile(self):

        seq = np.random.randint(0, 10, size=(5,))
        filename = 'CHAOS-tmp_' + ''.join([str(a) for a in seq]) + '.shc'
        filepath = os.path.join(ROOT, filename)

        model_mat = load_CHAOS_matfile(CHAOS_PATH)

        print('  On time-dependent part:')
        print('  ', end='')
        model_mat.save_shcfile(filepath, model='tdep')
        coeffs_tdep_mat = model_mat.model_tdep.coeffs

        model_shc = load_CHAOS_shcfile(filepath)
        coeffs_tdep_shc = model_shc.model_tdep.coeffs

        print('  Max Error =',
              np.amax(np.abs(coeffs_tdep_shc - coeffs_tdep_mat)))

        np.testing.assert_allclose(
            coeffs_tdep_shc, coeffs_tdep_mat, rtol=1e-2, atol=1e-3)

        print('  On static part:')
        print('  ', end='')
        model_mat.save_shcfile(filepath, model='static')
        coeffs_static_mat = model_mat.model_static.coeffs

        model_shc = load_CHAOS_shcfile(filepath)
        coeffs_static_shc = model_shc.model_static.coeffs

        print('  Max Error =',
              np.amax(np.abs(coeffs_static_shc - coeffs_static_mat)))

        np.testing.assert_allclose(
            coeffs_static_shc, coeffs_static_mat, rtol=1e-2, atol=1e-3)

        print(f"  Removing file {filepath}")
        os.remove(filepath)

    def test_complete_forward(self):

        n_data = int(300)
        t_start = -200.0
        t_end = 6000.0
        time = np.linspace(t_start, t_end, num=n_data)
        radius = R_REF * np.ones(time.shape)
        theta = np.linspace(1, 179, num=n_data)
        phi = np.linspace(-180, 179, num=n_data)

        model = load_CHAOS_matfile(CHAOS_PATH)

        B_radius, B_theta, B_phi = model(time, radius, theta, phi)

        # load matfile
        test = load_matfile(MATFILE_PATH, 'test_complete_forward')

        B_radius_mat = np.ravel(test['B_radius'])
        B_theta_mat = np.ravel(test['B_theta'])
        B_phi_mat = np.ravel(test['B_phi'])

        for component in ['B_radius', 'B_theta', 'B_phi']:
            res = np.abs(eval(component) - eval('_'.join((component, 'mat'))))
            print('  -------------------')
            print(f'  {component}:')
            print('  MAE =', np.mean(res), 'nT')
            print('  RMSE =', np.sqrt(np.mean(res**2)), 'nT')
            print('  Max Error =', np.amax(res), 'nT')
            print('  Min Error =', np.amin(res), 'nT')

        np.testing.assert_allclose(
            B_radius, B_radius_mat, rtol=1e-7, atol=1e-2)
        np.testing.assert_allclose(
            B_theta, B_theta_mat, rtol=1e-7, atol=1e-2)
        np.testing.assert_allclose(
            B_phi, B_phi_mat, rtol=1e-7, atol=1e-2)

    def test_surface_field(self):

        model = load_CHAOS_matfile(CHAOS_PATH)

        time = 2015.
        radius = R_REF
        theta = np.linspace(1, 179, num=180)
        phi = np.linspace(-180, 179, num=360)

        B_radius, B_theta, B_phi = model.synth_values_tdep(
            time, radius, theta, phi, nmax=13, grid=True, deriv=0)

        B_radius = np.ravel(B_radius, order='F')  # ravel column-major
        B_theta = np.ravel(B_theta, order='F')
        B_phi = np.ravel(B_phi, order='F')

        # load matfile
        test = load_matfile(MATFILE_PATH, 'test_surface_field')

        B_radius_mat = np.ravel(test['B_radius'])
        B_theta_mat = np.ravel(test['B_theta'])
        B_phi_mat = np.ravel(test['B_phi'])

        for component in ['B_radius', 'B_theta', 'B_phi']:
            res = np.abs(eval(component) - eval('_'.join((component, 'mat'))))
            print('  -------------------')
            print(f'  {component}:')
            print('  MAE =', np.mean(res), 'nT')
            print('  RMSE =', np.sqrt(np.mean(res**2)), 'nT')
            print('  Max Error =', np.amax(res), 'nT')
            print('  Min Error =', np.amin(res), 'nT')

        self.assertIsNone(np.testing.assert_allclose(
            B_radius, B_radius_mat))
        self.assertIsNone(np.testing.assert_allclose(
            B_theta, B_theta_mat))
        self.assertIsNone(np.testing.assert_allclose(
            B_phi, B_phi_mat))

    def test_sv_timeseries(self):

        # some observatory location
        radius = R_REF
        theta = 90-14.308
        phi = -16.950

        model = load_CHAOS_matfile(CHAOS_PATH)

        time = np.linspace(model.model_tdep.breaks[0],
                           model.model_tdep.breaks[-1], num=1000)

        B_radius, B_theta, B_phi = model.synth_values_tdep(
            time, radius, theta, phi, nmax=16, deriv=1)

        # load matfile
        test = load_matfile(MATFILE_PATH, 'test_sv_timeseries')

        B_radius_mat = np.ravel(test['B_radius'])
        B_theta_mat = np.ravel(test['B_theta'])
        B_phi_mat = np.ravel(test['B_phi'])

        for component in ['B_radius', 'B_theta', 'B_phi']:
            res = np.abs(eval(component) - eval('_'.join((component, 'mat'))))
            print('  -------------------')
            print(f'  {component}:')
            print('  MAE =', np.mean(res), 'nT')
            print('  RMSE =', np.sqrt(np.mean(res**2)), 'nT')
            print('  Max Error =', np.amax(res), 'nT')
            print('  Min Error =', np.amin(res), 'nT')

        self.assertIsNone(np.testing.assert_allclose(
            B_radius, B_radius_mat))
        self.assertIsNone(np.testing.assert_allclose(
            B_theta, B_theta_mat))
        self.assertIsNone(np.testing.assert_allclose(
            B_phi, B_phi_mat))

    def test_synth_sm_field(self):

        # load matfile
        test = load_matfile(MATFILE_PATH, 'test_synth_sm_field')

        model = load_CHAOS_matfile(CHAOS_PATH)

        N = int(1000)

        time = np.linspace(-200, 6000, num=N)
        radius = R_REF
        theta = np.linspace(1, 179, num=N)
        phi = np.linspace(-180, 179, num=N)

        B_radius = np.zeros(time.shape)
        B_theta = np.zeros(time.shape)
        B_phi = np.zeros(time.shape)

        for source in ['internal', 'external']:
            B_radius_new, B_theta_new, B_phi_new = model.synth_values_sm(
                time, radius, theta, phi, source=source)

            B_radius += B_radius_new
            B_theta += B_theta_new
            B_phi += B_phi_new

        B_radius_mat = np.ravel(test['B_radius'])
        B_theta_mat = np.ravel(test['B_theta'])
        B_phi_mat = np.ravel(test['B_phi'])

        for component in ['B_radius', 'B_theta', 'B_phi']:
            res = np.abs(eval(component) - eval('_'.join((component, 'mat'))))
            print('  -------------------')
            print(f'  {component}:')
            print('  MAE =', np.mean(res), 'nT')
            print('  RMSE =', np.sqrt(np.mean(res**2)), 'nT')
            print('  Max Error =', np.amax(res), 'nT')
            print('  Min Error =', np.amin(res), 'nT')

        self.assertIsNone(np.testing.assert_allclose(
            B_radius, B_radius_mat, rtol=1e-2, atol=1e-2))
        self.assertIsNone(np.testing.assert_allclose(
            B_theta, B_theta_mat, rtol=1e-2, atol=1e-2))
        self.assertIsNone(np.testing.assert_allclose(
            B_phi, B_phi_mat, rtol=1e-2, atol=1e-2))

    def test_synth_gsm_field(self):

        # load matfile
        test = load_matfile(MATFILE_PATH, 'test_synth_gsm_field')

        model = load_CHAOS_matfile(CHAOS_PATH)

        N = int(1000)

        time = np.linspace(-1000, 6000, num=N)
        radius = R_REF
        theta = np.linspace(1, 179, num=N)
        phi = np.linspace(-180, 179, num=N)

        B_radius = np.zeros(time.shape)
        B_theta = np.zeros(time.shape)
        B_phi = np.zeros(time.shape)

        for source in ['internal', 'external']:
            B_radius_new, B_theta_new, B_phi_new = model.synth_values_gsm(
                time, radius, theta, phi, source=source)

            B_radius += B_radius_new
            B_theta += B_theta_new
            B_phi += B_phi_new

        B_radius_mat = np.ravel(test['B_radius'])
        B_theta_mat = np.ravel(test['B_theta'])
        B_phi_mat = np.ravel(test['B_phi'])

        for component in ['B_radius', 'B_theta', 'B_phi']:
            res = np.abs(eval(component) - eval('_'.join((component, 'mat'))))
            print('  -------------------')
            print(f'  {component}:')
            print('  MAE =', np.mean(res), 'nT')
            print('  RMSE =', np.sqrt(np.mean(res**2)), 'nT')
            print('  Max Error =', np.amax(res), 'nT')
            print('  Min Error =', np.amin(res), 'nT')

        np.testing.assert_allclose(
            B_radius, B_radius_mat, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(
            B_theta, B_theta_mat, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(
            B_phi, B_phi_mat, rtol=1e-2, atol=1e-2)

    def test_rotate_gsm_vector(self):

        time = 20
        nmax = 1
        kmax = 1
        radius = R_REF
        theta_geo = 90/6
        phi_geo = 90/8
        g_gsm = np.array([1, 2, -1])

        base_1, base_2, base_3 = c.basevectors_gsm(time)
        matrix = c.rotate_gauss(nmax, kmax, base_1, base_2, base_3)

        g_geo = np.matmul(matrix, g_gsm)

        B_geo_1 = m.synth_values(g_geo, radius, theta_geo, phi_geo)
        B_geo_1 = np.array(B_geo_1)

        theta_gsm, phi_gsm, R = c.matrix_geo_to_base(theta_geo, phi_geo,
                                                     base_1, base_2, base_3)

        B_gsm = m.synth_values(g_gsm, radius, theta_gsm, phi_gsm)
        B_gsm = np.array(B_gsm)

        B_geo_2 = np.matmul(R.transpose(), B_gsm)

        np.testing.assert_allclose(B_geo_1, B_geo_2, rtol=1e-5)


def profiler_complete_forward(n_data=300):
    """
    Example:

    .. code-block:: python

        %load_ext line_profiler

        import sys
        sys.path.insert(0, <tests directory path>)
        from test_chaosmagpy import profiler_complete_forward as profiler
        import chaosmagpy as cp

        %lprun -f cp.coordinate_utils.synth_rotate_gauss profiler(n_data=300)

    """

    t_start = -200.0
    t_end = 6000.0
    time = np.linspace(t_start, t_end, num=n_data)
    radius = R_REF * np.ones(time.shape)
    theta = np.linspace(1, 179, num=n_data)
    phi = np.linspace(-180, 179, num=n_data)

    model = load_CHAOS_matfile(CHAOS_PATH)

    # B_radius, B_theta, B_phi = model(time, radius, theta, phi)
    B_radius, B_theta, B_phi = model(time, radius, theta, phi)
    print('Ran "profiler_complete_forward"')


if __name__ == '__main__':
    main()
