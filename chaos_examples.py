import numpy as np
import glob
import matplotlib.pyplot as plt
from chaosmagpy import load_CHAOS_matfile
from chaosmagpy.coordinate_utils import transform_points
from chaosmagpy.data_utils import mjd2000

FILEPATH_CHAOS = glob.glob('data/CHAOS-*.mat')[0]

R_REF = 6371.2


def main():

    example1()
    # example2()
    # example3()
    # example4()
    # example5()
    # example6()


def example1():

    # give inputs
    theta = np.array([55.676, 51.507, 64.133])  # colat in deg
    phi = np.array([12.568, 0.1275, -21.933])  # longitude in deg
    radius = np.array([0.0, 0.0, 500.0]) + R_REF  # radius from altitude in km
    time = np.array([3652.0, 5113.0, 5287.5])  # time in mjd2000

    # load the CHAOS model
    model = load_CHAOS_matfile(FILEPATH_CHAOS)
    print(model)

    print('Computing core field.')
    B_core = model.synth_values_tdep(time, radius, theta, phi)

    print('Computing crustal field up to degree 110.')
    B_crust = model.synth_values_static(radius, theta, phi, nmax=110)

    # complete internal contribution
    B_radius_int = B_core[0] + B_crust[0]
    B_theta_int = B_core[1] + B_crust[1]
    B_phi_int = B_core[2] + B_crust[2]

    print('Computing field due to external sources, incl. induced field: GSM.')
    B_gsm = model.synth_values_gsm(time, radius, theta, phi, source='all')

    print('Computing field due to external sources, incl. induced field: SM.')
    B_sm = model.synth_values_sm(time, radius, theta, phi, source='all')

    # complete external field contribution
    B_radius_ext = B_gsm[0] + B_sm[0]
    B_theta_ext = B_gsm[1] + B_sm[1]
    B_phi_ext = B_gsm[2] + B_sm[2]

    # complete forward computation
    B_radius = B_radius_int + B_radius_ext
    B_theta = B_theta_int + B_theta_ext
    B_phi = B_phi_int + B_phi_ext

    # save to output file
    data_CHAOS = np.stack([time, radius, theta, phi,
                           B_radius, B_theta, B_phi,
                           B_radius_int, B_theta_int, B_phi_int,
                           B_radius_ext, B_theta_ext, B_phi_ext], axis=-1)

    header = ('  t (mjd2000)    r (km) theta (deg)   phi (deg)       B_r   '
              'B_theta     B_phi       B_r   B_theta     B_phi       B_r   '
              'B_theta     B_phi\n'
              '                                                           '
              'model total                  model internal                '
              'model external')
    np.savetxt('example1_output.txt', data_CHAOS, delimiter=' ', header=header,
               fmt=['%15.8f', '%9.3f', '%11.5f', '%11.5f'] + 9*['%9.2f'])

    print('Saved output to example1_output.txt.')


def example2():
    """
    Plot difference between modelled and observed field strength using Swarm A
    data in August 2018 from a cdf-file.

    """
    import cdflib

    model = load_CHAOS_matfile(FILEPATH_CHAOS)
    print(model)

    cdf_file = cdflib.CDF('data/SW_OPER_MAGA_LR_1B_'
                          '20180801T000000_20180801T235959'
                          '_PT15S.cdf', 'r')
    # print(cdf_file.cdf_info())  # print cdf info/contents

    radius = cdf_file.varget('Radius') / 1000  # km
    theta = 90. - cdf_file.varget('Latitude')  # colat deg
    phi = cdf_file.varget('Longitude')  # deg
    time = cdf_file.varget('Timestamp')  # milli seconds since year 1
    time = time / (1e3*3600*24) - 730485  # time in modified Julian date 2000
    F_swarm = cdf_file.varget('F')
    cdf_file.close()

    theta_gsm, phi_gsm = transform_points(theta, phi,
                                          time=time, reference='gsm')
    index_day = np.logical_and(phi_gsm < 90, phi_gsm > -90)
    index_night = np.logical_not(index_day)

    # complete forward computation: pre-built not customizable (see ex. 1)
    B_radius, B_theta, B_phi = model(time, radius, theta, phi)

    # compute field strength and plot together with data
    F = np.sqrt(B_radius**2 + B_theta**2 + B_phi**2)

    print('RMSE of F: {:.5f} nT'.format(np.std(F-F_swarm)))

    plt.scatter(theta_gsm[index_day], F_swarm[index_day]-F[index_day],
                s=0.5, c='r', label='dayside')
    plt.scatter(theta_gsm[index_night], F_swarm[index_night]-F[index_night],
                s=0.5, c='b', label='nightside')
    plt.xlabel('dipole colatitude ($^\\circ$)')
    plt.ylabel('$\\mathrm{d} F$ (nT)')
    plt.legend(loc=2)
    plt.show()


def example3():
    """
    Plot maps of core field and its derivatives for different times and
    radii.

    """

    model = load_CHAOS_matfile(FILEPATH_CHAOS)

    radius = 0.53*R_REF  # radial distance in km of core-mantle boundary
    time = mjd2000(2015, 9, 1)  # year, month, day

    model.plot_maps_tdep(time, radius, nmax=16, deriv=1)


def example4():
    """
    Plot maps of static (i.e. small-scale crustal) magnetic field.

    """

    model = load_CHAOS_matfile(FILEPATH_CHAOS)

    radius = R_REF

    model.plot_maps_static(radius, nmax=85)


def example5():
    """
    Plot timeseries of the magnetic field at the ground observatory in MBour
    MBO (lat: 75.62°, east lon: 343.03°).

    """

    model = load_CHAOS_matfile(FILEPATH_CHAOS)

    # observatory location
    radius = 6.376832e+03
    theta = 75.69200
    phi = 343.05000

    model.plot_timeseries_tdep(radius, theta, phi, nmax=16, deriv=1)


def example6():
    """
    Plot maps of external and internal sources described in SM and GSM
    reference systems.

    """

    model = load_CHAOS_matfile(FILEPATH_CHAOS)

    radius = R_REF + 450
    time = mjd2000(2015, 9, 1, 12)

    model.plot_maps_external(time, radius, reference='all', source='all')


if __name__ == '__main__':

    main()
