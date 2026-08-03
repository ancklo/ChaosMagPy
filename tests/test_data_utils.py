# Copyright (C) 2026 Clemens Kloss
#
# This file is part of ChaosMagPy.
#
# ChaosMagPy is released under the MIT license. See LICENSE in the root of the
# repository for full licensing details.

import os
import textwrap
from datetime import UTC, datetime, timedelta
from unittest import TestCase, main

import numpy as np

from chaosmagpy import data_utils as cpd

ROOT = os.path.abspath(os.path.dirname(__file__))
MATFILE_PATH = os.path.join(ROOT, 'data/CHAOS_test.mat')

# check if mat-file exists in tests directory
if os.path.isfile(MATFILE_PATH) is False:
    MATFILE_PATH = str(input('Matfile path for data_utils test?: '))


class DataUtils(TestCase):
    def setUp(self):

        print(textwrap.dedent(f"""\

            {"":-^70}
            Running {self._testMethodName}:
            """))

    def test_load_RC_datfile(self):

        # make sure link is not broken
        cpd.load_RC_datfile(filepath=None, parse_dates=None)

    def test_time_dyear_conversion(self):

        days = [120, 278, 231, 48, 112, 105, 225, 320, 227, 245]
        hours = [19, 11, 19, 22, 23,  0, 23, 2, 16, 13]
        minutes = [25, 32, 9, 11, 58, 58, 31, 28, 25, 17]
        seconds = [ 2, 35, 33, 16, 41, 3, 5, 33, 52, 27]

        test_func = lambda x, y: np.testing.assert_allclose(x, y, atol=1e-8)

        for day, hour, minute, second in zip(days, hours, minutes, seconds):
            date = (timedelta(days=int(day)) +
                    datetime(1990, 1, 1, hour, minute, second, tzinfo=UTC))

            mjd = cpd.mjd2000(date.year, date.month, date.day, date.hour,
                              date.minute, date.second)

            # test datetime conversion
            np.testing.assert_equal(mjd, cpd.mjd2000(date))


            dyear = cpd.mjd_to_dyear(mjd, leap_year=True)
            mjd2 = cpd.dyear_to_mjd(dyear, leap_year=True)
            np.testing.assert_allclose(mjd2, mjd, atol=1e-8)

            dyear = cpd.mjd_to_dyear(mjd, leap_year=False)
            mjd2 = cpd.dyear_to_mjd(dyear, leap_year=False)
            np.testing.assert_allclose(mjd2, mjd, atol=1e-8)

            dyear = cpd.mjd_to_dyear(mjd, leap_year=True)
            mjd2 = cpd.dyear_to_mjd(dyear, leap_year=False)
            self.assertRaises(AssertionError, test_func, mjd2, mjd)

            dyear = cpd.mjd_to_dyear(mjd, leap_year=False)
            mjd2 = cpd.dyear_to_mjd(dyear, leap_year=True)
            self.assertRaises(AssertionError, test_func, mjd2, mjd)

            dyear = cpd.mjd_to_dyear(mjd, leap_year=False)
            mjd2 = cpd.dyear_to_mjd(dyear, leap_year=None)
            self.assertRaises(AssertionError, test_func, mjd2, mjd)

    def test_mjd2000_broadcasting(self):
        """
        Ensure broadcasting works
        """

        actual = cpd.mjd2000(np.arange(2000, 2003), 2, 1)
        desired = np.array([  31.,  397.,  762.])
        np.testing.assert_allclose(actual, desired)

        actual = cpd.mjd2000(2000, 2, 1, np.arange(3))
        desired = 31. + np.arange(3)/24
        np.testing.assert_allclose(actual, desired)

    def test_dyear_to_mjd(self):

        self.assertEqual(cpd.dyear_to_mjd(2000.0), 0.0)
        self.assertEqual(cpd.dyear_to_mjd(2001.0), 366.0)
        self.assertEqual(cpd.dyear_to_mjd(2003.0), 1096.0)
        self.assertEqual(cpd.dyear_to_mjd(2004.0), 4*365.25)

        self.assertEqual(cpd.dyear_to_mjd(2000.0, leap_year=False), 0.0)
        self.assertEqual(cpd.dyear_to_mjd(2001.0, leap_year=False), 365.25)
        self.assertEqual(cpd.dyear_to_mjd(2003.0, leap_year=False), 3*365.25)
        self.assertEqual(cpd.dyear_to_mjd(2004.0, leap_year=False), 4*365.25)

    def test_mjd_to_dyear(self):

        self.assertEqual(cpd.mjd_to_dyear(0.0), 2000.0)
        self.assertEqual(cpd.mjd_to_dyear(366.0), 2001.0)
        self.assertEqual(cpd.mjd_to_dyear(1096.0), 2003.0)
        self.assertEqual(cpd.mjd_to_dyear(4*365.25), 2004.0)

        self.assertEqual(cpd.mjd_to_dyear(0.0, leap_year=False), 2000.0)
        self.assertEqual(cpd.mjd_to_dyear(365.25, leap_year=False), 2001.0)
        self.assertEqual(cpd.mjd_to_dyear(3*365.25, leap_year=False), 2003.0)
        self.assertEqual(cpd.mjd_to_dyear(4*365.25, leap_year=False), 2004.0)

    def test_mjd2000_integer(self):

        # day resolution
        np.testing.assert_equal(cpd.mjd2000(2500, 1, 1), 182622.0)
        np.testing.assert_equal(cpd.mjd2000(2050, 1, 1), 18263.0)
        np.testing.assert_equal(cpd.mjd2000(1950, 1, 1), -18262.0)
        np.testing.assert_equal(cpd.mjd2000(1500, 1, 1), -182621.0)
        np.testing.assert_equal(cpd.mjd2000(0, 1, 1), -730485.0)
        np.testing.assert_equal(cpd.mjd2000(-1, 12, 31), -730486.0)
        np.testing.assert_equal(cpd.mjd2000(-500, 1, 1), -913106.0)

        # microsecond resolution
        np.testing.assert_equal(cpd.mjd2000(2000, 1, 1, 6, 9, 0, 27),
                                0.2562500003125)
        np.testing.assert_equal(cpd.mjd2000(1999, 12, 31, 17, 59, 59, 999973),
                                -0.2500000003125)

        # nanosecond resolution
        np.testing.assert_equal(cpd.mjd2000(2000, 1, 1, 6, 9, 0, 27, 27),
                                0.2562500003128125)

    def test_mjd2000_datetime64(self):

        # day resolution
        timestamp = np.datetime64('2050-01-01')
        np.testing.assert_allclose(cpd.mjd2000(timestamp), 18263.0)

        timestamp = np.datetime64('1950-01-01')
        np.testing.assert_allclose(cpd.mjd2000(timestamp), -18262.0)

        # microsecond resolution
        timestamp = np.datetime64('2000-01-01T06:09:00.000027')
        np.testing.assert_allclose(cpd.mjd2000(timestamp), 0.2562500003125)

        # nanosecond resolution
        timestamp = np.datetime64('2000-01-01T06:09:00.000027027')
        np.testing.assert_allclose(cpd.mjd2000(timestamp), 0.2562500003128125)

    def test_mjd2000_datetime(self):

        # day resolution
        timestamp = datetime(2050, 1, 1, tzinfo=UTC)
        np.testing.assert_allclose(cpd.mjd2000(timestamp), 18263.0)

        timestamp = datetime(1950, 1, 1, tzinfo=UTC)
        np.testing.assert_allclose(cpd.mjd2000(timestamp), -18262.0)

        # microsecond resolution
        timestamp = datetime(2000, 1, 1, 6, 9, 0, 27, tzinfo=UTC)
        np.testing.assert_allclose(cpd.mjd2000(timestamp), 0.2562500003125)

    def test_timestamp(self):

        # day resolution
        timestamp = np.datetime64('2050-01-01')
        np.testing.assert_equal(cpd.timestamp(18263.0), timestamp)

        # hour resolution
        timestamp = np.datetime64('1999-12-31T18:00:00')
        np.testing.assert_equal(cpd.timestamp(-0.25), timestamp)

        timestamp = np.datetime64('1950-01-01T06:00:00')
        np.testing.assert_equal(cpd.timestamp(-18261.75), timestamp)

        # microsecond resolution
        timestamp = np.datetime64('2000-01-01T06:09:00.000027', 'us')
        np.testing.assert_equal(cpd.timestamp(0.2562500003125), timestamp)


if __name__ == '__main__':
    main()
