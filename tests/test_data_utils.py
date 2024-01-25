# Copyright (C) 2024 Clemens Kloss
#
# This file is part of ChaosMagPy.
#
# ChaosMagPy is released under the MIT license. See LICENSE in the root of the
# repository for full licensing details.

import numpy as np
import os
import textwrap
from unittest import TestCase, main
from datetime import datetime, timedelta
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

    def test_time_conversion(self):

        size = (10,)
        days = np.random.randint(365., size=size)
        hours = np.random.randint(24, size=size)
        minutes = np.random.randint(60, size=size)
        seconds = np.random.randint(60, size=size)

        for day, hour, minute, second in zip(days, hours, minutes, seconds):
            date = (timedelta(days=int(day)) +
                    datetime(1990, 1, 1, hour, minute, second))

            mjd = cpd.mjd2000(date.year, date.month, date.day, date.hour,
                              date.minute, date.second)

            dyear = cpd.mjd_to_dyear(mjd, leap_year=True)
            mjd2 = cpd.dyear_to_mjd(dyear, leap_year=True)
            self.assertIsNone(np.testing.assert_allclose(mjd2, mjd, atol=1e-8))

            dyear = cpd.mjd_to_dyear(mjd, leap_year=False)
            mjd2 = cpd.dyear_to_mjd(dyear, leap_year=False)
            self.assertIsNone(np.testing.assert_allclose(mjd2, mjd, atol=1e-8))

            dyear = cpd.mjd_to_dyear(mjd, leap_year=True)
            mjd2 = cpd.dyear_to_mjd(dyear, leap_year=False)
            self.assertRaises(
                AssertionError, lambda: np.testing.assert_allclose(
                    mjd2, mjd, atol=1e-8))

            dyear = cpd.mjd_to_dyear(mjd, leap_year=False)
            mjd2 = cpd.dyear_to_mjd(dyear, leap_year=True)
            self.assertRaises(
                AssertionError, lambda: np.testing.assert_allclose(
                    mjd2, mjd, atol=1e-8))

            dyear = cpd.mjd_to_dyear(mjd, leap_year=False)
            mjd2 = cpd.dyear_to_mjd(dyear, leap_year=None)
            self.assertRaises(
                AssertionError, lambda: np.testing.assert_allclose(
                    mjd2, mjd, atol=1e-8))

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

    def test_mjd_to_dyear(self):

        self.assertEqual(cpd.mjd_to_dyear(0.0), 2000.0)
        self.assertEqual(cpd.mjd_to_dyear(366.0), 2001.0)
        self.assertEqual(cpd.mjd_to_dyear(731.0), 2002.0)
        self.assertEqual(cpd.mjd_to_dyear(1096.0), 2003.0)
        self.assertEqual(cpd.mjd_to_dyear(4*365.25), 2004.0)

        self.assertEqual(cpd.mjd_to_dyear(0.0, leap_year=False), 2000.0)
        self.assertEqual(cpd.mjd_to_dyear(365.25, leap_year=False), 2001.0)
        self.assertEqual(cpd.mjd_to_dyear(2*365.25, leap_year=False), 2002.0)
        self.assertEqual(cpd.mjd_to_dyear(3*365.25, leap_year=False), 2003.0)
        self.assertEqual(cpd.mjd_to_dyear(4*365.25, leap_year=False), 2004.0)


if __name__ == '__main__':
    main()
