"""
Parameters and options in ChaosMagPy are stored in a dictionary and can be
modified as desired. The following list gives an overview of the possible
keywords.

**Parameters**

 ====================  =============  =========================================
 Value                 Type           Description
 ====================  =============  =========================================
 'params.r_surf'       `float`        Reference radius in kilometers (defaults
                                      to Earth's surface radius 6371.2 km).
 'params.r_cmb'        `float`        Core-mantle boundary radius in kilometers
                                      (defaults to 3485.0 km).
 'params.dipole'       `list`,        Coefficients of the dipole (used for
                       `ndarray`,     GSM/SM coordinate transformations).
                       `shape (3,)`
 'params.version'      `str`          Default version of the CHAOS model, e.g.
                                      ``'6.x7'``.
 'params.cdf_to_mjd'   `int`          Number of days on Jan 01, 2000 since Jan
                                      01, 0000 (CDF start epoch)
 ====================  =============  =========================================

**Files**

 ==========================  ===========  =====================================
 Value                       Type         Description
 ==========================  ===========  =====================================
 'files.RC_index'            `h5-file`,   RC-index file (used for external
                             `txt-file`   field computation). See also
                                          :func:`data_utils.save_RC_h5file`.
 'files.GSM_spectrum'        `npz-file`   GSM transformation coefficients. See
                                          also :func:`coordinate_utils.\\
                                          rotate_gauss_fft`.
 'files.SM_spectrum'         `npz-file`   SM transformation coefficients. See
                                          also :func:`coordinate_utils.\\
                                          rotate_gauss_fft`.
 'files.Earth_conductivity'  `txt-file`   Conductivity model of a layered Earth
                                          (used for induced fields).
 ==========================  ===========  =====================================

**Plots**

 ==========================  ===========  =====================================
 Value                       Type         Description
 ==========================  ===========  =====================================
 'plots.figure_width'        `float`      Plot width in inches (defaults to 6.3
                                          or equiv. 16cm)
 ==========================  ===========  =====================================

"""

import os
import re
import numpy as np
from contextlib import contextmanager

ROOT = os.path.abspath(os.path.dirname(__file__))
LIB = os.path.join(ROOT, 'lib')


# copied/inspired by matplotlib.rcsetup
def check_path_exists(s):
    """Check that path to file exists."""
    if s is None or s == 'None':
        return None
    if os.path.exists(s):
        return s
    else:
        raise FileNotFoundError(f'{s} does not exist.')


def check_float(s):
    """Convert to float."""
    try:
        return float(s)
    except ValueError:
        raise ValueError(f'Could not convert {s} to float.')


def check_int(s):
    """Convert to integer."""
    try:
        return int(s)
    except ValueError:
        raise ValueError(f'Could not convert {s} to integer.')


def check_string(s):
    """Convert to string."""
    try:
        return str(s)
    except ValueError:
        raise ValueError(f'Could not convert {s} to string.')


def check_vector(s, len=None):
    """Check that input is vector with required length."""
    try:
        s = np.array(s)
        assert s.ndim == 1
        if len is not None:
            if s.size != len:
                raise ValueError(f'Wrong length: {s.size} != {len}.')
        return s
    except Exception as err:
        raise ValueError(f'Not a valid vector. {err}')


def check_version_string(s):
    """Check correct format of version string."""

    s = check_string(s)

    match = re.search(r'\d+\.x\d+', s)
    if match:
        return s
    else:
        raise ValueError(f'Not supported version format "{s}"')


DEFAULTS = {
    'params.r_surf': [6371.2, check_float],
    'params.r_cmb': [3485.0, check_float],
    'params.dipole': [np.array([-29442.0, -1501.0, 4797.1]),
                      lambda x: check_vector(x, len=3)],
    'params.version': ['6.x7', check_version_string],
    'params.cdf_to_mjd': [730485, check_int],

    # location of coefficient files
    'file.RC_index': [os.path.join(LIB, 'RC_index.h5'),
                      check_path_exists],
    'file.GSM_spectrum': [os.path.join(LIB, 'frequency_spectrum_gsm.npz'),
                          check_path_exists],
    'file.SM_spectrum': [os.path.join(LIB, 'frequency_spectrum_sm.npz'),
                         check_path_exists],
    'file.Earth_conductivity': [os.path.join(LIB, 'Earth_conductivity.dat'),
                                check_path_exists],

    # plot related configuration
    'plots.figure_width': [6.3, check_float],
}


class BasicConfig(dict):
    """Class for creating CHAOS configuration dictionary."""

    defaults = DEFAULTS

    def __init__(self, *args, **kwargs):
        super().update(*args, **kwargs)

    def __setitem__(self, key, value):
        """Set and check value before updating dictionary."""

        try:
            try:
                cval = self.defaults[key][1](value)
            except ValueError as err:
                raise ValueError(f'Key "{key}": {err}')
            super().__setitem__(key, cval)
        except KeyError:
            raise KeyError(f'"{key}" is not a valid parameter.')

    def __str__(self):
        return '\n'.join(map('{0[0]}: {0[1]}'.format, sorted(self.items())))

    def reset(self, key):
        """
        Load default values.

        Parameters
        ----------
        key : str
            Single keyword that is reset to the default.

        """
        self.__setitem__(key, self.defaults[key][0])

    def fullreset(self):
        """
        Load all default values.

        """
        super().update({key: val for key, (val, _) in self.defaults.items()})

    def load(self, filepath):
        """
        Load configuration dictionary from file.

        Parameters
        ----------
        filepath : str
            Filepath and name to configuration textfile.

        """

        with open(filepath, 'r') as f:
            for line in f.readlines():
                # skip comment and empty lines
                if not line.strip():
                    continue
                elif line.strip()[0] == '#':
                    continue

                key, value = line.split(':')
                value = value.split('#')[0].strip()  # remove comments and \n

                # check list input and convert to array
                if value[0] == '[' and value[-1] == ']':
                    value = np.fromstring(value[1:-1], sep=' ')

                self.__setitem__(key.strip(), value)

    def save(self, filepath):
        """
        Save configuration dictionary to a file.

        Parameters
        ----------
        filepath : str
            Filepath and name of the textfile that will be saved with the
            configuration values.

        """

        with open(filepath, 'w') as f:
            for key, value in self.items():
                f.write(f'{key} : {value}\n')
        f.close()
        print(f'Saved configuration textfile to {filepath}.')

    @contextmanager
    def context(self, key, value):
        """
        Use context manager to temporarily change setting.

        Parameters
        ----------
        key : str
            BasicConfig configuration key.
        value
            Value compatible with ``key``.

        Examples
        --------
        Temporarily change the radius of Earth's surface for a computation
        and then change it back to the original value.

        .. code-block:: python

          from chaosmagpy import basicConfig

          print('Before: ', basicConfig['params.r_surf'])

          # change Earth's radius to 10 km
          with basicConfig.context('params.r_surf', 10):
              # do something at r_surf = 10 km ...
              print('Inside: ', basicConfig['params.r_surf'])

          print('After: ', basicConfig['params.r_surf'])

        """
        old_value = self.__getitem__(key)
        self.__setitem__(key, value)
        yield
        self.__setitem__(key, old_value)


# load defaults
basicConfig = BasicConfig({key: val for key, (val, _) in DEFAULTS.items()})


if __name__ == '__main__':
    # ensure default passes tests
    for key, (value, test) in DEFAULTS.items():
        if not np.all(test(value) == value):
            print(f"{key}: {test(value)} != {value}")
