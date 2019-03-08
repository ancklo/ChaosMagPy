import os
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
LIB = os.path.join(ROOT, 'lib')


# copied/inspired by matplotlib.rcsetup
def check_path_exists(s):
    """Test if path to file exists."""
    if s is None or s == 'None':
        return None
    if os.path.exists(s):
        return s
    else:
        raise FileNotFoundError(f'{s} does not exist.')


def check_float(s):
    """Test for a float."""
    try:
        return float(s)
    except ValueError:
        raise ValueError(f'Could not convert {s} to float.')


def check_string(s):
    """Conver to string."""
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


DEFAULTS = {
    'params.r_surf': [6371.2, check_float],
    'params.dipole': [np.array([-29442.0, -1501.0, 4797.1]),
                      lambda x: check_vector(x, len=3)],
    'params.version': ['6.x7', check_string],

    # location of coefficient files
    'file.RC_index': [os.path.join(LIB, 'RC_index.h5'),
                      check_path_exists],
    'file.GSM_spectrum': [os.path.join(LIB, 'frequency_spectrum_gsm.npz'),
                          check_path_exists],
    'file.SM_spectrum': [os.path.join(LIB, 'frequency_spectrum_sm.npz'),
                         check_path_exists],
    'file.Earth_conductivity': [os.path.join(LIB, 'Earth_conductivity.dat'),
                                check_path_exists],  # placeholder
}


class ConfigCHAOS(dict):

    defaults = DEFAULTS

    def __init__(self, *args, **kwargs):
        super().update(*args, **kwargs)

    def __setitem__(self, key, value):
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

    def reset(self):
        super().update({key: val for key, (val, _) in self.defaults.items()})

    def load(self, filepath):
        with open(filepath, 'r') as f:
            for line in f.readlines():
                # skip comment lines
                if line[0] == '#':
                    continue

                key, value = line.split(' = ')
                value = value.split('#')[0].rstrip()  # remove comments and \n

                # check list input and convert to array
                if value[0] == '[' and value[-1] == ']':
                    value = np.fromstring(value[1:-1], sep=' ')

                self.__setitem__(key, value)
        f.close()

    def save(self, filepath):
        with open(filepath, 'w') as f:
            for key, value in self.items():
                f.write(f'{key} = {value}\n')
        f.close()


# load defaults
configCHAOS = ConfigCHAOS({key: val for key, (val, _) in DEFAULTS.items()})


def rc(key, value):
    configCHAOS[key] = value


if __name__ == '__main__':
    # ensure default passes tests
    for key, (value, test) in DEFAULTS.items():
        if not test(value) == value:
            print(f"{key}: {test(value)} != {value}")
