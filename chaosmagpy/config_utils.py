import os

ROOT = os.path.abspath(os.path.dirname(__file__))
LIB = os.path.join(ROOT, 'lib')


# copied/inspired by matplotlib.rcsetup
def check_path_exists(s):
    """Test if path to file exists."""
    if s is None:
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


default_config = {
    'params.r_surf': [6371.2, check_float],

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

    tests_dict = {key: test for key, (_, test) in default_config.items()}

    def __init__(self, *args, **kwargs):
        super().update(*args, **kwargs)

    def __setitem__(self, key, value):
        try:
            try:
                cval = self.tests_dict[key](value)
            except ValueError as err:
                raise ValueError(f'Key "{key}": {err}')
            super().__setitem__(key, cval)
        except KeyError:
            raise KeyError(f'"{key}" is not a valid parameter.')

    def __str__(self):
        return '\n'.join(map('{0[0]}: {0[1]}'.format, sorted(self.items())))

    @classmethod
    def from_file(self):
        pass

    @classmethod
    def from_defaults(self):
        cfg = {key: val for key, (val, _) in default_config.items()}
        return ConfigCHAOS(cfg)


configCHAOS = ConfigCHAOS.from_defaults()


def rc(key, value):
    configCHAOS[key] = value


if __name__ == '__main__':
    # ensure default passes tests
    for key, (value, test) in default_config.items():
        if not test(value) == value:
            print(f"{key}: {test(value)} != {value}")
