import os

ROOT = os.path.abspath(os.path.dirname(__file__))
LIB = os.path.join(ROOT, 'lib')


# copied/inspired by matplotlib.rcsetup
def test_path_exists(s):
    """Test if path to file exists."""
    if s is None:
        return None
    if os.path.exists(s):
        return s
    else:
        raise RuntimeError(f'{s} does not exist.')


default_config = {
    'RC_index_file': [os.path.join(LIB, 'RC_default.h5'),
                      test_path_exists],  # placeholder
    'GSM_spectrum_file': [os.path.join(LIB, 'frequency_spectrum_gsm.npz'),
                          test_path_exists],  # placeholder
    'SM_spectrum_file': [os.path.join(LIB, 'frequency_spectrum_sm.npz'),
                         test_path_exists],  # placeholder
    'Earth_conductivity_file': [os.path.join(LIB, 'Earth_conductivity.dat'),
                                test_path_exists],  # placeholder
}


if __name__ == '__main__':
    config = default_config
    # ensure default passes tests
    for key, (value, test) in config.items():
        if not test(value) == value:
            print(f"{key}: {test(value)} != {value}")
