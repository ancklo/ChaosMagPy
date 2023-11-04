# Copyright (C) 2023 Technical University of Denmark
#
# This file is part of ChaosMagPy.
#
# ChaosMagPy is released under the MIT license. See LICENSE in the root of the
# repository for full licensing details.

import hdf5storage as hdf


def load_matfile(filepath, test_name):
    """
    Load matfile and return dictionary of MATLAB outputs for comparing with
    Python outputs.

    Parameters
    ----------
    filepath : str
        Filepath to mat-file.
    test_name : str
        Name of test which is stored as structure in mat-file.

    Returns
    -------
    test : dict
        Dictionary containing MATLAB output.
    """

    mat_contents = hdf.loadmat(str(filepath), variable_names=[str(test_name)])
    test = mat_contents[str(test_name)]

    if test.ndim == 2:
        return test[0, 0]  # before v7.3
    else:
        return test[0]  # hdf5 compatible v7.3
