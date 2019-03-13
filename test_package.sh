#!/usr/bin/env bash

source activate  # needed here, run script from conda environment

# extract from __init__.py on line with __version__ the expr between ""
latest=$(grep __version__ chaosmagpy/__init__.py | sed 's/.*"\(.*\)".*/\1/')

read -p "Enter version [$latest]: " version
version=${version:-$latest}

env_name=Test_$version

printf "%s %s %s %s\n\n" -------Test ChaosMagPy Version $version-------

echo Setting up fresh conda environment.
conda env create --name $env_name -f environment.yml
conda activate $env_name

conda env list

echo Installing requested version of ChaosMagPy v$latest.
pip install dist/chaosmagpy-$version.tar.gz

echo Entering test directory.
cd tests

echo python -m unittest test_chaosmagpy
python -m unittest test_chaosmagpy

echo python -m unittest test_coordinate_utils
python -m unittest test_coordinate_utils

echo python -m unittest test_model_utils
python -m unittest test_model_utils

echo python -m unittest test_data_utils
python -m unittest test_data_utils

conda activate base
conda remove --name $env_name --all
