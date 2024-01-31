#!/usr/bin/env bash
# need to build the package first
# run script from conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

# extract from __init__.py on line with __version__ the expr between ""
latest=$(grep __version__ chaosmagpy/__init__.py | sed 's/.*"\(.*\)".*/\1/')

read -p "Enter version [$latest]: " version
version=${version:-$latest}

env_name=Test_$version

printf "\n%s %s %s %s\n\n" -------Test ChaosMagPy Version $version-------

# echo Setting up fresh conda environment.
conda env create --name $env_name -f environment.yml

conda activate $env_name

conda env list

while true; do
    read -p "Install ChaosMagPy in the starred environment (y/n)?: " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) echo Abort.; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo Installing requested version of ChaosMagPy v$version.
pip install dist/chaosmagpy-$version.tar.gz --dry-run
pip install dist/chaosmagpy-$version-py3-none-any.whl

echo Entering test directory.
cd tests

echo python -m unittest test_chaos
python -m unittest test_chaos

echo python -m unittest test_coordinate_utils
python -m unittest test_coordinate_utils

echo python -m unittest test_model_utils
python -m unittest test_model_utils

echo python -m unittest test_data_utils
python -m unittest test_data_utils

conda activate base
conda remove --name $env_name --all
