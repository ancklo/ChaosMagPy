#!/usr/bin/env bash

# extract from __init__.py on line with __version__ the expr between ""
version=$(grep __version__ chaosmagpy/__init__.py | sed 's/.*"\(.*\)".*/\1/')
out=chaosmagpy_package_$version.zip
chaos=data/CHAOS-7.2.mat

echo ---------------- ChaosMagPy Version $version ----------------
echo "Building package with CHAOS-matfile in '$chaos'."

while true; do
    read -p "Do you wish to continue building v$version (y/n)?: " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) echo Exit.; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

# delete old dist files if exist
rm -f dist/chaosmagpy-$version-py3-none-any.whl
rm -f dist/chaosmagpy-$version.tar.gz

# build distribution, build binay in tmp/tmpwheel because of windows shared dir
tempdir=$(mktemp -t -d XXXXXX)
echo "Creating temporary directory '$tempdir'"
python setup.py sdist bdist_wheel --bdist-dir $tempdir

# clean build and compile documentary as html
rm -rf ./docs/build/
make --directory ./docs html

# make temporary directory
tempdir=$(mktemp -t -d XXXXXX)
echo "Creating temporary directory '$tempdir'"
mkdir $tempdir/data $tempdir/html

# copy files to tmp directory
cp dist/chaosmagpy-$version.tar.gz $tempdir/.
cp chaos_examples.py $tempdir/.
cp $chaos $tempdir/data/.

# run example
cd $tempdir/
python $tempdir/chaos_examples.py
cat example1_output.txt  # print example file output for checking
cd -

cp data/SW_OPER_MAGA_LR_1B_20180801T000000_20180801T235959_PT15S.cdf $tempdir/data/SW_OPER_MAGA_LR_1B_20180801T000000_20180801T235959_PT15S.cdf
cp -r docs/build/html/* $tempdir/html/

# create readme.txt and write introduction (without citation rst link)
cat > $tempdir/readme.txt <(head -9 README.rst)

# create readme.txt and write documentation information
cat >> $tempdir/readme.txt <(tail -17 README.rst)

# create readme.txt and write introduction
cat >> $tempdir/readme.txt <(echo) CITATION

# include License
cat >> $tempdir/readme.txt <(echo) LICENSE

# include installation instructions
cat >> $tempdir/readme.txt <(echo) INSTALL.rst

# append explanations of package's contents
cat >> $tempdir/readme.txt << EOF

Contents
========

The directory contains the files/directories:

1. "chaosmagpy-x.x*.tar.gz": pip installable archive of the chaosmagpy package
   (version x.x*)

2. "chaos_examples.py": executable Python script containing several examples
   that can be run by changing the examples in line 16, save and run in the
   command line:

   >>> python chaos_examples.py

   example 1: Calculate CHAOS model field predictions from input coordinates
              and time and output simple data file
   example 2: Calculate and plot residuals between CHAOS model and
              Swarm A data (from L1b MAG cdf data file, example from May 2014).
   example 3: Calculate core field and its time derivatives for specified times
              and radii and plot maps
   example 4: Calculate static (i.e. small-scale crustal) magnetic field and
              plot maps (may take a few seconds)
   example 5: Calculate timeseries of the magnetic field at a ground
              observatory and plot
   example 6: Calculate external and associated induced fields described in SM
              and GSM reference systems and plot maps

3. "data/CHAOS-x.x.mat": mat-file containing the CHAOS model version x.x.

4. "SW_OPER_MAGA_LR_1B_20180801T000000_20180801T235959_PT15S.cdf":
   cdf-file containing Swarm A magnetic field data from August 1, 2018.

5. directory called "html" containing the built documentation as
   html-files. Open "index.html" in your browser to access the main site.


Clemens Kloss (ancklo@space.dtu.dk)

EOF

# include changelog
cat >> $tempdir/readme.txt <(echo) CHANGELOG.rst

# build archive recursively in tmp directory
cd $tempdir/
zip $out -r *
cd -

# move archive to build
mv -i $tempdir/$out build/$out

# clean up
while true; do
    read -p "Do you wish to delete temporary files in '$tempdir' (y/n)?: " yn
    case $yn in
        [Yy]* ) rm -r $tempdir; break;;
        [Nn]* ) echo Exit.; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

cat << EOF
-------------------------------------------------------------
Check creation date of dist files (*.tar.gz and *.whl).
Check license date.
Check changelog date.
Check url to CHAOS at the top of the usage section.
Check overview section in the documentation.
Check Python example content and copied data (CHAOS matfile).
Check basicConfig default CHAOS version.
-------------------------------------------------------------
EOF

# upload to PyPI
# twine check dist/chaosmagpy-$version*
# twine upload dist/chaosmagpy-$version*
