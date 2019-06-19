#!/usr/bin/bash
#sh install.sh [--user]
git clone https://github.com/eigenteam/eigen-git-mirror.git
git clone https://github.com/pybind/pybind11.git
export EIGEN=`pwd`/eigen-git-mirror
export PYBIND=`pwd`/pybind11/include
echo $EIGEN
python setup.py install $1
echo "Cleaning up"
rm -rf $EIGEN
rm -rf `pwd`/pybind11
rm -rf `pwd`/build
rm -rf `pwd`/dist
rm -rf `pwd`/xclib.egg-info
