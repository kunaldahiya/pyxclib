#!/usr/bin/bash
#sh install.sh [--user]
git clone https://github.com/eigenteam/eigen-git-mirror.git
export EIGEN=`pwd`/eigen-git-mirror
echo $EIGEN
python setup.py install $1
rm -rf $EIGEN
