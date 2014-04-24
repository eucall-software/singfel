# This is my README

Dependencies:
ARMADILLO
PYTHON
BOOST
HDF5
Optional Dependencies:
MPI

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Installing dependencies without sudo:
PYTHON
This should already exist in standard Linux

BOOST
./bootstrap.sh --prefix=/path/to/boost/directory --with-python=/which/python
Add "using mpi ;" in /path/to/boost/directory/project-config.jam
./b2 install

ARMADILLO
cmake .
make
make install DESTDIR=/path/to/armadillo/directory
Note: If you to use svd(), uncomment #define ARMA_USE_LAPACK in 
/path/to/armadillo/directory/include/armadillo_bits/config.hpp
You don't need to recompile armadillo

HDF5
./configure --prefix=/path/to/hdf/directory --enable-cxx
make
make install
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Add the following lines to .bashrc which is used by CMake
# HDF5
export HDF5_DIR=/path/to/hdf5/directory
# ARMADILLO
export ARMA_DIR=/path/to/armadillo/directory
# BOOST
export BOOST_DIR=/path/to/boost/directory

# SINGFEL
export LIBRARY_PATH=/path/to/SingFEL/build/src:$LIBRARY_PATH
export LD_RUN_PATH=/path/to/SingFEL/build/src:$LD_RUN_PATH

# MPICH
export PATH=/data/yoon/mpich-install/bin:$PATH
export LD_LIBRARY_PATH=/opt/intel/2013/lib/intel64:$LD_LIBRARY_PATH

To compile programs:
$mkdir build
$cmake ..
$make
The executables will be placed in /path/to/SingFEL/build/src

When recompiling, remember to delete /path/to/SingFEL/build/CmakeCache.txt

# DOXYGEN
Documentation of the source code
Using man pages to browse documentation
Add the following lines to .bashrc
export MANPATH=/path/to/SingFEL/man:$MANPATH
e.g.: $man beam_CBeam
will display manual for the beam class

Using html to browse documentation
$firefox html/index.html

Using pdf to browse documentation
$cd latex && make && evince refman.pdf
