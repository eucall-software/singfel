# This is my README

Dependencies:
ARMADILLO
PYTHON
BOOST
HDF5
Optional Dependencies:
MPI

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

To compile programs:
$mkdir build
$cmake ..
$make
The executables will be placed in /path/to/SingFEL/build/src

When recompiling, remember to delete /path/to/SingFEL/build/CmakeCache.txt
