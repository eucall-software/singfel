# This is my README

Dependencies:
ARMADILLO
PYTHON
BOOST
HDF5
MPI

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Installing dependencies without sudo:
PYTHON
This should already exist in standard Linux

## Do this before building boost ##
MPICH
./configure --disable-fortran --prefix=/path/to/mpich-install 2>&1 | tee c.txt
make 2>&1 | tee m.txt
make install 2>&1 | tee mi.txt
PATH=/path/to/mpich-install/bin:$PATH ; export PATH
Check that everything is in order at this point by doing:
      which mpicc
      which mpiexec

BOOST
./bootstrap.sh --prefix=/path/to/boost/directory --with-python=/which/python
Add "using mpi ;" in /path/to/boost/directory/project-config.jam
./b2 install

-ARMADILLO
cmake .
make
make install DESTDIR=/path/to/armadillo/directory
Note: If you want to use svd(), uncomment #define ARMA_USE_LAPACK in 
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

# Compiling SingFEL #

Dependencies:

* ARMADILLO
* PYTHON
* BOOST
* HDF5
* MPI

## Multi-user (with sudo) ##
For apt-get system:
### MPICH ###
    sudo apt-get install mpich2 libmpich2-dev

### HDF5 ###
    sudo apt-get install hdf5-tools libhdf5-mpich2-dev libhdf5-7 libhdf5-dev

Possible issues with HDF5: no rule to make libhdf5.so
    sudo ln -s /usr/lib/x86_64-linux-gnu/libhdf5.so /lib/libhdf5.so
    sudo ln -s /usr/lib/x86_64-linux-gnu/libhdf5_cpp.so /lib/libhdf5_cpp.so


### ARMADILLO ###
    sudo apt-get install libarmadillo-dev libarmadillo4
    
### BOOST ###
    sudo apt-get install libboost-dev libboost-mpi-dev libboost-python-dev libboost-thread-dev libboost-date-time-dev libboost-system-dev libboost-filesystem-dev libboost-test-dev
    #libboost-all-dev

### GSL ###
    sudo apt-get install gsl-bin libgsl0-dev


## Single-user (without sudo) ##
    export DEST_ROOT=$(pwd)

### PYTHON ###
This should already exist in standard Linux

### MPICH ###
    tar -xf mpich-*.tar.gz
    cd $(echo mpich-*/)
    echo "Configure MPICH [$(date)]" > ../mpich.log
    ./configure --disable-fortran --prefix=$DEST_ROOT/mpich 2>&1 >> ../mpich.log
    echo "Make MPICH" >> ../mpich.log
    make 2>&1 >> ../mpich.log
    echo "Install MPICH" >> ../mpich.log
    make install 2>&1 >> ../mpich.log
    echo 'export PATH='$DEST_ROOT'/mpich/bin:$PATH' >> ~/.profile
    echo 'export LD_LIBRARY_PATH=/opt/intel/2013/lib/intel64:$LD_LIBRARY_PATH' >> ~/.profile

### BOOST ###
    tar -xf boost_*.tar.gz
    cd $(echo boost_*/)
    echo "Booststrap BOOST [$(date)]" > ../boost.log
    ./bootstrap.sh --prefix=$DEST_ROOT/boost --with-python=$(which python) -with-libraries=thread,date_time,system,filesystem,test,mpi,python,program_options,serialization 2>&1 >> ../boost.log
    sed -i 's/import feature ;/import feature ;\nusing mpi ;/g' project-config.jam
    echo "Install BOOST" >> ../boost.log
    ./b2 install threading=multi 2>&1 >> ../boost.log
    echo 'export BOOST_DIR='$DEST_ROOT'/boost' >> ~/.profile
    
### ARMADILLO ###
    tar -xf armadillo-*.tar.gz
    cd $(echo armadillo-*/)
    echo "cmake Armadillo [$(date)]" > ../arma.log
    cmake CMakeLists.txt -DCMAKE_INSTALL_PREFIX=$DEST_ROOT/arma 2>&1 >> ../arma.log
    echo "make Armadillo" >> ../arma.log
    make 2>&1 >> ../arma.log
    echo "install Armadillo" >> ../arma.log
    make install 2>&1 >> ../arma.log
    echo 'export ARMA_DIR='$DEST_ROOT'/arma' >> ~/.profile

### HDF5 ###
    tar -xf hdf5-*.tar.gz
    cd $(echo hdf5-*/)
    echo "cmake HDF5 [$(date)]" > ../hdf5.log
    ./configure --prefix=$DEST_ROOT/hdf5 --enable-cxx >> ../hdf5.log
    echo "make HDF5" >> ../hdf5.log
    make 2>&1 >> ../hdf5.log
    echo "install HDF5" >> ../hdf5.log
    make install 2>&1 >> ../hdf5.log
    echo 'export HDF5_DIR='$DEST_ROOT'/hdf5' >> ~/.profile
    
## SingFEL ##
    
    echo 'export LIBRARY_PATH='$DEST_ROOT'/singfel/build/src:$LIBRARY_PATH' >> ~/.profile
    echo 'export LD_RUN_PATH='$DEST_ROOT'/singfel/build/src:$LD_RUN_PATH' >> ~/.profile
    . ~/.profile
	mkdir $DEST_ROOT/singfel/build && cd $DEST_ROOT/singfel/build
	echo "cmake SingFEL [$(date)]" > ../../singfel.log
	cmake ../CMakeLists.txt 2>&1 >> ../../singfel.log
	echo "make SingFEL" >> ../../singfel.log
	make 2>&1 >> ../../singfel.log

The executables will be placed in SingFEL/build/src

When recompiling, remember to delete SingFEL/build/CmakeCache.txt    

## DOXYGEN ##
Documentation of the source code
Using man pages to browse documentation
Add the following lines to .bashrc

    export MANPATH=/path/to/SingFEL/man:$MANPATH

Usage:

    $man beam_CBeam

will display manual for the beam class

Using html to browse documentation

    $firefox html/index.html

Using pdf to browse documentation

    $cd latex && make && evince refman.pdf