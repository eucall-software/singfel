INCLUDE_DIRECTORIES(${SINGFELPROJECT_SOURCE_DIR}/src)
LINK_DIRECTORIES(${SINGFELPROJECT_BINARY_DIR}/src)

SET(LIBSINGFEL ${SINGFELPROJECT_SOURCE_DIR}/libsingfel)
INCLUDE_DIRECTORIES(${LIBSINGFEL})
LINK_DIRECTORIES(${LIBSINGFEL})

MESSAGE("HELLO EVERYBODY!!")
MESSAGE("${SINGFELPROJECT_SOURCE_DIR}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "DEBUG")
  #SET(CMAKE_BUILD_TYPE "RELEASE")
  #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

#FIND_PACKAGE(HDF5 REQUIRED)
#IF(HDF5_FOUND)
#  MESSAGE("HDF5 FOUND!!!")
#  MESSAGE("${HDF5_INCLUDE_DIRS} ${HDF5_LIBRARIES}")
#  INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})
#  LINK_DIRECTORIES(${HDF5_LIBRARIES})
#ELSE()
#  MESSAGE("HDF5 IS NOT FOUND!!!!")
#ENDIF()
############ HACK ############
set(HDF5_INCLUDE_DIRS "$ENV{HDF5_DIR}/include")
set(HDF5_LIBRARIES "$ENV{HDF5_DIR}/lib/libhdf5.so" "$ENV{HDF5_DIR}/lib/libhdf5_cpp.so")
#set(HDF5_INCLUDE_DIRS $ENV{HDF5_INCLUDE_DIRS}) # /home/beams/EPIX34ID/yoon/hdf5-1.8.12/include)
#set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS} /home/beams/EPIX34ID/yoon/hdf5-1.8.12/include)
#set(HDF5_LIBRARIES ${HDF5_LIBRARIES} /home/beams/EPIX34ID/yoon/hdf5-1.8.12/lib/libhdf5.so /home/beams/EPIX34ID/yoon/hdf5-1.8.12/lib/libhdf5_cpp.so) #/data/yoon/hdf5-1.8.12/lib/libhdf5.so /data/yoon/hdf5-1.8.12/lib/libhdf5_cpp.so)
MESSAGE("CHUCK SAYS: ${HDF5_INCLUDE_DIRS}")
##############################
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#include_directories(${HDF5_INCLUDE_DIRS})
#link_directories(${HDF5_LIBRARIES})

FIND_PACKAGE(GSL REQUIRED)

FIND_PACKAGE(Armadillo 3.9 REQUIRED)
IF(HAVE_ARMADILLO)
  INCLUDE(ARMA_FindMKL) # Look for MKL
  MESSAGE("CHUCK ARMA SAYS: ${ARMADILLO_INCLUDES}")
  #SET(ARMADILLO_LIBRARIES "ENV{ARMA_DIR}/libarmadillo.so")
  MESSAGE("CHUCK ARMA SAYS: ${ARMADILLO_LIBRARIES}")
  MESSAGE("CHUCK ARMA SAYS: ${ARMADILLO_INCLUDE_DIRS}")
  INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${ARMADILLO_INCLUDES})
  LINK_DIRECTORIES(${ARMADILLO_LIBRARIES} ${HDF5_LIBRARIES})
  MESSAGE(STATUS "MKL_FOUND     = ${MKL_FOUND}")
  IF(MKL_FOUND)
    MESSAGE("MKL_INCLUDE_DIRS = ${MKL_INCLUDE_DIRS}")
    INCLUDE_DIRECTORIES(${MKL_INCLUDE_DIRS})
    #set(BOOST_ROOT /usr/local/include)
    LINK_DIRECTORIES(${ARMADILLO_LIBRARIES} ${MKL_INCLUDE_DIRS})
    LINK_LIBRARIES(iomp5)
  ENDIF()
ELSEIF(NOT HAVE_ARMADILLO)
  MESSAGE("ARMA NOT FOUND. GOODBYE!!!")
ENDIF()

FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
SET(BOOST_ROOT $ENV{BOOST_DIR})
FIND_PACKAGE(Boost 1.55.0 REQUIRED COMPONENTS python thread date_time system mpi serialization filesystem)
IF(Boost_FOUND)
  add_definitions(-DCOMPILE_WITH_BOOST) # set #define COMPILE_WITH_BOOST to 1 in .cpp
  MESSAGE("BOOST FOUND")
  MESSAGE("CHUCK SAYS: ${Boost_INCLUDE_DIRS} ${Boost_LIBRARY_DIRS} ${Boost_LIBRARIES} ${BOOST_ROOT}")

#  SET(Boost_USE_STATIC_LIBS ON)
#  SET(Boost_USE_MULTITHREADED ON)
#  SET(Boost_USE_STATIC_RUNTIME OFF)

  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
  LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${GSL_LIBRARIES}) # Deprecated but so convenient!

  #PYTHON_ADD_MODULE(hello hello.cpp)
  #FILE(COPY hello.py DESTINATION .)
  #ADD_TEST(01-HelloWorld hello.py)

  #PYTHON_ADD_MODULE(member member.cpp)
  #FILE(COPY member.py DESTINATION .)
  #ADD_TEST(04-HelloWorld member.py)

  #PYTHON_ADD_MODULE(toolbox2 toolbox.cpp)
  #FILE(COPY toolbox.py DESTINATION .)
  #ADD_TEST(toolbox2 toolbox.py)

ELSEIF(NOT Boost_FOUND)
  MESSAGE("Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()
###### FIND BOOST WORKS, BUT LET'S USE LOCAL INSTALL ######
#set(Boost_INCLUDE_DIRS "$ENV{BOOST_DIR}/include")
#FILE(GLOB myBoostLibs "$ENV{BOOST_DIR}/lib/libboost*.so")
#set(Boost_LIBRARIES ${myBoostLibs})
#MESSAGE("CHUCKIE BOOST SAYS: ${Boost_INCLUDE_DIRS} ${Boost_LIBRARY_DIRS} ${Boost_LIBRARIES}")
#INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

# MPICH
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
MESSAGE("CHUCK MPI SAYS: ${MPI_INCLUDE_PATH} ${MPI_LIBRARY_DIRS} ${MPI_LIBRARIES}")

SET(CMAKE_CPP_FLAGS "-g -Wall")

IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_DEFINITIONS("-Wall")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()

message("${CMAKE_SOURCE_DIR}")
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})

MESSAGE("${CMAKE_MODULE_PATH}")

find_package(CUDA)
IF(CUDA_FOUND)
  MESSAGE("cuda found")
  MESSAGE("${CUDA_LIBRARIES}")
  MESSAGE("${ARMADILLO_LIBRARIES}")
    
  add_definitions(-DCOMPILE_WITH_CUDA) # set #define COMPILE_WITH_CUDA to 1 in .cpp

  CUDA_ADD_EXECUTABLE(radiationSINGFEL radiationDamageS2E.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  TARGET_LINK_LIBRARIES(radiationSINGFEL ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})

#  CUDA_ADD_EXECUTABLE(singlePattern_gpu spOptimize.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
#  TARGET_LINK_LIBRARIES(singlePattern_gpu ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

#  CUDA_ADD_EXECUTABLE(singlePatternQuaternion_gpu spQuaternion.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
#  TARGET_LINK_LIBRARIES(singlePatternQuaternion_gpu ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

#  CUDA_ADD_LIBRARY(SINGFEL SHARED spOptimize.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu ${LIBSINGFEL}/io.cpp)
#  TARGET_LINK_LIBRARIES(SINGFEL ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

ELSEIF(NOT CUDA_FOUND)
  MESSAGE("damn! no cuda")

  ADD_EXECUTABLE(radiationSINGFEL radiationDamageS2E.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  TARGET_LINK_LIBRARIES(radiationSINGFEL ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})

  #ADD_EXECUTABLE(singlePattern spOptimize.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(singlePattern ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

  #ADD_EXECUTABLE(singlePatternQuaternion spQuaternion.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(singlePatternQuaternion ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

  ADD_EXECUTABLE(spQuaternion spQuaternion.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  TARGET_LINK_LIBRARIES(spQuaternion ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})
  
  #ADD_LIBRARY(SINGFEL SHARED spOptimize.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(SINGFEL ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

  #ADD_EXECUTABLE(makeDiffractionVolume makeDiffractionVolume.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(makeDiffractionVolume ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})
  
  #ADD_EXECUTABLE(makeDiffractionVolumeSingleAxisStream makeDiffractionVolumeSingleAxisStream.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(makeDiffractionVolumeSingleAxisStream ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})  
  #MESSAGE("CHUCK SAYS: ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES}")

  #ADD_EXECUTABLE(mergeDiffractionVolume_chignolin mergeDiffractionVolume_chignolin.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(mergeDiffractionVolume_chignolin ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})  
  #MESSAGE("CHUCK MERGE SAYS: ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES}")

  #ADD_EXECUTABLE(mergeDiffractionVolume_1JB0 mergeDiffractionVolume_1JB0.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(mergeDiffractionVolume_1JB0 ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})  
  #MESSAGE("CHUCK MERGE SAYS: ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES}")  
  
  #ADD_EXECUTABLE(mergeDiffractionVolume_testX mergeDiffractionVolume_testX.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(mergeDiffractionVolume_testX ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})  
  #MESSAGE("CHUCK MERGE SAYS: ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES}")   
  
  #ADD_EXECUTABLE(MaximumXCorrelation MaximumXCorrelation.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(MaximumXCorrelation ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})  
  #MESSAGE("CHUCK MERGE SAYS: ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES}")
  
  ADD_EXECUTABLE(MaximumXCorrelationMPI MaximumXCorrelationMPI.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  TARGET_LINK_LIBRARIES(MaximumXCorrelationMPI ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${Boost_LIBRARIES})  
  MESSAGE("CHUCK MERGE SAYS: ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_LIBRARIES}")  
  #if(MPI_COMPILE_FLAGS)
  #set_target_properties(MaximumXCorrelationMPI PROPERTIES
  #  COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  #endif()
  #if(MPI_LINK_FLAGS)
  #set_target_properties(MaximumXCorrelationMPI PROPERTIES
  #  LINK_FLAGS "${MPI_LINK_FLAGS}")
  #endif()

  #ADD_EXECUTABLE(mergeDiffractionVolume_4HWY mergeDiffractionVolume_4HWY.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(mergeDiffractionVolume_4HWY ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})  
  #MESSAGE("CHUCK MERGE SAYS: ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES}") 

  ADD_EXECUTABLE(unitTest unitTest.cpp ${LIBSINGFEL}/temp.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  TARGET_LINK_LIBRARIES(unitTest ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})

ENDIF()
