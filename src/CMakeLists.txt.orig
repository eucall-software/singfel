INCLUDE_DIRECTORIES(${SINGFELPROJECT_SOURCE_DIR}/src)
LINK_DIRECTORIES(${SINGFELPROJECT_BINARY_DIR}/src)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "DEBUG")
  #SET(CMAKE_BUILD_TYPE "RELEASE")
  #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

FIND_PACKAGE(GSL REQUIRED)

FIND_PACKAGE(Armadillo 3.9 REQUIRED)
IF(HAVE_ARMADILLO)
  INCLUDE(ARMA_FindMKL) # Look for MKL
  MESSAGE("${ARMADILLO_INCLUDES}")
  MESSAGE("${ARMADILLO_LIBRARIES}")
  MESSAGE("${ARMADILLO_INCLUDE_DIRS}")
  INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${ARMADILLO_INCLUDES})
  LINK_DIRECTORIES(${ARMADILLO_LIBRARIES} ${HDF5_LIBRARY_DIRS})
  MESSAGE(STATUS "MKL_FOUND     = ${MKL_FOUND}")
  IF(MKL_FOUND)
    MESSAGE("MKL_INCLUDE_DIRS = ${MKL_INCLUDE_DIRS}")
    INCLUDE_DIRECTORIES(${MKL_INCLUDE_DIRS})
    #set(BOOST_ROOT /usr/local/include)
    LINK_DIRECTORIES(${ARMADILLO_LIBRARIES} ${MKL_INCLUDE_DIRS})
    LINK_LIBRARIES(iomp5)
  ENDIF()
ELSEIF(NOT HAVE_ARMADILLO)
  MESSAGE("ARMA NOT FOUND GOODBYE!!!")
ENDIF()

FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
#FIND_PACKAGE(Boost 1.53.0 COMPONENTS python thread date_time system)
IF(Boost_FOUND)
  add_definitions(-DCOMPILE_WITH_BOOST) # set #define COMPILE_WITH_BOOST to 1 in .cpp
  MESSAGE("BOOST FOUND")
  MESSAGE("${Boost_INCLUDE_DIRS} ${Boost_LIBRARY_DIRS}")

  SET(Boost_USE_STATIC_LIBS ON)
  SET(Boost_USE_MULTITHREADED ON)
  SET(Boost_USE_STATIC_RUNTIME OFF)

  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
  LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${GSL_LIBRARIES}) # Deprecated but so convenient!

  PYTHON_ADD_MODULE(hello hello.cpp)
  FILE(COPY hello.py DESTINATION .)
  ADD_TEST(01-HelloWorld hello.py)

  PYTHON_ADD_MODULE(member member.cpp)
  FILE(COPY member.py DESTINATION .)
  ADD_TEST(04-HelloWorld member.py)

  PYTHON_ADD_MODULE(toolbox2 toolbox.cpp)
  FILE(COPY toolbox.py DESTINATION .)
  ADD_TEST(toolbox2 toolbox.py)

ELSEIF(NOT Boost_FOUND)
  MESSAGE("Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()

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

  CUDA_ADD_EXECUTABLE(radiationSINGFEL_gpu radiationDamage.cpp detector.cpp beam.cpp particle.cpp diffraction.cpp toolbox.cpp diffraction.cu)
  TARGET_LINK_LIBRARIES(radiationSINGFEL_gpu ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

  CUDA_ADD_EXECUTABLE(singlePattern_gpu spOptimize.cpp detector.cpp beam.cpp particle.cpp diffraction.cpp toolbox.cpp diffraction.cu)
  TARGET_LINK_LIBRARIES(singlePattern_gpu ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

  CUDA_ADD_LIBRARY(SINGFEL SHARED spOptimize.cpp detector.cpp beam.cpp particle.cpp diffraction.cpp toolbox.cpp diffraction.cu io.cpp)
  TARGET_LINK_LIBRARIES(SINGFEL ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

ELSEIF(NOT CUDA_FOUND)
  MESSAGE("damn! no cuda")

<<<<<<< HEAD
  ADD_EXECUTABLE(radiationSINGFEL radiationDamage.cpp detector.cpp beam.cpp particle.cpp diffraction.cpp toolbox.cpp diffraction.cu)
  TARGET_LINK_LIBRARIES(radiationSINGFEL ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

  ADD_EXECUTABLE(singlePattern spOptimize.cpp detector.cpp beam.cpp particle.cpp diffraction.cpp toolbox.cpp diffraction.cu)
  TARGET_LINK_LIBRARIES(singlePattern ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})
  
  ADD_LIBRARY(SINGFEL SHARED spOptimize.cpp detector.cpp beam.cpp particle.cpp diffraction.cpp toolbox.cpp io.cpp diffraction.cu)
=======
  ADD_EXECUTABLE(radiationSINGFEL radiationDamage.cpp detector.cpp beam.cpp particle.cpp diffraction.cpp toolbox.cpp)
  TARGET_LINK_LIBRARIES(radiationSINGFEL ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

  ADD_EXECUTABLE(cpuSINGFEL spOptimize.cpp detector.cpp beam.cpp particle.cpp diffraction.cpp toolbox.cpp)
  TARGET_LINK_LIBRARIES(cpuSINGFEL ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})
  
  ADD_LIBRARY(SINGFEL SHARED spOptimize.cpp detector.cpp beam.cpp particle.cpp diffraction.cpp toolbox.cpp io.cpp)
>>>>>>> 6144a612fef8378771a8ee3313c12f441672d62f
  TARGET_LINK_LIBRARIES(SINGFEL ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

ENDIF()
