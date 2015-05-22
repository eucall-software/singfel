INCLUDE_DIRECTORIES(${SINGFELPROJECT_SOURCE_DIR}/src)
LINK_DIRECTORIES(${SINGFELPROJECT_BINARY_DIR}/src)
				
find_package(CUDA)
IF(CUDA_FOUND)
  MESSAGE("CUDA FOUND")
  add_definitions(-DCOMPILE_WITH_CUDA) # set #define COMPILE_WITH_CUDA to 1 in .cpp
  
  #CUDA_ADD_EXECUTABLE(radiationSINGFEL_old radiationDamage.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(radiationSINGFEL_old ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})

  #CUDA_ADD_EXECUTABLE(MaximumXCorrelation MaximumXCorrelation.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(MaximumXCorrelation ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})  

  #CUDA_ADD_EXECUTABLE(MaximumXCorrelationMPI MaximumXCorrelationMPI.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(MaximumXCorrelationMPI ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${Boost_LIBRARIES})   
  
  CUDA_ADD_EXECUTABLE(radiationDamageMPI radiationDamageMPI.cpp ${GENERAL_SRC} ${LIBSINGFEL}/diffraction.cu)
  TARGET_LINK_LIBRARIES(radiationDamageMPI ${GENERAL_LIB} ${MPI_LIBRARIES})
  CUDA_ADD_EXECUTABLE(radiationDamage radiationDamage.cpp ${GENERAL_SRC} ${LIBSINGFEL}/diffraction.cu)
  TARGET_LINK_LIBRARIES(radiationDamage ${GENERAL_LIB})
  
  #gprof switches
  #set_target_properties(radiationDamageMPI PROPERTIES  COMPILE_FLAGS "-pg")
  #set_target_properties(radiationDamageMPI PROPERTIES  LINK_FLAGS "-pg")

  
  #if(MPI_COMPILE_FLAGS)
  #set_target_properties(MaximumXCorrelationMPI PROPERTIES
  #  COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  #endif()
  #if(MPI_LINK_FLAGS)
  #set_target_properties(MaximumXCorrelationMPI PROPERTIES
  #  LINK_FLAGS "${MPI_LINK_FLAGS}")
  #endif()
  
#  CUDA_ADD_EXECUTABLE(singlePattern_gpu spOptimize.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
#  TARGET_LINK_LIBRARIES(singlePattern_gpu ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

#  CUDA_ADD_EXECUTABLE(singlePatternQuaternion_gpu spQuaternion.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
#  TARGET_LINK_LIBRARIES(singlePatternQuaternion_gpu ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

#  CUDA_ADD_LIBRARY(SINGFEL SHARED spOptimize.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu ${LIBSINGFEL}/io.cpp)
#  TARGET_LINK_LIBRARIES(SINGFEL ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

ELSEIF(NOT CUDA_FOUND)
  MESSAGE("CUDA NOT FOUND")

  # Shared library: libsingfel
  #ADD_LIBRARY(singfel SHARED ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(singfel ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

  #ADD_EXECUTABLE(radiationSINGFEL_old radiationDamage.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(radiationSINGFEL_old ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})

  #ADD_EXECUTABLE(rotationCheck pmi_rotationCheckS2E.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(rotationCheck ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})


  ADD_EXECUTABLE(radiationDamageMPI radiationDamageMPI.cpp ${GENERAL_SRC})
  TARGET_LINK_LIBRARIES(radiationDamageMPI ${GENERAL_LIB} ${MPI_LIBRARIES})   

  #ADD_EXECUTABLE(singlePattern spOptimize.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(singlePattern ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES})

  #ADD_EXECUTABLE(spQuaternion spQuaternion.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(spQuaternion ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})

  #ADD_EXECUTABLE(spQuaternion spQuaternion.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(spQuaternion ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})
  
  #ADD_EXECUTABLE(makeDiffractionVolume makeDiffractionVolume.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(makeDiffractionVolume ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})
  
  #ADD_EXECUTABLE(makeDiffractionVolumeSingleAxisStream makeDiffractionVolumeSingleAxisStream.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(makeDiffractionVolumeSingleAxisStream ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})  
  #MESSAGE("CHUCK SAYS: ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES}")

  ADD_EXECUTABLE(mergeDiffractionVolume mergeDiffractionVolume.cpp ${GENERAL_SRC})
  TARGET_LINK_LIBRARIES(mergeDiffractionVolume ${GENERAL_LIB}) 
    
  #ADD_EXECUTABLE(MaximumXCorrelation MaximumXCorrelation.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(MaximumXCorrelation ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES} ${Boost_LIBRARIES})  
  
  #ADD_EXECUTABLE(MaximumXCorrelationMPI MaximumXCorrelationMPI.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(MaximumXCorrelationMPI ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${Boost_LIBRARIES})  

  #ADD_EXECUTABLE(reconMPI reconMPI.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/diffractionVolume.cpp ${LIBSINGFEL}/diffractionPattern.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(reconMPI ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${Boost_LIBRARIES})  

#  ADD_EXECUTABLE(slavesMPI slavesMPI.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
#  TARGET_LINK_LIBRARIES(slavesMPI ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${Boost_LIBRARIES})  
#  MESSAGE("CHUCK MERGE SAYS: ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_LIBRARIES}") 
  #if(MPI_COMPILE_FLAGS)
  #set_target_properties(MaximumXCorrelationMPI PROPERTIES
  #  COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  #endif()
  #if(MPI_LINK_FLAGS)
  #set_target_properties(MaximumXCorrelationMPI PROPERTIES
  #  LINK_FLAGS "${MPI_LINK_FLAGS}")
  #endif()

  #ADD_EXECUTABLE(expandVolume expandVolume.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(expandVolume ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})  

  #ADD_EXECUTABLE(mergeDiffractionVolume_4HWY mergeDiffractionVolume_4HWY.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(mergeDiffractionVolume_4HWY ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})  
  #MESSAGE("CHUCK MERGE SAYS: ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES}") 

  #ADD_EXECUTABLE(unitTest unitTest.cpp ${LIBSINGFEL}/temp.cpp ${LIBSINGFEL}/io.cpp ${LIBSINGFEL}/detector.cpp ${LIBSINGFEL}/beam.cpp ${LIBSINGFEL}/particle.cpp ${LIBSINGFEL}/diffraction.cpp ${LIBSINGFEL}/toolbox.cpp ${LIBSINGFEL}/diffraction.cu)
  #TARGET_LINK_LIBRARIES(unitTest ${ARMADILLO_LIBRARIES} ${GSL_LIBRARIES} ${HDF5_LIBRARIES})

  #ADD_EXECUTABLE(boostTest boostTest.cpp ${LIBSINGFEL}/toolbox.cpp)
  #TARGET_LINK_LIBRARIES(boostTest ${Boost_LIBRARIES} ${ARMADILLO_LIBRARIES})  

ENDIF()
