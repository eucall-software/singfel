INCLUDE_DIRECTORIES(${SINGFELPROJECT_SOURCE_DIR}/test)
LINK_DIRECTORIES(${SINGFELPROJECT_BINARY_DIR}/test)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SINGFELPROJECT_SOURCE_DIR}/test)

#find_package(CUDA)
IF(CUDA_FOUND)
  add_definitions(-DCOMPILE_WITH_CUDA) # set #define COMPILE_WITH_CUDA to 1 in .cpp
  # Place GPU tests here
ELSEIF(NOT CUDA_FOUND)
  ADD_EXECUTABLE(boostTest boostTest.cpp ${GENERAL_SRC})
  TARGET_LINK_LIBRARIES(boostTest ${GENERAL_LIB} ${MPI_LIBRARIES} )  
ENDIF()
