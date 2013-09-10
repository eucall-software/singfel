
# - Try to find Intel Math Kernel Library (MKL)
# Once done, this will define
#  MKL_FOUND
#  MKL_INCLUDE_DIRS
#  MKL_LIBRARIES

# find where is "ifort"
if(os STREQUAL "Mac OS X" OR os STREQUAL "Linux")
	execute_process(COMMAND which ifort OUTPUT_VARIABLE ifort_path)
else(os STREQUAL "Mac OS X" OR os STREQUAL "Linux")
	message(FATAL_ERROR "Unsupported platform \"${OS}\"")
endif(os STREQUAL "Mac OS X" OR os STREQUAL "Linux")

if(ifort_path STREQUAL "")
	set(MKL_FOUND false)
	if(DEFINED MKL_FIND_REQUIRED)
		message(FATAL_ERROR "MKL is required, but not found")
	endif(DEFINED MKL_FIND_REQUIRED)
	return()
endif(ifort_path STREQUAL "")

# use "ifort_path" as a hint to get the path of MKL
execute_process(
	COMMAND echo ${ifort_path}
	COMMAND sed -e "s/\\/bin.*$//"
	OUTPUT_VARIABLE MKL_PATH
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(os STREQUAL "Mac OS X")
	set(MKL_PATH ${MKL_PATH}/Frameworks/mkl)
elseif(os STREQUAL "Linux")
	set(MKL_PATH ${MKL_PATH}/mkl)
endif(os STREQUAL "Mac OS X")

set(MKL_INCLUDE_DIR ${MKL_PATH}/include)
set(MKL_LIBRARY ${MKL_PATH}/lib)

set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
if(mach STREQUAL "32")
	set(MKL_LIBRARIES
		${MKL_LIBRARY}/32/libmkl_intel.dylib
		${MKL_LIBRARY}/32/libmkl_sequential.dylib
		${MKL_LIBRARY}/32/libmkl_core.dylib
		-lpthread
	)
elseif(mach STREQUAL "64")
	set(MKL_LIBRARIES
		${MKL_LIBRARY}/em64t/libmkl_intel_lp64.dylib
		${MKL_LIBRARY}/em64t/libmkl_sequential.dylib
		${MKL_LIBRARY}/em64t/libmkl_core.dylib
		-lpthread
	)
endif(mach STREQUAL "32")

message(STATUS "Found Intel MKL under ${MKL_PATH}")

set(MKL_FOUND true)
