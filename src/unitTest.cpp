#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <armadillo>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include "detector.h"
#include "beam.h"
#include "particle.h"
#include "diffraction.h"
#include "toolbox.h"
#include "diffraction.cuh"

#include "io.h"

#include <algorithm>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
#include <string>

#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

const H5std_string	FILE_NAME( "dset.h5" );
const H5std_string	DATASET_NAME( "dset" );
const int 	DIM0 = 3;                    // dataset dimensions
const int 	DIM1 = 4;
const int 	DIM2 = 2;

//#define ARMA_NO_DEBUG

#define USE_CUDA 0
#define USE_CHUNK 0

int main( int argc, char* argv[] ){

	// Particle
	CParticle particle = CParticle();
	CBeam beam = CBeam();
	CDetector det = CDetector();

	typedef boost::tokenizer<boost::char_separator<char> > Tok;

	// Data initialization.
    // Write based on h5_rdwt.cpp: http://www.hdfgroup.org/HDF5/Tutor/rdwt.html
    // Read based on readdata.cpp
    int i, j, k;

	int data[1];
	data[0] = 6;

	int data0[DIM0];
	int counter = 0;
	for (j = 0; j < DIM0; j++)
		data0[j] = counter++;

	float data1[DIM0][DIM1];
	counter = 0;
	for (j = 0; j < DIM0; j++)
		for (i = 0; i < DIM1; i++)
			data1[j][i] = counter++;

	float data2[DIM0][DIM1][DIM2];
	counter = 0;
	for (k = 0; k < DIM0; k++)
		for (j = 0; j < DIM1; j++)
			for (i = 0; i < DIM2; i++)
				data2[k][j][i] = counter++;

   // Try block to detect exceptions raised by any of the calls inside it
   try
   {
      
      // Turn off the auto-printing when failure occurs so that we can
      // handle the errors appropriately
       
      Exception::dontPrint();

	  // Create a new file using the default property lists. 
     
      H5File file( FILE_NAME, H5F_ACC_TRUNC );

	  // WRITE SCALAR

      // Create the data space for the dataset.
      int myRank = 1;
      hsize_t dims[myRank];              // dataset dimensions
      dims[0] = 1;
      //dims[1] = 6;
      DataSpace dataspace ( 0, dims );
	  // Data type
      IntType datatype( PredType::NATIVE_INT );//IntType datatype( PredType::STD_I32BE );
      // Create the dataset.     
      DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
	  // Open an existing file and dataset.
      //H5File file( FILE_NAME, H5F_ACC_RDWR );
      dataset = file.openDataSet( DATASET_NAME );
      // Write the data to the dataset using default memory space, file
      // space, and transfer properties.
      dataset.write( data, PredType::NATIVE_INT );

	  // WRITE VECTOR

      // Create the data space for the dataset.
      myRank = 1;
      hsize_t dims0[myRank];              // dataset dimensions
      dims0[0] = DIM0;
      //dims[1] = 6;
      DataSpace dataspace0 ( myRank, dims0 );
	  // Data type
      IntType datatype0( PredType::NATIVE_INT );//IntType datatype( PredType::STD_I32BE );
      // Create the dataset.     
      DataSet dataset0 = file.createDataSet( "dset0", datatype0, dataspace0 );
	  // Open an existing file and dataset.
      //H5File file( FILE_NAME, H5F_ACC_RDWR );
      dataset0 = file.openDataSet( "dset0" );
      // Write the data to the dataset using default memory space, file
      // space, and transfer properties.
      dataset0.write( data0, PredType::NATIVE_INT );
      
	  // WRITE MATRIX

      // Create the data space for the dataset.
	  myRank = 2;
      hsize_t dims1[myRank];              // dataset dimensions
      dims1[0] = DIM0;
      dims1[1] = DIM1;
      DataSpace dataspace1 ( myRank, dims1 );
	  // Data type
      FloatType datatype1( PredType::NATIVE_FLOAT );
      // Create the dataset.     
      DataSet dataset1 = file.createDataSet( "dset1", datatype1, dataspace1 );
	  // Open an existing file and dataset.
      //H5File file( FILE_NAME, H5F_ACC_RDWR );
      dataset1 = file.openDataSet( "dset1" );
      // Write the data to the dataset using default memory space, file
      // space, and transfer properties.
      dataset1.write( data1, PredType::NATIVE_FLOAT );

	  // WRITE CUBE

      // Create the data space for the dataset.
	  myRank = 3;
      hsize_t dims2[myRank];              // dataset dimensions
      dims2[0] = DIM0;
      dims2[1] = DIM1;
      dims2[2] = DIM2;
      DataSpace dataspace2 ( myRank, dims2 );
	  // Data type
      FloatType datatype2( PredType::NATIVE_FLOAT );
      // Create the dataset.     
      DataSet dataset2 = file.createDataSet( "dset2", datatype2, dataspace2 );
	  // Open an existing file and dataset.
      //H5File file( FILE_NAME, H5F_ACC_RDWR );
      dataset2 = file.openDataSet( "dset2" );
      // Write the data to the dataset using default memory space, file
      // space, and transfer properties.
      dataset2.write( data2, PredType::NATIVE_FLOAT );
      
      // READ SCALAR
      /*
       * Get dataspace of the dataset.
       */
      dataspace = dataset.getSpace();
      /*
       * Get the number of dimensions in the dataspace.
       */
      int rank = dataspace.getSimpleExtentNdims();
      /*
       * Get the dimension size of each dimension in the dataspace and
       * display them.
       */
      hsize_t dims_out[rank];
      int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);
      cout << "rank " << rank << endl;
      // invalid dims_out
	  /*
       * Read data from hyperslab in the file into the hyperslab in
       * memory and display the data.
       */
	  int data_out[1];
	  data_out[0] = 0;
      dataset.read( data_out, PredType::NATIVE_INT);//, memspace, dataspace );
      cout << "SCALAR:" << endl;
      for (j = 0; j < 1; j++){
	  	cout << data_out[j] << endl;
	  }
	  
      // READ VECTOR
      /*
       * Get dataspace of the dataset.
       */
      dataspace = dataset0.getSpace();
      /*
       * Get the number of dimensions in the dataspace.
       */
      rank = dataspace.getSimpleExtentNdims();
      /*
       * Get the dimension size of each dimension in the dataspace and
       * display them.
       */
      hsize_t dims_out0[rank];
      ndims = dataspace.getSimpleExtentDims( dims_out0, NULL);
      cout << "rank " << rank << endl;
	  cout << "dimensions " <<
	      (unsigned long)(dims_out0[0]) << endl;
	  /*
       * Read data from hyperslab in the file into the hyperslab in
       * memory and display the data.
       */
	  int data_out0[rank];
	  for (j = 0; j < dims_out0[0]; j++) {
	  	data_out0[0] = 0;
	  }
      dataset0.read( data_out0, PredType::NATIVE_INT);//, memspace, dataspace );
      cout << "VECTOR:" << endl;
      for (j = 0; j < dims_out0[0]; j++){
	  	cout << setw(3) << data_out0[j] << " ";
	  }
	  cout << endl << endl;
	  	  
      // READ MATRIX
      /*
       * Get dataspace of the dataset.
       */
      dataspace = dataset1.getSpace();
      /*
       * Get the number of dimensions in the dataspace.
       */
      rank = dataspace.getSimpleExtentNdims();
      /*
       * Get the dimension size of each dimension in the dataspace and
       * display them.
       */
      hsize_t dims_out1[rank];
      ndims = dataspace.getSimpleExtentDims( dims_out1, NULL);
      cout << "rank " << rank << endl;
	  cout << "dimensions " <<
	      (unsigned long)(dims_out1[0]) << " x " <<
	      (unsigned long)(dims_out1[1]) << endl;
	  /*
       * Read data from hyperslab in the file into the hyperslab in
       * memory and display the data.
       */
	  float data_out1[dims_out1[0]][dims_out1[1]];
	  for (j = 0; j < dims_out1[0]; j++) {
	  for (i = 0; i < dims_out1[1]; i++) {
	  	data_out1[j][i] = 0;
	  }
	  }
      dataset1.read( data_out1, PredType::NATIVE_FLOAT);//, memspace, dataspace );
      cout << "MATRIX:" << endl;
	  for (j = 0; j < dims_out1[0]; j++) {
	  for (i = 0; i < dims_out1[1]; i++) {
	  	cout << setw(3) << data_out1[j][i] << " ";
	  }
	  cout << endl;
	  }
	  cout << endl << endl;
	  	      
      // READ CUBE
      /*
       * Get dataspace of the dataset.
       */
      dataspace = dataset2.getSpace();
      /*
       * Get the number of dimensions in the dataspace.
       */
      rank = dataspace.getSimpleExtentNdims();
      /*
       * Get the dimension size of each dimension in the dataspace and
       * display them.
       */
      hsize_t dims_out2[rank];
      ndims = dataspace.getSimpleExtentDims( dims_out2, NULL);
      cout << "rank " << rank << endl;
	  cout << "dimensions " <<
	      (unsigned long)(dims_out2[0]) << " x " <<
	      (unsigned long)(dims_out2[1]) << " x " <<
	      (unsigned long)(dims_out2[2]) << endl;
	  /*
       * Read data from hyperslab in the file into the hyperslab in
       * memory and display the data.
       */
	  float data_out2[dims_out2[0]][dims_out2[1]][dims_out2[2]];
	  for (k = 0; k < dims_out2[0]; k++) {
	  for (j = 0; j < dims_out2[1]; j++) {
	  for (i = 0; i < dims_out2[2]; i++) {
	  	data_out2[k][j][i] = 0;
	  }
	  }
	  }
      dataset2.read( data_out2, PredType::NATIVE_FLOAT);//, memspace, dataspace );
      cout << "CUBE:" << endl;
	  for (k = 0; k < dims_out2[0]; k++) {
	  for (j = 0; j < dims_out2[1]; j++) {
	  for (i = 0; i < dims_out2[2]; i++) {
	  	cout << setw(3) << data_out2[k][j][i] << " ";
	  }
	  cout << endl;
	  }
	  cout << endl;
	  }
	  cout << endl << endl;
	  
   }  // end of try block

   // catch failure caused by the H5File operations
   catch( FileIException error )
   {
      error.printError();
      return -1;
   }

   // catch failure caused by the DataSet operations
   catch( DataSetIException error )
   {
      error.printError();
      return -1;
   }

   //fmat myNph = hdf5readT<fmat>(FILE_NAME,DATASET_NAME);
   //cout << myNph << endl;
	
	cout << "Successful!" << endl;

  	return 0;
}

