#ifndef CIO_H
#define CIO_H
#include <armadillo>
#include <string>
#include <hdf5.h>

//#include <boost/tokenizer.hpp>
//#include <boost/algorithm/string.hpp>

#include <boost/filesystem.hpp>

#ifndef H5_NO_NAMESPACE
#ifndef H5_NO_STD
    using std::cout;
    using std::endl;
#endif  // H5_NO_STD
#endif

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif

#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

#include <typeinfo>

using namespace std;
using namespace arma;


template<typename T> int hdf5writeCube(std::string filename, std::string groupname, std::string subgroupname, std::string datasetname, T data, int createSubgroup){

	const H5std_string FILE_NAME( filename );
	const H5std_string DATASET_NAME( datasetname );
	
	int myRank;
	int DIM0, DIM1, DIM2;
	if (typeid(data) == typeid(cube)) {
		DIM0 = data.n_rows;
		DIM1 = data.n_cols;
		DIM2 = data.n_slices;
		myRank = 3;
	}

	// Try block to detect exceptions raised by any of the calls inside it
	try
	{
		// Turn off the auto-printing when failure occurs so that we can
		// handle the errors appropriately
		Exception::dontPrint();
	
		H5File file;
		// Check if file exists, if not create a new file
		if ( !boost::filesystem::exists( filename ) ) {
			//std::cout << "file does not exist" << std::endl;
			// Create a file.
			file = H5File( FILE_NAME, H5F_ACC_TRUNC );
		} else {
			//std::cout << "file does exist" << std::endl;
			file = H5File(FILE_NAME, H5F_ACC_RDWR);
		}

		// Check if group exists
		// Access the group.
		Group group;
		try { // to determine if the dataset exists in the group
			//cout << " Trying to open group" << endl;
			group = Group( file.openGroup( groupname ));
			//cout << " Opened existing group" << endl;
		}
		catch( FileIException not_found_error ) {
			//cout << " Group not found." << endl;
			// create group
			group = Group( file.createGroup( groupname ));
			//cout << " Group created." << endl;
		}

		if (createSubgroup) {
			group = Group( file.createGroup( subgroupname ));
			//cout << " Subgroup created." << endl;
		}

		if (!subgroupname.empty()){
			group = Group( file.openGroup( subgroupname ));
			//cout << " writing to subgroup." << endl;
		}
		

		  // WRITE CUBE

		  // Create the data space for the dataset.
		  if (typeid(data) == typeid(cube)) {
				hsize_t dims[myRank];              // dataset dimensions
				dims[0] = DIM0;
				dims[1] = DIM1;
				dims[2] = DIM2;
				DataSpace dataspace ( myRank, dims );
				PredType datatype( PredType::NATIVE_DOUBLE );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				double dataW[DIM0][DIM1][DIM2];
				for (int k = 0; k < DIM2; k++){
				for (int j = 0; j < DIM1; j++){
				for (int i = 0; i < DIM0; i++){
					dataW[i][j][k] = data.at(i,j,k);
				}
				}
				}
				dataset.write( dataW, PredType::NATIVE_DOUBLE );
			} else if (typeid(data) == typeid(fcube)) {
				hsize_t dims[myRank];              // dataset dimensions
				dims[0] = DIM0;
				dims[1] = DIM1;
				dims[2] = DIM2;
				DataSpace dataspace ( myRank, dims );
				PredType datatype( PredType::NATIVE_FLOAT );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				float dataW[DIM0][DIM1][DIM2];
				for (int k = 0; k < DIM2; k++){
				for (int j = 0; j < DIM1; j++){
				for (int i = 0; i < DIM0; i++){
					dataW[i][j][k] = data.at(i,j,k);
				}
				}
				}
				dataset.write( dataW, PredType::NATIVE_FLOAT );
			} else if (typeid(data) == typeid(icube)) {
				hsize_t dims[myRank];              // dataset dimensions
				dims[0] = DIM0;
				dims[1] = DIM1;
				dims[2] = DIM2;
				DataSpace dataspace ( myRank, dims );
				PredType datatype( PredType::NATIVE_INT );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				int dataW[DIM0][DIM1][DIM2];
				for (int k = 0; k < DIM2; k++){
				for (int j = 0; j < DIM1; j++){
				for (int i = 0; i < DIM0; i++){
					dataW[i][j][k] = data.at(i,j,k);
				}
				}
				}
				dataset.write( dataW, PredType::NATIVE_INT );
			} else if (typeid(data) == typeid(ucube)) {
				hsize_t dims[myRank];              // dataset dimensions
				dims[0] = DIM0;
				dims[1] = DIM1;
				dims[2] = DIM2;
				DataSpace dataspace ( myRank, dims );
				PredType datatype( PredType::NATIVE_UINT );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				unsigned int dataW[DIM0][DIM1][DIM2];
				for (int k = 0; k < DIM2; k++){
				for (int j = 0; j < DIM1; j++){
				for (int i = 0; i < DIM0; i++){
					dataW[i][j][k] = data.at(i,j,k);
				}
				}
				}
				dataset.write( dataW, PredType::NATIVE_UINT );
			}
			
	}
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
	// catch failure caused by the DataSpace operations
	catch( DataSpaceIException error )
	{
	error.printError();
	return -1;
	}
	return 0;
}

template<typename T> int hdf5writeVector(std::string filename, std::string groupname, std::string subgroupname, std::string datasetname, T data, int createSubgroup){

	const H5std_string FILE_NAME( filename );
	const H5std_string DATASET_NAME( datasetname );
	
	int myRank;
	int DIM0, DIM1, DIM2;
	if (typeid(data) == typeid(vec) || typeid(data) == typeid(fvec) || typeid(data) == typeid(ivec) || typeid(data) == typeid(uvec) || typeid(data) == typeid(rowvec) || typeid(data) == typeid(frowvec) || typeid(data) == typeid(irowvec) || typeid(data) == typeid(urowvec)) {
		DIM0 = data.n_elem;
		myRank = 1;
	} else if (typeid(data) == typeid(mat) || typeid(data) == typeid(fmat) || typeid(data) == typeid(imat) || typeid(data) == typeid(umat)) {
		DIM0 = data.n_rows;
		DIM1 = data.n_cols;
		myRank = 2;
	}

	// Try block to detect exceptions raised by any of the calls inside it
	try
	{
		// Turn off the auto-printing when failure occurs so that we can
		// handle the errors appropriately
		Exception::dontPrint();
	
		H5File file;
		// Check if file exists, if not create a new file
		if ( !boost::filesystem::exists( filename ) ) {
			//std::cout << "file does not exist" << std::endl;
			// Create a file.
			file = H5File( FILE_NAME, H5F_ACC_TRUNC );
		} else {
			//std::cout << "file does exist" << std::endl;
			file = H5File(FILE_NAME, H5F_ACC_RDWR);
		}

		// Check if group exists
		// Access the group.
		Group group;
		try { // to determine if the dataset exists in the group
			//cout << " Trying to open group" << endl;
			group = Group( file.openGroup( groupname ));
			//cout << " Opened existing group" << endl;
		}
		catch( FileIException not_found_error ) {
			//cout << " Group not found." << endl;
			// create group
			group = Group( file.createGroup( groupname ));
			//cout << " Group created." << endl;
		}

		if (createSubgroup) {
			group = Group( file.createGroup( subgroupname ));
			//cout << " Subgroup created." << endl;
		}

		if (!subgroupname.empty()){
			group = Group( file.openGroup( subgroupname ));
			//cout << " writing to subgroup." << endl;
		}
		

		  // WRITE VECTOR

		  // Create the data space for the dataset.
		  if (typeid(data) == typeid(vec) || typeid(data) == typeid(rowvec)) {
				hsize_t dims[myRank];              // dataset dimensions
				dims[0] = DIM0;
				DataSpace dataspace ( myRank, dims );
				PredType datatype( PredType::NATIVE_DOUBLE );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				double dataW[DIM0];
				for (int j = 0; j < DIM0; j++){
					//cout << data.at(j) << endl;
					dataW[j] = data.at(j);
				}
				dataset.write( dataW, PredType::NATIVE_DOUBLE );
			} else if (typeid(data) == typeid(fvec) || typeid(data) == typeid(frowvec)) {
				hsize_t dims[myRank];              // dataset dimensions
				dims[0] = DIM0;
				DataSpace dataspace ( myRank, dims );
				PredType datatype( PredType::NATIVE_FLOAT );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				float dataW[DIM0];
				for (int j = 0; j < DIM0; j++){
					//cout << data.at(j) << endl;
					dataW[j] = data.at(j);
				}
				dataset.write( dataW, PredType::NATIVE_FLOAT );
			} else if (typeid(data) == typeid(ivec) || typeid(data) == typeid(irowvec)) {
				hsize_t dims[myRank];              // dataset dimensions
				dims[0] = DIM0;
				DataSpace dataspace ( myRank, dims );
				PredType datatype( PredType::NATIVE_INT );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				int dataW[DIM0];
				for (int j = 0; j < DIM0; j++){
					//cout << data.at(j) << endl;
					dataW[j] = data.at(j);
				}
				dataset.write( dataW, PredType::NATIVE_INT );
			} else if (typeid(data) == typeid(uvec) || typeid(data) == typeid(urowvec)) {
				hsize_t dims[myRank];              // dataset dimensions
				dims[0] = DIM0;
				DataSpace dataspace ( myRank, dims );
				PredType datatype( PredType::NATIVE_UINT );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				unsigned int dataW[DIM0];
				for (int j = 0; j < DIM0; j++){
					//cout << data.at(j) << endl;
					dataW[j] = data.at(j);
				}
				dataset.write( dataW, PredType::NATIVE_UINT );
			} else if (typeid(data) == typeid(mat)) {
				hsize_t dims[myRank];              // dataset dimensions
				dims[0] = DIM0;
				dims[1] = DIM1;
				DataSpace dataspace ( myRank, dims );
				PredType datatype( PredType::NATIVE_DOUBLE );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				double dataW[DIM0][DIM1];
				for (int j = 0; j < DIM0; j++)
					for (int i = 0; i < DIM1; i++)
						dataW[j][i] = data.at(j,i);
				dataset.write( dataW, PredType::NATIVE_DOUBLE );
			} else if (typeid(data) == typeid(fmat)) {
			//cout << "Enter fmat" << endl;
				hsize_t dims[myRank];              // dataset dimensions
				dims[0] = DIM0;
				dims[1] = DIM1;
			//cout << myRank << " " << DIM0 << " " << DIM1 << endl;	
				DataSpace dataspace ( myRank, dims );
				PredType datatype( PredType::NATIVE_FLOAT );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				float dataW[DIM0][DIM1];
				for (int j = 0; j < DIM0; j++)
					for (int i = 0; i < DIM1; i++)
						dataW[j][i] = data.at(j,i);
				dataset.write( dataW, PredType::NATIVE_FLOAT );
			} else if (typeid(data) == typeid(imat)) {
				hsize_t dims[myRank];              // dataset dimensions
				dims[0] = DIM0;
				dims[1] = DIM1;
				DataSpace dataspace ( myRank, dims );
				PredType datatype( PredType::NATIVE_INT );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				int dataW[DIM0][DIM1];
				for (int j = 0; j < DIM0; j++)
					for (int i = 0; i < DIM1; i++)
						dataW[j][i] = data.at(j,i);
				dataset.write( dataW, PredType::NATIVE_INT );
			} else if (typeid(data) == typeid(umat)) {
				hsize_t dims[myRank];              // dataset dimensions
				dims[0] = DIM0;
				dims[1] = DIM1;
				DataSpace dataspace ( myRank, dims );
				PredType datatype( PredType::NATIVE_UINT );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				unsigned int dataW[DIM0][DIM1];
				for (int j = 0; j < DIM0; j++)
					for (int i = 0; i < DIM1; i++)
						dataW[j][i] = data.at(j,i);
				dataset.write( dataW, PredType::NATIVE_UINT );
			}
			
	}
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
	// catch failure caused by the DataSpace operations
	catch( DataSpaceIException error )
	{
	error.printError();
	return -1;
	}
	return 0;
}

template<typename T> int hdf5writeScalar(std::string filename, std::string groupname, std::string subgroupname, std::string datasetname, T data, int createSubgroup){

	const H5std_string FILE_NAME( filename );
	const H5std_string DATASET_NAME( datasetname );
	 
	// Try block to detect exceptions raised by any of the calls inside it
	try
	{
		// Turn off the auto-printing when failure occurs so that we can
		// handle the errors appropriately
		Exception::dontPrint();
	
		H5File file;
		// Check if file exists, if not create a new file
		if ( boost::filesystem::exists( filename ) ) {
			//std::cout << "file does exist" << std::endl;
			file = H5File(FILE_NAME, H5F_ACC_RDWR);
		} else {
			//std::cout << "file does not exist" << std::endl;
			file = H5File( FILE_NAME, H5F_ACC_TRUNC );
		}

		// Check if group exists
		// Access the group.
		Group group;
		try { // to determine if the dataset exists in the group
			//cout << " Trying to open group" << endl;
			group = Group( file.openGroup( groupname ));
			//cout << " Opened existing group" << endl;
		}
		catch( FileIException not_found_error ) {
			//cout << " Group not found." << endl;
			// create group
			group = Group( file.createGroup( groupname ));
			//cout << " Group created." << endl;
		}

		if (createSubgroup) {
			group = Group( file.createGroup( subgroupname ));
			//cout << "create subgroupname: " << subgroupname  << endl;
			//cout << " Subgroup created." << endl;
		}

		if (!subgroupname.empty()){
			group = Group( file.openGroup( subgroupname ));
			//cout << "open subgroupname: " << subgroupname << endl;
			//cout << " writing to subgroup." << endl;
		}

		  // WRITE SCALAR

		  // Create the data space for the dataset.
		  int myRank = 1;
		  hsize_t dims[myRank];              // dataset dimensions
		  dims[0] = 1;
		  DataSpace dataspace ( 0, dims );
		  // Data type
			if (typeid(data) == typeid(double)) {
				PredType datatype( PredType::NATIVE_DOUBLE );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				double dataW[1];
				dataW[0] = data;      
				dataset.write( dataW, PredType::NATIVE_DOUBLE );
			} else if (typeid(data) == typeid(float)) {
				PredType datatype( PredType::NATIVE_FLOAT );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				float dataW[1];
				dataW[0] = data;      
				dataset.write( dataW, PredType::NATIVE_FLOAT );
			} else if (typeid(data) == typeid(int)) {
				PredType datatype( PredType::NATIVE_INT );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				int dataW[1];
				dataW[0] = data;      
				dataset.write( dataW, PredType::NATIVE_INT );
			} else if (typeid(data) == typeid(uint)) {
				PredType datatype( PredType::NATIVE_UINT );
				DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
				dataset = file.openDataSet( DATASET_NAME );
				unsigned int dataW[1];
				dataW[0] = data;      
				dataset.write( dataW, PredType::NATIVE_UINT );
			}
	} // end of try block
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
	// catch failure caused by the DataSpace operations
	catch( DataSpaceIException error )
	{
	error.printError();
	return -1;
	}
	return 0;
}
/*
template<typename T> int hdf5writeT(std::string filename, std::string groupname, std::string subgroupname, std::string datasetname, T data, int createSubgroup){
	if (typeid(data) == typeid(double) || typeid(data) == typeid(float) || typeid(data) == typeid(int) || typeid(data) == typeid(uint)) {
		hdf5writeScalar(filename, groupname, subgroupname, datasetname, data, createSubgroup);
	} else if (typeid(data) == typeid(vec) || typeid(data) == typeid(fvec) || typeid(data) == typeid(ivec) || typeid(data) == typeid(uvec) || typeid(data) == typeid(rowvec) || typeid(data) == typeid(frowvec) || typeid(data) == typeid(irowvec) || typeid(data) == typeid(urowvec)) {
		hdf5writeVector(filename, groupname, subgroupname, datasetname, data, createSubgroup);
	} else if (typeid(data) == typeid(mat) || typeid(data) == typeid(fmat) || typeid(data) == typeid(imat) || typeid(data) == typeid(umat)) {
		hdf5writeVector(filename, groupname, subgroupname, datasetname, data, createSubgroup);
	} else if (typeid(data) == typeid(cube) || typeid(data) == typeid(fcube) || typeid(data) == typeid(icube) | typeid(data) == typeid(ucube)) {
		hdf5writeCube(filename, groupname, subgroupname, datasetname, data, createSubgroup);
	}
}
*/


template<typename T> T hdf5readScalar(std::string filename, std::string datasetname){
	const H5std_string FILE_NAME( filename );
	const H5std_string DATASET_NAME( datasetname );
	int NX, NY, NZ;
   
    T myData;
	
   // Try block to detect exceptions raised by any of the calls inside it
   try
   {
      // Turn off the auto-printing when failure occurs so that we can
      // handle the errors appropriately
		Exception::dontPrint();
	
	  // Open the specified file and the specified dataset in the file.
      	H5File file( FILE_NAME, H5F_ACC_RDONLY );
      	DataSet dataset = file.openDataSet( DATASET_NAME );

      // Get the class of the datatype that is used by the dataset.
      	H5T_class_t type_class = dataset.getTypeClass();

      // Get class of datatype and print message if it's a float.
      if( type_class == H5T_FLOAT ) {
	     //cout << "Data set has FLOAT type" << endl;

         // Get the integer datatype
	 	 FloatType intype = dataset.getFloatType();

         // Get order of datatype and print message if it's a little endian.
	 	 H5std_string order_string;
         H5T_order_t order = intype.getOrder( order_string );
	 	 //cout << "Endian:" << order << endl;

         // Get size of the data element stored in file and print it.
         size_t size = intype.getSize();
         //cout << "Data size is " << size << endl;
      } else if( type_class == H5T_INTEGER ) {
	 	//cout << "Data set has INTEGER type" << endl;
	 	 // Get the integer datatype
	 	 IntType intype = dataset.getIntType();

         // Get order of datatype and print message if it's a little endian.
	 	 H5std_string order_string;
         H5T_order_t order = intype.getOrder( order_string );
	 	 //cout << "Endian:" << order << endl;

         // Get size of the data element stored in file and print it.
         size_t size = intype.getSize();
         //cout << "Data size is " << size << endl;
	  }
      // Get dataspace of the dataset.
      DataSpace dataspace = dataset.getSpace();

      // Get the number of dimensions in the dataspace.
      int rank = dataspace.getSimpleExtentNdims();
	  //cout << "rank: " << rank << endl;

      // Get the dimension size of each dimension in the dataspace and
      // display them.
      hsize_t dims_out[rank];
      int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);

	  if (rank == 0 ) {
			if( type_class == H5T_NATIVE_DOUBLE ) {
				double data_out[1];
				data_out[0] = 0;
				dataset.read( data_out, PredType::NATIVE_DOUBLE );
				myData = data_out[0];
			} else if( type_class == H5T_NATIVE_FLOAT || type_class == H5T_FLOAT ) {
				float data_out[1];
				data_out[0] = 0;
				dataset.read( data_out, PredType::NATIVE_FLOAT );
				myData = data_out[0];
			} else if( type_class == H5T_NATIVE_INT || type_class == H5T_INTEGER ) {
				int data_out[1];
				data_out[0] = 0;
				dataset.read( data_out, PredType::NATIVE_INT );
				myData = data_out[0];
			} else if( type_class == H5T_NATIVE_UINT ) {
				unsigned int data_out[1];
				data_out[0] = 0;
				dataset.read( data_out, PredType::NATIVE_UINT );
				myData = data_out[0];
			}
		}

	return myData;
	}  // end of try block

   // catch failure caused by the H5File operations
   catch( FileIException error )
   {
      error.printError();
      //return -1;
   }

   // catch failure caused by the DataSet operations
   catch( DataSetIException error )
   {
      error.printError();
      //return -1;
   }

   // catch failure caused by the DataSpace operations
   catch( DataSpaceIException error )
   {
      error.printError();
      //return -1;
   }

   // catch failure caused by the DataSpace operations
   catch( DataTypeIException error )
   {
      error.printError();
      //return -1;
   }
}

template<typename T> T hdf5readVector(std::string filename, std::string datasetname){
	const H5std_string FILE_NAME( filename );
	const H5std_string DATASET_NAME( datasetname );
	int NX, NY, NZ;
   
    T myData;
	
   // Try block to detect exceptions raised by any of the calls inside it
   try
   {
      // Turn off the auto-printing when failure occurs so that we can
      // handle the errors appropriately
		Exception::dontPrint();
	
	  // Open the specified file and the specified dataset in the file.
      	H5File file( FILE_NAME, H5F_ACC_RDONLY );
      	DataSet dataset = file.openDataSet( DATASET_NAME );

      // Get the class of the datatype that is used by the dataset.
      	H5T_class_t type_class = dataset.getTypeClass();

//cout << "type: " << type_class << endl;

      // Get class of datatype and print message if it's a float.
      if( type_class == H5T_FLOAT ) {
	 	//cout << "Data set has FLOAT type" << endl;

         // Get the integer datatype
	 	 FloatType intype = dataset.getFloatType();

         // Get order of datatype and print message if it's a little endian.
	 	 H5std_string order_string;
         H5T_order_t order = intype.getOrder( order_string );
	 	 //cout << "Endian:" << order << endl;

         // Get size of the data element stored in file and print it.
         size_t size = intype.getSize();
         //cout << "Data size is " << size << endl;
      } else if( type_class == H5T_INTEGER ) {
	 	//cout << "Data set has INTEGER type" << endl;
	 	 // Get the integer datatype
	 	 IntType intype = dataset.getIntType();

         // Get order of datatype and print message if it's a little endian.
	 	 H5std_string order_string;
         H5T_order_t order = intype.getOrder( order_string );
	 	 //cout << "Endian:" << order << endl;

         // Get size of the data element stored in file and print it.
         size_t size = intype.getSize();
         //cout << "Data size is " << size << endl;
	  }
      // Get dataspace of the dataset.
      DataSpace dataspace = dataset.getSpace();

      // Get the number of dimensions in the dataspace.
      int rank = dataspace.getSimpleExtentNdims();

      // Get the dimension size of each dimension in the dataspace and
      // display them.
      hsize_t dims_out[rank];
      int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);

	  if (rank == 1) {
//cout << "rank: " << rank << endl;
//cout << "dimensions " << (unsigned long)(dims_out[0]) << endl;
			if( type_class == H5T_NATIVE_DOUBLE ) {
//cout << "Read doubles" << endl;			
				double data_out[dims_out[0]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
					data_out[i] = 0;
				}
				dataset.read( data_out, PredType::NATIVE_DOUBLE);
				myData(dims_out[0]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
					myData.at(i) = data_out[i];
				}
			} else if( type_class == H5T_NATIVE_FLOAT ) {
//cout << "Read floats" << endl;			
				float data_out[dims_out[0]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
					data_out[i] = 0;
				}
				dataset.read( data_out, PredType::NATIVE_FLOAT);
				myData(dims_out[0]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
					myData.at(i) = data_out[i];
				}
			} else if( type_class == H5T_NATIVE_INT ) {
//cout << "Read ints" << endl;
				int data_out[dims_out[0]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
					data_out[i] = 0;
				}
				dataset.read( data_out, PredType::NATIVE_INT);
				myData(dims_out[0]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
					myData.at(i) = data_out[i];
				}
			} else if( type_class == H5T_NATIVE_UINT ) {
//cout << "Read uints" << endl;			
				unsigned int data_out[dims_out[0]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
					data_out[i] = 0;
				}
				dataset.read( data_out, PredType::NATIVE_UINT);
				myData(dims_out[0]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
					myData.at(i) = data_out[i];
				}
			} else if( type_class == H5T_STD_I32LE ) {
//cout << "Read 32bit ints" << endl;			
			} else {
//cout << "Read no type" << endl;
				int data_out[dims_out[0]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
					data_out[i] = 0;
				}
				dataset.read( data_out, PredType::NATIVE_INT);
				myData(dims_out[0]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
					myData.at(i) = data_out[i];
				}


	
			}
		} else if (rank == 2) {
//cout << "rank: " << rank << endl;		
//cout << "dimensions " << (unsigned long)(dims_out[0]) << "x" << (unsigned long)(dims_out[1]) << endl;
			if( type_class == H5T_NATIVE_DOUBLE ) {
				double data_out[dims_out[0]][dims_out[1]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
					data_out[i][j] = 0;
				}
				}
				dataset.read( data_out, PredType::NATIVE_DOUBLE);
				myData(dims_out[0],dims_out[1]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
					myData.at(i,j) = data_out[i][j];
				}
				}
			} else if( type_class == H5T_NATIVE_FLOAT ) {
//cout << "Enter fmat" << endl;			
				float data_out[dims_out[0]][dims_out[1]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
					data_out[i][j] = 0;
				}
				}
				dataset.read( data_out, PredType::NATIVE_FLOAT);
				myData(dims_out[0],dims_out[1]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
					myData.at(i,j) = data_out[i][j];
				}
				}
			} else if( type_class == H5T_NATIVE_INT ) {
				int data_out[dims_out[0]][dims_out[1]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
					data_out[i][j] = 0;
				}
				}
				dataset.read( data_out, PredType::NATIVE_INT);
				myData(dims_out[0],dims_out[1]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
					myData.at(i,j) = data_out[i][j];
				}
				}
			} else if( type_class == H5T_NATIVE_UINT ) {
				unsigned int data_out[dims_out[0]][dims_out[1]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
					data_out[i][j] = 0;
				}
				}
				dataset.read( data_out, PredType::NATIVE_UINT);
				myData(dims_out[0],dims_out[1]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
					myData.at(i,j) = data_out[i][j];
				}
				}
			}
		}
	return myData;
	}  // end of try block

   // catch failure caused by the H5File operations
   catch( FileIException error )
   {
      error.printError();
      //return -1;
   }

   // catch failure caused by the DataSet operations
   catch( DataSetIException error )
   {
      error.printError();
      //return -1;
   }

   // catch failure caused by the DataSpace operations
   catch( DataSpaceIException error )
   {
      error.printError();
      //return -1;
   }

   // catch failure caused by the DataSpace operations
   catch( DataTypeIException error )
   {
      error.printError();
      //return -1;
   }
}

template<typename T> T hdf5readCube(std::string filename, std::string datasetname){
	const H5std_string FILE_NAME( filename );
	const H5std_string DATASET_NAME( datasetname );
	int NX, NY, NZ;
   
    T myData;
	
   // Try block to detect exceptions raised by any of the calls inside it
   try
   {
      // Turn off the auto-printing when failure occurs so that we can
      // handle the errors appropriately
		Exception::dontPrint();
	
	  // Open the specified file and the specified dataset in the file.
      	H5File file( FILE_NAME, H5F_ACC_RDONLY );
      	DataSet dataset = file.openDataSet( DATASET_NAME );

      // Get the class of the datatype that is used by the dataset.
      	H5T_class_t type_class = dataset.getTypeClass();

      // Get class of datatype and print message if it's a float.
      if( type_class == H5T_FLOAT ) {
	 	//cout << "Data set has FLOAT type" << endl;

         // Get the integer datatype
	 	 FloatType intype = dataset.getFloatType();

         // Get order of datatype and print message if it's a little endian.
	 	 H5std_string order_string;
         H5T_order_t order = intype.getOrder( order_string );
	 	 //cout << "Endian:" << order << endl;

         // Get size of the data element stored in file and print it.
         size_t size = intype.getSize();
         //cout << "Data size is " << size << endl;
      } else if( type_class == H5T_INTEGER ) {
	 	//cout << "Data set has INTEGER type" << endl;
	 	 // Get the integer datatype
	 	 IntType intype = dataset.getIntType();

         // Get order of datatype and print message if it's a little endian.
	 	 H5std_string order_string;
         H5T_order_t order = intype.getOrder( order_string );
	 	 //cout << "Endian:" << order << endl;

         // Get size of the data element stored in file and print it.
         size_t size = intype.getSize();
         //cout << "Data size is " << size << endl;
	  }
      // Get dataspace of the dataset.
      DataSpace dataspace = dataset.getSpace();

      // Get the number of dimensions in the dataspace.
      int rank = dataspace.getSimpleExtentNdims();

      // Get the dimension size of each dimension in the dataspace and
      // display them.
      hsize_t dims_out[rank];
      int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);

	  if (rank == 3) {
			//cout << "dimensions " << (unsigned long)(dims_out[0]) << "x" << (unsigned long)(dims_out[1]) << "x" << (unsigned long)(dims_out[2]) << endl;
			if( type_class == H5T_NATIVE_DOUBLE ) {
				double data_out[dims_out[0]][dims_out[1]][dims_out[2]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
				for (unsigned int k = 0; k < dims_out[2]; k++) {
					data_out[i][j][k] = 0;
				}
				}
				}
				dataset.read( data_out, PredType::NATIVE_DOUBLE);
				myData(dims_out[0],dims_out[1],dims_out[2]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
				for (unsigned int k = 0; k < dims_out[2]; k++) {
					myData.at(i,j,k) = data_out[i][j][k];
				}
				}
				}
			} else if( type_class == H5T_NATIVE_FLOAT ) {
				float data_out[dims_out[0]][dims_out[1]][dims_out[2]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
				for (unsigned int k = 0; k < dims_out[2]; k++) {
					data_out[i][j][k] = 0;
				}
				}
				}
				dataset.read( data_out, PredType::NATIVE_FLOAT);
				myData(dims_out[0],dims_out[1],dims_out[2]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
				for (unsigned int k = 0; k < dims_out[2]; k++) {
					myData.at(i,j,k) = data_out[i][j][k];
				}
				}
				}
			} else if( type_class == H5T_NATIVE_INT ) {
				int data_out[dims_out[0]][dims_out[1]][dims_out[2]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
				for (unsigned int k = 0; k < dims_out[2]; k++) {
					data_out[i][j][k] = 0;
				}
				}
				}
				dataset.read( data_out, PredType::NATIVE_INT);
				myData(dims_out[0],dims_out[1],dims_out[2]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
				for (unsigned int k = 0; k < dims_out[2]; k++) {
					myData.at(i,j,k) = data_out[i][j][k];
				}
				}
				}
			} else if( type_class == H5T_NATIVE_UINT ) {
				unsigned int data_out[dims_out[0]][dims_out[1]][dims_out[2]];
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
				for (unsigned int k = 0; k < dims_out[2]; k++) {
					data_out[i][j][k] = 0;
				}
				}
				}
				dataset.read( data_out, PredType::NATIVE_UINT);
				myData(dims_out[0],dims_out[1],dims_out[2]);
				for (unsigned int i = 0; i < dims_out[0]; i++) {
				for (unsigned int j = 0; j < dims_out[1]; j++) {
				for (unsigned int k = 0; k < dims_out[2]; k++) {
					myData.at(i,j,k) = data_out[i][j][k];
				}
				}
				}
			}
		}	
	return myData;
	}  // end of try block

   // catch failure caused by the H5File operations
   catch( FileIException error )
   {
      error.printError();
      //return -1;
   }

   // catch failure caused by the DataSet operations
   catch( DataSetIException error )
   {
      error.printError();
      //return -1;
   }

   // catch failure caused by the DataSpace operations
   catch( DataSpaceIException error )
   {
      error.printError();
      //return -1;
   }

   // catch failure caused by the DataSpace operations
   catch( DataTypeIException error )
   {
      error.printError();
      //return -1;
   }
}

template<typename T> T hdf5readT(std::string filename, std::string datasetname){
	const H5std_string FILE_NAME( filename );
	const H5std_string DATASET_NAME( datasetname );
	int NX, NY, NZ;
   
    T myData;

   /*
    * Try block to detect exceptions raised by any of the calls inside it
    */
   try
   {
      /*
       * Turn off the auto-printing when failure occurs so that we can
       * handle the errors appropriately
       */
		Exception::dontPrint();
	
	  /*
       * Open the specified file and the specified dataset in the file.
       */
      	H5File file( FILE_NAME, H5F_ACC_RDONLY );
      	DataSet dataset = file.openDataSet( DATASET_NAME );

      /*
       * Get the class of the datatype that is used by the dataset.
       */
      	H5T_class_t type_class = dataset.getTypeClass();

      /*
       * Get class of datatype and print message if it's a float.
       */
      if( type_class == H5T_FLOAT ) {// H5T_NATIVE_DOUBLE ) {//H5T_FLOAT ) {
	 	//cout << "Data set has FLOAT type" << endl;

         /*
	  	  * Get the integer datatype
          */
	 	 FloatType intype = dataset.getFloatType();

         /*
          * Get order of datatype and print message if it's a little endian.
          */
	 	 H5std_string order_string;
         H5T_order_t order = intype.getOrder( order_string );
	 	 //cout << "Endian:" << order << endl;

         /*
          * Get size of the data element stored in file and print it.
          */
         size_t size = intype.getSize();
         //cout << "Data size is " << size << endl;
      } else if( type_class == H5T_INTEGER ) {
	 	// cout << "Data set has INTEGER type" << endl;
	 	 /*
	  	  * Get the integer datatype
          */
	 	 IntType intype = dataset.getIntType();

         /*
          * Get order of datatype and print message if it's a little endian.
          */
	 	 H5std_string order_string;
         H5T_order_t order = intype.getOrder( order_string );
	 	 //cout << "Endian:" << order << endl;

         /*
          * Get size of the data element stored in file and print it.
          */
         size_t size = intype.getSize();
         //cout << "Data size is " << size << endl;
	  }
      /*
       * Get dataspace of the dataset.
       */
      DataSpace dataspace = dataset.getSpace();

      /*
       * Get the number of dimensions in the dataspace.
       */
      int rank = dataspace.getSimpleExtentNdims();
      //cout << "rank " << rank << endl;
      /*
       * Get the dimension size of each dimension in the dataspace and
       * display them.
       */
      hsize_t dims_out[rank];
      int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);
/*      
		if (rank == 1) {
			cout << "rank " << rank << ", dimensions " <<
				(unsigned long)(dims_out[0]) << endl;
		} else if (rank == 2) {
			cout << "rank " << rank << ", dimensions " <<
				(unsigned long)(dims_out[0]) << " x " <<
				(unsigned long)(dims_out[1]) << endl;
		} else if (rank == 3) {
			cout << "rank " << rank << ", dimensions " <<
				(unsigned long)(dims_out[0]) << " x " <<
				(unsigned long)(dims_out[1]) << " x " <<
				(unsigned long)(dims_out[2]) << endl;
		}
*/		
		if (rank == 1) {
			NX = (unsigned long)(dims_out[0]);
		} else if (rank == 2) {
			NX = (unsigned long)(dims_out[0]);
			NY = (unsigned long)(dims_out[1]);
		} else if (rank == 3) {
			NX = (unsigned long)(dims_out[0]);
			NY = (unsigned long)(dims_out[1]);
			NZ = (unsigned long)(dims_out[2]);
		}
		  
		/*
		* Define hyperslab in the dataset; implicitly giving strike and
		* block NULL.
		*/
		hsize_t      offset[rank];	// hyperslab offset in the file
		hsize_t      count[rank];	// size of the hyperslab in the file
		if (rank == 1) {
			offset[0] = 0;
			count[0]  = NX;
		} else if (rank == 2) {
			offset[0] = 0;
			offset[1] = 0;
			count[0]  = NX;
			count[1]  = NY;
		} else if (rank == 2) {
			offset[0] = 0;
			offset[1] = 0;
			offset[2] = 0;
			count[0]  = NX;
			count[1]  = NY;
			count[2]  = NZ;
		}
		dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );

		/*
		* Define the memory dataspace.
		*/
      hsize_t     dimsm[rank];              /* memory space dimensions */
		if (rank == 1) {
			dimsm[0] = NX;
		} else if (rank == 2) {
			dimsm[0] = NX;
      		dimsm[1] = NY;
		} else if (rank == 3) {
			dimsm[0] = NX;
      		dimsm[1] = NY;
      		dimsm[1] = NZ;
		}
      DataSpace memspace( rank, dimsm );
	  
      /*
       * Define memory hyperslab.
       */
      hsize_t      offset_out[rank];	// hyperslab offset in memory
      hsize_t      count_out[rank];	// size of the hyperslab in memory
		if (rank == 1) {
			offset_out[0] = 0;
      		count_out[0]  = NX;
		} else if (rank == 2) {
			offset_out[0] = 0;
      		offset_out[1] = 0;
      		//offset_out[2] = 0;
      		count_out[0]  = NX;
      		count_out[1]  = NY;
      		//count_out[2]  = 1;
		} else if (rank == 3) {
			offset_out[0] = 0;
      		offset_out[1] = 0;
      		offset_out[2] = 0;
      		count_out[0]  = NX;
      		count_out[1]  = NY;
      		count_out[2]  = NZ;
		}
      memspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out );
	  
      /*
       * Read data from hyperslab in the file into the hyperslab in
       * memory and display the data.
       */
		if (typeid(myData) == typeid(mat)) {
			//cout << "typeid = mat" << endl;
			double data_out[NX][NY]; /* output buffer */	
			dataset.read( data_out, PredType::NATIVE_DOUBLE, memspace, dataspace );
			myData.zeros(NX,NY); 
			for (int j = 0; j < NY; j++) {
			for (int i = 0; i < NX; i++) {
				myData(i,j) = data_out[i][j];  
			}
			}      
		} else if (typeid(myData) == typeid(fmat)) {
		//cout << "typeid = fmat" << endl;
			float data_out[NX][NY]; /* output buffer */	
			dataset.read( data_out, PredType::NATIVE_FLOAT, memspace, dataspace );
			myData.zeros(NX,NY); 
			for (int j = 0; j < NY; j++) {
			for (int i = 0; i < NX; i++) {
				myData(i,j) = data_out[i][j];  
			}
			}      
		} else if (typeid(myData) == typeid(imat)) {
			int data_out[NX][NY]; /* output buffer */	
			dataset.read( data_out, PredType::NATIVE_INT, memspace, dataspace );
			myData.zeros(NX,NY); 
			for (int j = 0; j < NY; j++) {
			for (int i = 0; i < NX; i++) {
				myData(i,j) = data_out[i][j];  
			}
			}      
		} else if (typeid(myData) == typeid(umat)) {
			unsigned int data_out[NX][NY]; /* output buffer */	
			dataset.read( data_out, PredType::NATIVE_UINT, memspace, dataspace );
			myData.zeros(NX,NY); 
			for (int j = 0; j < NY; j++) {
			for (int i = 0; i < NX; i++) {
				myData(i,j) = data_out[i][j];  
			}
			}      
		} else if (typeid(myData) == typeid(vec) || 
                   typeid(myData) == typeid(rowvec)) {
                   //cout << "typeid = vec | rowvec" << endl;
            if (rank == 2 && NY > NX) NX = NY;
			double data_out[NX]; /* output buffer */
			dataset.read( data_out, PredType::NATIVE_DOUBLE, memspace, dataspace );
			myData.zeros(NX); 
			for (int i = 0; i < NX; i++) {
				myData(i) = data_out[i];
			}
		} else if (typeid(myData) == typeid(fvec) || 
                   typeid(myData) == typeid(frowvec)) {
                   //cout << "typeid = fvec | frowvec" << endl;
            if (rank == 2 && NY > NX) NX = NY;
			float data_out[NX]; /* output buffer */	
			dataset.read( data_out, PredType::NATIVE_FLOAT, memspace, dataspace );
			myData.zeros(NX); 
			for (int i = 0; i < NX; i++) {
				myData(i) = data_out[i];  
			}
		} else if (typeid(myData) == typeid(ivec) || 
                   typeid(myData) == typeid(irowvec)) {
            if (rank == 2 && NY > NX) NX = NY;
			int data_out[NX]; /* output buffer */	
			dataset.read( data_out, PredType::NATIVE_INT, memspace, dataspace );
			myData.zeros(NX); 
			for (int i = 0; i < NX; i++) {
				myData(i) = data_out[i];  
			}
		} else if (typeid(myData) == typeid(uvec) || 
                   typeid(myData) == typeid(urowvec)) {
            if (rank == 2 && NY > NX) NX = NY;
			unsigned int data_out[NX]; /* output buffer */	
			dataset.read( data_out, PredType::NATIVE_UINT, memspace, dataspace );
			myData.zeros(NX);
			for (int i = 0; i < NX; i++) {
				myData(i) = data_out[i];  
			}
		}/* else if (typeid(myData) == typeid(cube)) {
			double data_out[NX][NY][NZ];
			dataset.read( data_out, PredType::NATIVE_DOUBLE, memspace, dataspace );
			myData.zeros(NX,NY,NZ);
			for (int k = 0; k < NZ; k++) {
			for (int j = 0; j < NY; j++) {
			for (int i = 0; i < NX; i++) {
				myData(i,j,k) = data_out[i][j][k];  
			}
			}
		} else if (typeid(myData) == typeid(fcube)) {
			float data_out[NX][NY][NZ];
			dataset.read( data_out, PredType::NATIVE_FLOAT, memspace, dataspace );
			myData.zeros(NX,NY,NZ);
			for (int k = 0; k < NZ; k++) {
			for (int j = 0; j < NY; j++) {
			for (int i = 0; i < NX; i++) {
				myData(i,j,k) = data_out[i][j][k];  
			}
			}
		} else if (typeid(myData) == typeid(icube)) {
			int data_out[NX][NY][NZ];
			dataset.read( data_out, PredType::NATIVE_INT, memspace, dataspace );
			myData.zeros(NX,NY,NZ);
			for (int k = 0; k < NZ; k++) {
			for (int j = 0; j < NY; j++) {
			for (int i = 0; i < NX; i++) {
				myData(i,j,k) = data_out[i][j][k];  
			}
			}
		} else if (typeid(myData) == typeid(ucube)) {
			unsigned int data_out[NX][NY][NZ];
			dataset.read( data_out, PredType::NATIVE_UINT, memspace, dataspace );
			myData.zeros(NX,NY,NZ);
			for (int k = 0; k < NZ; k++) {
			for (int j = 0; j < NY; j++) {
			for (int i = 0; i < NX; i++) {
				myData(i,j,k) = data_out[i][j][k];  
			}
			}
		} else if (typeid(myData) == typeid(double)) {
			double data_out[1];
			dataset.read( data_out, PredType::NATIVE_DOUBLE, memspace, dataspace );
			myData(0) = data_out[0];  
		} else if (typeid(myData) == typeid(float)) {
			float data_out[1];
			dataset.read( data_out, PredType::NATIVE_FLOAT, memspace, dataspace );
			myData(0) = data_out[0];
		} else if (typeid(myData) == typeid(int)) {
			int data_out[1];
			dataset.read( data_out, PredType::NATIVE_INT, memspace, dataspace ); 
			myData(0) = data_out[0];
		} else if (typeid(myData) == typeid(unsigned int)) {
			unsigned int data_out[1];
			dataset.read( data_out, PredType::NATIVE_UINT, memspace, dataspace );
			myData(0) = data_out[0];
		}*/
	return myData;
	}  // end of try block

   // catch failure caused by the H5File operations
   catch( FileIException error )
   {
      error.printError();
      //return -1;
   }

   // catch failure caused by the DataSet operations
   catch( DataSetIException error )
   {
      error.printError();
      //return -1;
   }

   // catch failure caused by the DataSpace operations
   catch( DataSpaceIException error )
   {
      error.printError();
      //return -1;
   }

   // catch failure caused by the DataSpace operations
   catch( DataTypeIException error )
   {
      error.printError();
      //return -1;
   }
}

#ifdef __cplusplus
extern "C" {
#endif

// Convert a string containing comma separated integers to ivec
arma::ivec str2ivec(std::string);
arma::fvec str2fvec(std::string);

arma::fmat hdf5read(std::string,std::string);
//int hdf5write(std::string,std::string,arma::fmat);
int hdf5write(std::string,std::string,arma::fmat);

arma::fmat load_asciiImage(std::string x);
arma::fvec load_asciiEuler(std::string x);
arma::fvec load_asciiQuaternion(std::string x);
arma::fmat load_asciiRotation(std::string x);

void load_constant(int);
void load_array(int*,int);
void load_array2D(int*,int,int);

void load_constantf(float);
void load_arrayf(float*,int);
void load_array2Df(float*,int,int);

void load_atomType(int*,int);
void load_atomPos(float*,int,int);
void load_xyzInd(int*,int);
void load_fftable(float*,int,int);
void load_qSample(float*,int);

struct Packet {
// particle
	int* atomType;
	float* atomPos;
	int* xyzInd;
	float* ffTable;
	float* qSample;
	int T;
	int N;
	int Q;
// detector
	unsigned* dp;
	double d;				// (m) detector distance
	double pix_width;		// (m)
	double pix_height;		// (m)
	int px;					// number of pixels in x
	int py;					// number of pixels in y
// beam
	double lambda; 							// (m) wavelength
	double focus;						// (m)
	double n_phot;							// number of photons per pulse
	unsigned freeElectrons;					// number of free electrons in the beam
// extra
	int finish;
	unsigned long seed;
	int useGPU;
};

void calculate_dp(Packet*);
int write_HDF5(char *);

arma::fmat load_readNthLine(std::string input, int N);

class CIO{
	//int b;
public:
	void get_image();
	double get_size();
};

#ifdef __cplusplus
}
#endif

#endif

