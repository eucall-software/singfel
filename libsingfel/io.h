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

#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

#include <typeinfo>

using namespace std;
using namespace arma;

/*
#include <fstream>
bool fexists(const std::string filename) {
  ifstream ifile(filename.c_str());
  return ifile;
}*/

template<typename T> int hdf5writeT(std::string filename, std::string groupname, std::string subgroupname, std::string datasetname, T data, int appendDataset, int createSubgroup){
	const H5std_string FILE_NAME( filename );
	const H5std_string DATASET_NAME( datasetname );
	
	// TO DO: Use tokenizer to get groupname from datasetname
	
	int myDim1 = data.n_rows; // 6
	int myDim2 = data.n_cols; // 10

	// TO DO: Handle cubes and vectors
	
	const int MSPACE1_RANK = 1; // Rank of the first dataset in memory
	const int MSPACE1_DIM = myDim1*myDim2; // Dataset size in memory
	//const int MSPACE2_RANK = 1; // Rank of the second dataset in memory
	//const int MSPACE2_DIM = 4; // Dataset size in memory
	const int FSPACE_RANK = 2; // Dataset rank as it is stored in the file
	const int FSPACE_DIM1 = myDim1; // Dimension sizes of the dataset as it is
	const int FSPACE_DIM2 = myDim2; // stored in the file
	//const int MSPACE_RANK = 2; // Rank of the first dataset in memory
	//const int MSPACE_DIM1 = 8; // We will read dataset back from the file
	//const int MSPACE_DIM2 = 9; // to the dataset in memory with these
	// dataspace parameters
	//const int NPOINTS = 4; // Number of points that will be selected
	// and overwritten
	int i,j; // loop indices */
	 
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
	
	H5File* file;
	// Check if file exists, if not create a new file
	if(appendDataset) {
		//std::cout << "file does exist" << std::endl;
		file = new H5File(FILE_NAME, H5F_ACC_RDWR);
	} else {
		//std::cout << "file does not exist" << std::endl;
		// Create a file.
		file = new H5File( FILE_NAME, H5F_ACC_TRUNC );
	}
/*
	if ( !boost::filesystem::exists( filename ) ) {
		std::cout << "file does not exist" << std::endl;
		// Create a file.
		file = new H5File( FILE_NAME, H5F_ACC_TRUNC );
	} else {
		std::cout << "file does exist" << std::endl;
		file = new H5File(FILE_NAME, H5F_ACC_RDWR);
	}
*/

	// Check if group exists
	/*
	* Access the group.
	*/
	Group* group;
	try { // to determine if the dataset exists in the group
		//cout << " Trying to open group" << endl;
		group = new Group( file->openGroup( groupname ));
		//cout << " Opened existing group" << endl;
	}
	catch( FileIException not_found_error ) {
		//cout << " Group not found." << endl;
		// create group
		group = new Group( file->createGroup( groupname ));
		//cout << " Group created." << endl;
	}

	if (createSubgroup) {
		group = new Group( file->createGroup( subgroupname ));
		//cout << " Subgroup created." << endl;
	}

	if (!subgroupname.empty()){
		group = new Group( file->openGroup( subgroupname ));
		//cout << " writing to subgroup." << endl;
	}

	/*
	* Create property list for a dataset and set up fill values.
	*/
	int fillvalue = 0; /* Fill value for the dataset */
	DSetCreatPropList plist;
	if (typeid(data) == typeid(mat)) {
		plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue); // init with zeros
	} else if (typeid(data) == typeid(fmat)) {
    	plist.setFillValue(PredType::NATIVE_FLOAT, &fillvalue); // init with zeros
	} else if (typeid(data) == typeid(imat)) {
		plist.setFillValue(PredType::NATIVE_INT, &fillvalue); // init with zeros
	} else if (typeid(data) == typeid(umat)) {
		plist.setFillValue(PredType::NATIVE_UINT, &fillvalue); // init with zeros
	}
	/*
	* Create dataspace for the dataset in the file.
	*/
	hsize_t fdim[] = {FSPACE_DIM1, FSPACE_DIM2}; // dim sizes of ds (on disk)
	DataSpace fspace( FSPACE_RANK, fdim );
	/*
	* Create dataset and write it into the file.
	*/
	DataSet* dataset;
	if (typeid(data) == typeid(mat)) {
			dataset = new DataSet(file->createDataSet(
	DATASET_NAME, PredType::NATIVE_DOUBLE, fspace, plist));
	} else if (typeid(data) == typeid(fmat)) {
    		dataset = new DataSet(file->createDataSet(
	DATASET_NAME, PredType::NATIVE_FLOAT, fspace, plist));
	} else if (typeid(data) == typeid(imat)) {
			dataset = new DataSet(file->createDataSet(
	DATASET_NAME, PredType::NATIVE_INT, fspace, plist));
	} else if (typeid(data) == typeid(umat)) {
			dataset = new DataSet(file->createDataSet(
	DATASET_NAME, PredType::NATIVE_UINT, fspace, plist));
	}

	/*
	* Select hyperslab for the dataset in the file, using 3x2 blocks,
	* (4,3) stride and (2,4) count starting at the position (0,1).
	*/
	hsize_t start[2]; // Start of hyperslab
	hsize_t stride[2]; // Stride of hyperslab
	hsize_t count[2]; // Block count
	hsize_t block[2]; // Block sizes
	start[0] = 0; start[1] = 0;
	stride[0] = 1; stride[1] = 1;
	count[0] = myDim1; count[1] = myDim2; // how many unit cells (2x4)
	block[0] = 1; block[1] = 1; // unit cell size (3x2)
	fspace.selectHyperslab( H5S_SELECT_SET, count, start, stride, block);
	/*
	* Create dataspace for the first dataset.
	*/
	hsize_t dim1[] = {MSPACE1_DIM}; /* Dimension size of the first dataset 50
	(in memory) */
	DataSpace mspace1( MSPACE1_RANK, dim1 );
	/*
	* Select hyperslab.
	* We will use 48 elements of the vector buffer starting at the
	* second element. Selected elements are 1 2 3 . . . 48
	*/
	start[0] = 0;
	stride[0] = 1;
	count[0] = myDim1*myDim2;
	block[0] = 1;
	mspace1.selectHyperslab( H5S_SELECT_SET, count, start, stride, block);
	/*
	* Write selection from the vector buffer to the dataset in the file.
	*/
	
	int counter = 0;
	if (typeid(data) == typeid(mat)) {
		double vector[MSPACE1_DIM]; // vector buffer for dset 50
		for (i = 0; i < myDim1 ; i++){
			for (j = 0; j < myDim2 ; j++){
				vector[counter] = data(i,j); // write out as is
				counter++;
			}
		}
		dataset->write( vector, PredType::NATIVE_DOUBLE, mspace1, fspace );
	} else if (typeid(data) == typeid(fmat)) {
    	float vector[MSPACE1_DIM]; // vector buffer for dset 50
		for (i = 0; i < myDim1 ; i++){
			for (j = 0; j < myDim2 ; j++){
				vector[counter] = data(i,j); // write out as is
				counter++;
			}
		}
		dataset->write( vector, PredType::NATIVE_FLOAT, mspace1, fspace );
	} else if (typeid(data) == typeid(imat)) {
		int vector[MSPACE1_DIM]; // vector buffer for dset 50
		for (i = 0; i < myDim1 ; i++){
			for (j = 0; j < myDim2 ; j++){
				vector[counter] = data(i,j); // write out as is
				counter++;
			}
		}
		dataset->write( vector, PredType::NATIVE_INT, mspace1, fspace );
	} else if (typeid(data) == typeid(umat)) {
		unsigned int vector[MSPACE1_DIM]; // vector buffer for dset 50
		for (i = 0; i < myDim1 ; i++){
			for (j = 0; j < myDim2 ; j++){
				vector[counter] = data(i,j); // write out as is
				counter++;
			}
		}
		dataset->write( vector, PredType::NATIVE_UINT, mspace1, fspace );
	}
	
	/*
	* Reset the selection for the file dataspace fid.
	*/
	fspace.selectNone();
	/*
	* Close the dataset and the file.
	*/
	delete dataset;
	delete group;
	delete file;
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
      if( type_class == H5T_FLOAT ) {
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
	 	//cout << "Data set has INTEGER type" << endl;
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
			double data_out[NX][NY]; /* output buffer */	
			dataset.read( data_out, PredType::NATIVE_DOUBLE, memspace, dataspace );
			myData.zeros(NX,NY); 
			for (int j = 0; j < NY; j++) {
			for (int i = 0; i < NX; i++) {
				myData(i,j) = data_out[i][j];  
			}
			}      
		} else if (typeid(myData) == typeid(fmat)) {
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
			double data_out[NX]; /* output buffer */	
			dataset.read( data_out, PredType::NATIVE_DOUBLE, memspace, dataspace );
			myData.zeros(NX); 
			for (int i = 0; i < NX; i++) {
				myData(i) = data_out[i];
			}
		} else if (typeid(myData) == typeid(fvec) || 
                   typeid(myData) == typeid(frowvec)) {
			float data_out[NX]; /* output buffer */	
			dataset.read( data_out, PredType::NATIVE_FLOAT, memspace, dataspace );
			myData.zeros(NX); 
			for (int i = 0; i < NX; i++) {
				myData(i) = data_out[i];  
			}
		} else if (typeid(myData) == typeid(ivec) || 
                   typeid(myData) == typeid(irowvec)) {
			int data_out[NX]; /* output buffer */	
			dataset.read( data_out, PredType::NATIVE_INT, memspace, dataspace );
			myData.zeros(NX); 
			for (int i = 0; i < NX; i++) {
				myData(i) = data_out[i];  
			}
		} else if (typeid(myData) == typeid(uvec) || 
                   typeid(myData) == typeid(urowvec)) {
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


arma::fmat hdf5read(std::string,std::string);
//int hdf5write(std::string,std::string,arma::fmat);
int hdf5write(std::string,std::string,arma::fmat);

arma::fmat load_asciiImage(std::string x);
arma::fvec load_asciiEuler(std::string x);
arma::fvec load_asciiQuaternion(std::string x);

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

