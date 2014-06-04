#ifndef CIO_H
#define CIO_H
#include <armadillo>
#include <string>
#include <hdf5.h>

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
int hdf5write(std::string,std::string,arma::fmat);
int hdf5writeT(std::string,std::string,arma::fmat);

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

