//#include <boost/python.hpp>
#include <iostream>
#include "io.h"
#include "particle.h"
#include "detector.h"
#include "beam.h"
#include "diffraction.h"

//#ifdef COMPILE_WITH_CUDA
#include "diffraction.cuh"
//#endif

#include "toolbox.h"

#include <hdf5.h>
//#include "H5Cpp.h"

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <armadillo>

using namespace std;
using namespace arma;
using namespace particle;
using namespace detector;
using namespace beam;
using namespace diffraction;
using namespace toolbox;

#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
#include <string>

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

int hdf5write(std::string filename, std::string datasetname, arma::fmat data){
	const H5std_string FILE_NAME( filename );
	const H5std_string DATASET_NAME( datasetname );
	
	int myDim1 = data.n_rows; // 6
	int myDim2 = data.n_cols; // 10
	
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
	/*
	* Create a file.
	*/
	H5File* file = new H5File( FILE_NAME, H5F_ACC_TRUNC );
	/*
	* Create property list for a dataset and set up fill values.
	*/
	int fillvalue = 0; /* Fill value for the dataset */
	DSetCreatPropList plist;
	plist.setFillValue(PredType::NATIVE_FLOAT, &fillvalue); // init with zeros
	/*
	* Create dataspace for the dataset in the file.
	*/
	hsize_t fdim[] = {FSPACE_DIM1, FSPACE_DIM2}; // dim sizes of ds (on disk)
	DataSpace fspace( FSPACE_RANK, fdim );
	/*
	* Create dataset and write it into the file.
	*/
	DataSet* dataset = new DataSet(file->createDataSet(
	DATASET_NAME, PredType::NATIVE_FLOAT, fspace, plist));
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
	float vector[MSPACE1_DIM]; // vector buffer for dset 50
	/*
	* Buffer initialization.
	*/
	int counter = 0;
	for (i = 0; i < myDim1 ; i++){
		for (j = 0; j < myDim2 ; j++){
			vector[counter] = data(i,j); // write out as is
			counter++;
		}
	}
//	for (i = 0; i < MSPACE1_DIM ; i++){ // written out column-wise
//		vector[i] = data[i];//i;
//	}

	dataset->write( vector, PredType::NATIVE_FLOAT, mspace1, fspace );
	/*
	* Reset the selection for the file dataspace fid.
	*/
	fspace.selectNone();
	/*
	* Close the dataset and the file.
	*/
	delete dataset;
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

arma::fmat hdf5read(std::string filename, std::string datasetname){
//template<typename T> T hdf5read(std::string filename, std::string datasetname){
	const H5std_string FILE_NAME( filename );
	const H5std_string DATASET_NAME( datasetname );
	//const int    NX_SUB = 3;	// hyperslab dimensions
	//const int    NY_SUB = 4;
	//const int    NX = 1456;		// output buffer dimensions
	//const int    NY = 1456;
	int NX, NY;
	const int    RANK_OUT = 2;
   
    fmat myDP;
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
      if( type_class == H5T_FLOAT )
      {
	 	//cout << "Data set has FLOAT type" << endl;

         /*
	  	  * Get the integer datatype
          */
	 	 IntType intype = dataset.getIntType();

         /*
          * Get order of datatype and print message if it's a little endian.
          */
	 	 H5std_string order_string;
         H5T_order_t order = intype.getOrder( order_string );
	 	 cout << "Endian:" << order << endl;

         /*
          * Get size of the data element stored in file and print it.
          */
         size_t size = intype.getSize();
         cout << "Data size is " << size << endl;
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
      hsize_t dims_out[2];
      int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);
      cout << "rank " << rank << ", dimensions " <<
	      (unsigned long)(dims_out[0]) << " x " <<
	      (unsigned long)(dims_out[1]) << endl;
	  
	  NX = (unsigned long)(dims_out[0]);
	  NY = (unsigned long)(dims_out[1]);
	  	  
      /*
       * Define hyperslab in the dataset; implicitly giving strike and
       * block NULL.
       */
      hsize_t      offset[2];	// hyperslab offset in the file
      hsize_t      count[2];	// size of the hyperslab in the file
      offset[0] = 0;
      offset[1] = 0;
      count[0]  = NX;//NX_SUB;
      count[1]  = NY;//NY_SUB;
      dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );

      /*
       * Define the memory dataspace.
       */
      hsize_t     dimsm[2];              /* memory space dimensions */
      dimsm[0] = NX;
      dimsm[1] = NY;
      //dimsm[2] = 1 ;
      DataSpace memspace( RANK_OUT, dimsm );
	  
      /*
       * Define memory hyperslab.
       */
      hsize_t      offset_out[2];	// hyperslab offset in memory
      hsize_t      count_out[2];	// size of the hyperslab in memory
      offset_out[0] = 0;
      offset_out[1] = 0;
      //offset_out[2] = 0;
      count_out[0]  = NX;
      count_out[1]  = NY;
      //count_out[2]  = 1;
      memspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out );
	  
      /*
       * Read data from hyperslab in the file into the hyperslab in
       * memory and display the data.
       */
      float data_out[NX][NY]; /* output buffer */	
      dataset.read( data_out, PredType::NATIVE_FLOAT, memspace, dataspace );

	  myDP.zeros(NY,NX); 
	  for (int j = 0; j < NY; j++)
   	  {
      	for (int i = 0; i < NX; i++)
      	{
	    	myDP(j,i) = data_out[j][i];
      	}
      }
	return myDP;
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

fmat load_asciiImage(string x){
	fmat B;
	bool status = B.load(x,raw_ascii);
	if(status == false){
		cout << "Error: problem with loading file, " << x << endl;
		exit(EXIT_FAILURE);
	}
	//cout << "myDP(0): " << B(0) << endl;
	//cout << "myDP(5): " << B(5) << endl;
	return B;
}

fvec load_asciiEuler(string x){
	fvec B;
	bool status = B.load(x,raw_ascii);
	if(status == false){
		cout << "Error: problem with loading file, " << x << endl;
		exit(EXIT_FAILURE);
	}
	return B;
}

fvec load_asciiQuaternion(string x){
	fvec quaternion;
	bool status = quaternion.load(x,raw_ascii);
	if(status == false){
		cout << "Error: problem with loading file, " << x << endl;
		exit(EXIT_FAILURE);
	}
	return quaternion;
}

void load_constant(int ang){
	cout << ang << endl;
}

void load_array(int *array, int size){
	for (int i = 0; i < size; i++) {
		cout << array[i] << endl;
	}
}

void load_array2D(int *array2D, int rows, int cols){
	for (int i = 0; i < rows; i++) {
	for (int j = 0; j < cols; j++) {
		cout << array2D[i*cols+j] << " ";
	}
	cout << endl;
	}
	irowvec dang(array2D, rows*cols);
	dang.print("read in as array1D: ");

	imat dung(array2D, cols, rows);
	dung = trans(dung);
	dung.print("read in as matrix: ");
}

void load_constantf(float ang){
	cout << ang << endl;
}

void load_arrayf(float *array, int size){
	for (int i = 0; i < size; i++) {
		cout << array[i] << endl;
	}
}

void load_array2Df(float *array2D, int rows, int cols){
	for (int i = 0; i < rows; i++) {
	for (int j = 0; j < cols; j++) {
		cout << array2D[i*cols+j] << " ";
	}
	cout << endl;
	}
	frowvec dang(array2D, rows*cols);
	dang.print("read in as array1D: ");

	fmat dung(array2D, cols, rows);
	dung = trans(dung);
	dung.print("read in as matrix: ");
}

void load_atomType(int *array,int size){
	for (int i = 0; i < size; i++) {
		cout << array[i] << endl;
	}
}

void load_atomPos(float *array2D,int rows,int cols){
	for (int i = 0; i < rows; i++) {
	for (int j = 0; j < cols; j++) {
		cout << array2D[i*cols+j] << " ";
	}
	cout << endl;
	}
	fmat dung(array2D, cols, rows);
	dung = trans(dung);
	dung.print("read in as matrix: ");
}

void load_xyzInd(int *array,int size){
	for (int i = 0; i < size; i++) {
		cout << array[i] << endl;
	}
}

void load_fftable(float *array2D,int rows,int cols){
	for (int i = 0; i < rows; i++) {
	for (int j = 0; j < cols; j++) {
		cout << array2D[i*cols+j] << " ";
	}
	cout << endl;
	}
	fmat dung(array2D, cols, rows);
	dung = trans(dung);
	dung.print("read in as matrix: ");
}

void load_qSample(float *array,int size){
	for (int i = 0; i < size; i++) {
		cout << array[i] << endl;
	}
}

void calculate_dp(Packet *pack){

	wall_clock timer;
	timer.tic();
	
	// particle
	CParticle particle = CParticle();
	//particle.set_param(pack);
	
	// Temporary code for testing
	particle.load_atomType("../atomType.dat"); 	// rowvec atomType
	particle.load_atomPos("../pos.dat");		// mat pos
	particle.load_xyzInd("../xyzInd.dat");		// rowvec xyzInd (temporary)
	particle.load_ffTable("../ffTable.dat");	// mat ffTable (atomType x qSample)
	particle.load_qSample("../qSample.dat");	// rowvec q vector sin(theta)/lambda
	/*
	CParticle::set_atomType(pack); // This is the actual code when reading from Zoltan's program
	CParticle::set_atomPos(pack);
	CParticle::set_xyzInd(pack);
	CParticle::set_ffTable(pack);
	CParticle::set_qSample(pack);
	*/
	// beam
	CBeam beam = CBeam();
	beam.set_param(pack);

	// detector
	CDetector det = CDetector();
	det.set_param(pack);
	det.init_dp(&beam);

	CDiffraction::calculate_atomicFactor(&particle,&det); // get f_hkl
#ifdef COMPILE_WITH_CUDA
	if(pack->useGPU) {
		cout << "start of io loop" << endl;
		int max_chunkSize = 500;
		int chunkSize = 0;

		fmat F_hkl_sq;
		F_hkl_sq.zeros(pack->py,pack->px); // F_hkl_sq: py x px

		fcube f_hkl = conv_to<fcube>::from(CDiffraction::f_hkl); // f_hkl: py x px x numAtomTypes
		float* f_mem = f_hkl.memptr();

		fcube q_xyz = conv_to<fcube>::from(det.q_xyz); // q_xyz: py x px x 3
		float* q_mem = q_xyz.memptr();

		irowvec xyzInd = conv_to<irowvec>::from(particle.xyzInd); // xyzInd: 1 x numAtom
		fmat pos = conv_to<fmat>::from(particle.atomPos); // pos: numAtom x 3

		fmat pad_real;
		fmat pad_imag;
		fmat sumDr;
		sumDr.zeros(pack->py*pack->px,1);
		fmat sumDi;
		sumDi.zeros(pack->py*pack->px,1);
			
		int first_ind = 0;
		int last_ind = 0;
		while (first_ind < particle.numAtoms) {		 
			last_ind = min((last_ind + max_chunkSize),particle.numAtoms);
			chunkSize = last_ind-first_ind;
			
			pad_real.zeros(pack->py*pack->px,chunkSize);
		 	float* pad_real_mem = pad_real.memptr();

			pad_imag.zeros(pack->py*pack->px,chunkSize);
		 	float* pad_imag_mem = pad_imag.memptr();
		 	
			irowvec xyzInd_sub = xyzInd.subvec( first_ind,last_ind-1 );
			int* i_mem = xyzInd_sub.memptr();	
			fmat pos_sub = pos( span(first_ind,last_ind-1), span::all );
			float* p_mem = pos_sub.memptr();

			//timer.tic();
			cuda_structureFactorChunkParallel(pad_real_mem, pad_imag_mem, f_mem, q_mem, i_mem, p_mem, particle.numAtomTypes, det.numPix, chunkSize);
			//cout<<"Chunk: Elapsed time is "<<timer.toc()<<" seconds."<<endl;

			sumDr += sum(pad_real,1);
			sumDi += sum(pad_imag,1);

			first_ind += max_chunkSize;
		}

		sumDr.reshape(pack->py,pack->px);
		sumDi.reshape(pack->py,pack->px);
		F_hkl_sq = sumDr % sumDr + sumDi % sumDi;

		//timer.tic();
		fmat detector_intensity;
		detector_intensity.copy_size(F_hkl_sq);
		detector_intensity = F_hkl_sq % det.solidAngle * as_scalar(det.thomson) * beam.get_photonsPerPulsePerArea();//2.105e30;//beam.phi_in;

		//timer.tic();
		stringstream ss;	//create a stringstream
    	ss << pack->finish;	//add number to the stream
    			
		string name = "../detector_counts_" + ss.str() + ".dat";
		umat detector_counts = CToolbox::convert_to_poisson(detector_intensity);
		detector_counts.save(name,raw_ascii);
		
		string name1 = "../dp_" + ss.str() + ".dat";
		det.dp.save(name1,raw_ascii);
	
		det.dp = det.dp + detector_counts;//detector_counts = det.dp + detector_counts;
	
		//cout<<"Calculate detector_counts: Elapsed time is "<<timer.toc()<<" seconds."<<endl; // 14.25s

		string name2 = "../F_hkl_sq_cudaChunk_" + ss.str() + ".dat";
		F_hkl_sq.save(name2,raw_ascii);

cout << "pack->finish: " << pack->finish << endl;

		if (pack->finish == 0) {
			det.solidAngle.save("../solidAngle.dat",raw_ascii);
			//timer.tic();
			det.dp.save("../diffraction.dat",raw_ascii);
			//det.dp.save("../diffraction.h5",hdf5_binary);
			//cout<<"Save image: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
		}
		cout << "End of F save" << endl;
		det.dp = trans(det.dp);//detector_counts = trans(detector_counts);
		pack->dp = det.dp.memptr();//pack->dp = detector_counts.memptr();
		cout << "pack->dp: " << pack->dp << endl;
		cout << "End of io loop: " << timer.toc() <<" seconds."<<endl;
	} 
#endif
	//else {
	if(!pack->useGPU) {
		timer.tic();
		CDiffraction::get_atomicFormFactorList(&particle,&det);
		cout<<"Make list: Elapsed time is "<<timer.toc()<<" seconds."<<endl; // 1.00s

		timer.tic();
		stringstream ss;	//create a stringstream
    	ss << pack->finish;	//add number to the stream
    
		fmat F_hkl_sq; // F_hkl_sq: py x px
		F_hkl_sq = CDiffraction::calculate_intensity(&particle,&det);
		fmat detector_intensity;
		detector_intensity.copy_size(F_hkl_sq);
		detector_intensity = F_hkl_sq % det.solidAngle * as_scalar(det.thomson) * beam.get_photonsPerPulsePerArea();//beam.phi_in;//2.105e30;

		string name = "../detector_counts_" + ss.str() + ".dat";
		umat detector_counts = CToolbox::convert_to_poisson(detector_intensity);
		detector_counts.save(name,raw_ascii);
	
		string name1 = "../dp_" + ss.str() + ".dat";
		det.dp.save(name1,raw_ascii);
	
		detector_counts = conv_to<umat>::from(det.dp) + detector_counts;
	
		cout<<"Calculate detector_counts: Elapsed time is "<<timer.toc()<<" seconds."<<endl; // 14.25s

		string name2 = "../F_hkl_sq_" + ss.str() + ".dat";
		F_hkl_sq.save(name2,raw_ascii);

		if (pack->finish == 0) {
			det.solidAngle.save("../solidAngle.dat",raw_ascii);
			timer.tic();
			det.dp.save("../diffraction.dat",raw_ascii);
			cout<<"Save image: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
		}
		detector_counts = trans(detector_counts);
		pack->dp = detector_counts.memptr();	
	}
}

void CIO::get_image(){
	cout << "Hello" << endl;
}

double CIO::get_size(){
	double size = 2.5;
	cout << "bye" << endl;
	return size;
}

/*
BOOST_PYTHON_MODULE(libio)
{
  using namespace boost::python;
  class_<CIO>("CIO").def("get_image", &CIO::get_image)
  					.def("get_size", &CIO::get_size);
}
*/
