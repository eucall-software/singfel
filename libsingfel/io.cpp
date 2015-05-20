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

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

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

// Convert a string containing comma separated integers to ivec
ivec str2ivec(std::string line){
	int numElements = 100;
	ivec myVec(numElements); // number of elements unknown
	typedef boost::tokenizer<boost::char_separator<char> > Tok;
	boost::char_separator<char> sep(","); // default constructed
	Tok tok(line, sep);
	int counter = 0;
	for(Tok::iterator tok_iter = tok.begin(); tok_iter != tok.end(); ++tok_iter){
		if (counter == numElements) {
			cout << "Too many elements in the string" << endl;
			exit(0);
		}
		string temp = *tok_iter;
		myVec[counter] = atoi(temp.c_str());
		counter++;
	}
	ivec cleanVec = myVec.subvec(0,counter-1);
	return cleanVec;
}

// Convert a string containing comma separated integers to ivec
fvec str2fvec(std::string line){
	int numElements = 100;
	fvec myVec(numElements); // number of elements unknown
	typedef boost::tokenizer<boost::char_separator<char> > Tok;
	boost::char_separator<char> sep(","); // default constructed
	Tok tok(line, sep);
	int counter = 0;
	for(Tok::iterator tok_iter = tok.begin(); tok_iter != tok.end(); ++tok_iter){
		if (counter == numElements) {
			cout << "Too many elements in the string" << endl;
			exit(0);
		}
		string temp = *tok_iter;
		myVec[counter] = atof(temp.c_str());
		counter++;
	}
	fvec cleanVec = myVec.subvec(0,counter-1);
	return cleanVec;
}

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

fmat load_asciiImage(string x){
	fmat B;
	bool status = B.load(x,raw_ascii);
	if(status == false){
		cout << "Error: problem with loading file, " << x << endl;
		exit(EXIT_FAILURE);
	}
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

fmat load_asciiRotation(string x){
	fmat rotation;
	bool status = rotation.load(x,raw_ascii);
	if(status == false){
		cout << "Error: problem with loading file, " << x << endl;
		exit(EXIT_FAILURE);
	}
	return rotation;
}

void load_constant(int ang){
	cout << ang << endl;
}

void load_array(int *array, int size){
	for (int i = 0; i < size; i++) {
		cout << array[i] << endl;
	}
}
/*
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
*/
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

/*
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
	
	//CParticle::set_atomType(pack); // This is the actual code when reading from Zoltan's program
	//CParticle::set_atomPos(pack);
	//CParticle::set_xyzInd(pack);
	//CParticle::set_ffTable(pack);
	//CParticle::set_qSample(pack);
	
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
		umat detector_counts = CToolbox::convert_to_poisson(&detector_intensity);
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
		//pack->dp = det.dp.memptr();
		pack->dp = detector_counts.memptr();
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
		umat detector_counts = CToolbox::convert_to_poisson(&detector_intensity);
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
*/

void CIO::get_image(){
	cout << "Hello" << endl;
}

double CIO::get_size(){
	double size = 2.5;
	cout << "bye" << endl;
	return size;
}


int prepS2E(const char* filename,const char* outputName,const char* configFile){
	hid_t file_in,file_out;
	hid_t grp_hist,grp_hist_parent,grp_hist_parent_detail,grp_tmp; 
	hid_t attr,dataset;
	hid_t str_type;
	hid_t dataspace;
	hid_t ocpypl_id;
	hid_t props;
	int parent_name_size;
	hsize_t rank,dimens_1d[1];
	char parent_name[PATH_MAX];
	char filename_full[PATH_MAX];
	char outputNameTmp[PATH_MAX];
	int tmpsa,tmpsb;
	sprintf(outputNameTmp,"%s~",outputName);
	file_in  = H5Fopen  (filename,   H5F_ACC_RDONLY, H5P_DEFAULT);
	file_out = H5Fcreate(outputNameTmp, H5F_ACC_TRUNC,  H5P_DEFAULT, H5P_DEFAULT);
	
	grp_hist = H5Gcreate (file_out, "history", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    //H5Gclose(grp_hist);
	ocpypl_id = H5Pcreate(H5P_OBJECT_COPY);
	H5Pset_copy_object(ocpypl_id, H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG);
	H5Ocopy(file_in,"history/parent",file_out,"history/parent",ocpypl_id,H5P_DEFAULT);
	    
    grp_hist_parent = H5Gopen(file_out,"history/parent",H5P_DEFAULT);
	if (grp_hist_parent<0)
		grp_hist_parent = H5Gcreate (file_out, "history/parent", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	grp_hist_parent_detail = H5Gopen(file_out,"history/parent/detail",H5P_DEFAULT);
	if (grp_hist_parent_detail<0)
		grp_hist_parent_detail = H5Gcreate (file_out, "history/parent/detail", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
	
	if (NULL==realpath(filename,filename_full))
		return -1;
	for(tmpsb=strlen(filename_full)-1;tmpsb>=0 && filename_full[tmpsb]!='/';tmpsb--);
	for(tmpsa=tmpsb-1;tmpsa>=0 && filename_full[tmpsa]!='/';tmpsa--);

	if (tmpsb==0){
		tmpsa=0;
		tmpsb=1;
	}else
		tmpsa++;

	//COPY PARENT NAME TO grp_hist_parent[name]
	parent_name_size=tmpsb-tmpsa+1;
	parent_name[0]='_';
	memcpy(parent_name+1,filename_full+tmpsa,parent_name_size-1);
	parent_name[parent_name_size]='\0';
	
	//!!H5Tset_size (str_type, H5T_VARIABLE);
	if (H5Aexists(grp_hist_parent,"name"))
		H5Adelete(grp_hist_parent,"name");
		
	dataspace = H5Screate(H5S_SCALAR);
	str_type = H5Tcopy (H5T_C_S1);
	H5Tset_size (str_type, parent_name_size);
	attr = H5Acreate (grp_hist_parent, "name", str_type, dataspace,H5P_DEFAULT,H5P_DEFAULT);
	H5Awrite (attr, str_type, parent_name);
	H5Sclose (dataspace);
	H5Aclose (attr);
	H5Giterate(file_in, "/", NULL, file_info, &file_out );
	
	if (H5Lexists( grp_hist_parent_detail, "data", H5P_DEFAULT))
		H5Ldelete( grp_hist_parent_detail, "data", H5P_DEFAULT); 
	H5Lcreate_external( filename_full, "data", grp_hist_parent_detail, "data", H5P_DEFAULT, H5P_DEFAULT );
	
	str_type = H5Tcopy (H5T_C_S1);	
	rank = 1;
	dimens_1d[0] = 1;
	props = H5Pcreate (H5P_DATASET_CREATE);
	
	grp_tmp = H5Gcreate (file_out, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Gclose(grp_tmp);		
	grp_tmp = H5Gcreate (file_out, "params", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	char *config;
	FILE *cfgf=NULL;
	cfgf=fopen(configFile,"r");
	if (cfgf==NULL){
		printf("[E]File not found %s\n",configFile);
		return -1;
	}
	fseek(cfgf, 0L, SEEK_END);
	size_t cfgsz;
	cfgsz=ftell(cfgf);
	fseek(cfgf, 0L, SEEK_SET);
	config=(char*)malloc(cfgsz);
	fread(config,1,cfgsz,cfgf);
	fclose(cfgf);
	H5Tset_size (str_type, strlen(config)+1);
	dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	dataset = H5Dcreate(grp_tmp,"info",str_type,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	H5Dwrite(dataset, str_type, H5S_ALL,H5S_ALL,H5P_DEFAULT,config);
	H5Sclose (dataspace);
	H5Gclose(grp_tmp);
	delete(config);
	grp_tmp = H5Gcreate (file_out, "misc", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Gclose(grp_tmp);
	grp_tmp = H5Gcreate (file_out, "info", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	//make info
	string text="SingFEL v0.1.0";
	H5Tset_size (str_type, text.size()+1);
	dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	dataset = H5Dcreate(grp_tmp,"package_version",str_type,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	H5Dwrite(dataset, str_type, H5S_ALL,H5S_ALL,H5P_DEFAULT,text.c_str());
	H5Sclose (dataspace);
	text="This dataset contains a diffraction pattern generated using SingFEL.";
	H5Tset_size (str_type, text.size()+1);
	dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	dataset = H5Dcreate(grp_tmp,"data_description",str_type,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	H5Dwrite(dataset, str_type, H5S_ALL,H5S_ALL,H5P_DEFAULT,text.c_str());
	H5Sclose (dataspace);
	text="Form factors of the radiation damaged molecules are calculated in time slices. At each time slice, the coherent scattering is calculated and incoherently added to the final diffraction pattern. Finally, Poissonian noise is added to the diffraction pattern.";
	H5Tset_size (str_type, text.size()+1);
	dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	dataset = H5Dcreate(grp_tmp,"method_description",str_type,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	H5Dwrite(dataset, str_type, H5S_ALL,H5S_ALL,H5P_DEFAULT,text.c_str());
	H5Sclose (dataspace);
	
	char* strcontact[2];
	string text1="Name: Chunhong Yoon";
	strcontact[0]=(char*)text1.c_str();
	string text2="Email: chun.hong.yoon@desy.de";
	strcontact[1]=(char*)text2.c_str();
	H5Tset_size (str_type, H5T_VARIABLE);
	dimens_1d[0] = 2;
	
	dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	dataset = H5Dcreate(grp_tmp,"contact",str_type,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	H5Dwrite(dataset, str_type, H5S_ALL,H5S_ALL,H5P_DEFAULT,strcontact);
	H5Sclose (dataspace);
	H5Gclose(grp_tmp);
	
	dimens_1d[0] = 1;
	float ver=0.1;
	dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	dataset = H5Dcreate(file_out,"version",H5T_IEEE_F32LE,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	H5Dwrite(dataset, H5T_IEEE_F32LE, H5S_ALL,H5S_ALL,H5P_DEFAULT,&ver);
	H5Sclose (dataspace);
	
	H5Gclose(grp_hist);
	H5Gclose(grp_hist_parent);
	H5Gclose(grp_hist_parent_detail);
	H5Fclose(file_in);
	H5Fclose(file_out);
	
	//clean up : copy only not un-linked data from tmp to out
	file_in  = H5Fopen  (outputNameTmp, H5F_ACC_RDONLY, H5P_DEFAULT);
	file_out = H5Fcreate(outputName,    H5F_ACC_TRUNC,  H5P_DEFAULT, H5P_DEFAULT);
	H5Giterate(file_in, "/", NULL, file_copy, &file_out);
	H5Fclose(file_in);
	H5Fclose(file_out);
	//delete tmp file
	remove(outputNameTmp);
	
	return 0 && configFile==NULL;
}

herr_t file_info(hid_t loc_id, const char *name, void *opdata)
{
	hid_t ocpypl_id,prev_grp,prev_ds;
	if (0 == strncmp(name, "data", 5) || 0 == strncmp(name, "history", 8))
		return 0;
    H5G_stat_t statbuf;
    H5Gget_objinfo(loc_id, name, false, &statbuf);
    char* dststr;
    dststr=(char*)malloc(strlen(name)+2+strlen("history/parent/detail/"));
    sprintf(dststr,"history/parent/detail/%s",name);
    switch (statbuf.type) {
    case H5G_GROUP: 
		prev_grp = H5Gopen(*(hid_t*)opdata,dststr,H5P_DEFAULT);
		if (prev_grp>=0){
			H5Gclose(prev_grp);
			H5Ldelete (*(hid_t*)opdata,dststr,H5P_DEFAULT);
		}
		ocpypl_id = H5Pcreate(H5P_OBJECT_COPY);
		H5Pset_copy_object(ocpypl_id, H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG);
		H5Ocopy(loc_id,name,*(hid_t*)opdata,dststr,ocpypl_id,H5P_DEFAULT);
		break;
    case H5G_DATASET: 
		prev_ds = H5Dopen(*(hid_t*)opdata,dststr,H5P_DEFAULT);
		if (prev_ds>=0){
			H5Dclose(prev_ds);
			H5Ldelete (*(hid_t*)opdata,dststr,H5P_DEFAULT);
		}
		ocpypl_id = H5Pcreate(H5P_OBJECT_COPY);
		H5Pset_copy_object(ocpypl_id, H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG);
		H5Ocopy(loc_id,name,*(hid_t*)opdata,dststr,ocpypl_id,H5P_DEFAULT);
		break;
    default:
		;
    }
    delete(dststr); 
    return 0;
}	

herr_t file_copy(hid_t loc_id, const char *name, void *opdata)
{
	hid_t ocpypl_id;
	ocpypl_id = H5Pcreate(H5P_OBJECT_COPY);
	H5Pset_copy_object(ocpypl_id, H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG);
	H5Ocopy(loc_id,name,*(hid_t*)opdata,name,ocpypl_id,H5P_DEFAULT);
    return 0;
}	





/*
BOOST_PYTHON_MODULE(libio)
{
  using namespace boost::python;
  class_<CIO>("CIO").def("get_image", &CIO::get_image)
  					.def("get_size", &CIO::get_size);
}
*/
