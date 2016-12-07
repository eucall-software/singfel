/*
 * Program for simulating diffraction patterns
 */
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <algorithm>
#include <fstream>
#include <string>
//SHM for local rank
#include <sys/ipc.h>
#include <sys/stat.h>
#include <sys/shm.h>
// Armadillo library
#include <armadillo>
// Boost library
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/program_options.hpp>
// HDF5 library
#include "hdf5.h"
#include "hdf5_hl.h"
// SingFEL library
#include "detector.h"
#include "beam.h"
#include "particle.h"
#include "diffraction.h"
#include "toolbox.h"
#include "io.h"

#ifdef COMPILE_WITH_CUDA
#include "diffraction.cuh"
#endif

namespace mpi = boost::mpi;
namespace opt = boost::program_options;
using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

#define QTAG 1	// quaternion
#define DPTAG 2	// diffraction pattern
#define DIETAG 3 // die signal
#define DONETAG 4 // done signal

#define MPI_SHMKEY 0x6FB1407
//#define MPI_SHMKEY 0x6FB10407

const int master = 0; // Process with rank=0 is the master
const int msgLength = 4; // MPI message length
#ifdef COMPILE_WITH_CUDA
int localRank=0;
int deviceCount=cuda_getDeviceCount();
#endif

static void master_diffract(mpi::communicator* comm, opt::variables_map vm);
static void slave_diffract(mpi::communicator* comm, opt::variables_map vm);

opt::variables_map parse_input(int argc, char* argv[], mpi::communicator* comm);
void make1Diffr(const fmat& myQuaternions,int counter,opt::variables_map vm, string outputName);

void generateRotations(const bool uniformRotation, \
                       const string rotationAxis, const int numQuaternions, \
                       fmat* myQuaternions);
void loadParticle(const opt::variables_map vm, const string filename, \
                  const int timeSlice, CParticle* particle);
void setTimeSliceInterval(const int numSlices, int* sliceInterval, \
                          int* timeSlice, int* done);
void rotateParticle(fvec* quaternion, CParticle* particle);
void setFluenceFromFile(const string filename, const int timeSlice, \
                        const int sliceInterval, CBeam* beam);
void setEnergyFromFile(const string filename, CBeam* beam);
void setFocusFromFile(const string filename, CBeam* beam);
void getComptonScattering(const opt::variables_map vm, CParticle* particle, \
                          CDetector* det, fmat* Compton);
void savePhotonField(const string filename, const int isFirstSlice, \
                     const int timeSlice, fmat* photon_field);
void saveAsDiffrOutFile(const string outputName, const string inputName, int counter, umat* detector_counts, \
                        fmat* detector_intensity, fvec* quaternion, \
                        CDetector* det, CBeam* beam, double total_phot);

int prepH5(opt::variables_map vm, string outputName);
int linkHistory( hid_t, const string, const string);

int main( int argc, char* argv[] ){

	// Initialize MPI
  	mpi::environment env;
  	mpi::communicator world;
	mpi::communicator* comm = &world;

	// All processes parse the input
	opt::variables_map vm = parse_input(argc, argv, comm);
	// Set random seed
	//srand( vm["pmiStartID"].as<int>() + world.rank() + (unsigned)time(NULL) );
#ifdef COMPILE_WITH_CUDA

	srand( 0x01333337);
	int shmid;
	key_t shmkey = (key_t)MPI_SHMKEY;
	static int *shmval;
	if (world.rank() != master){
		shmid = shmget(shmkey,sizeof(int),0666 | IPC_CREAT);
		if ( 0 > shmid )
			perror("shmget");
		shmval = (int*)shmat(shmid,NULL,0);
		if ( 0 > shmval )
			perror("shmval");
		*shmval=0;
	}

	world.barrier();
	if (world.rank() != master){
		//printf("Local %d\n",localRank);fflush(NULL);
		localRank = __sync_fetch_and_add( shmval,1);
		//printf("Local %d\n",localRank);fflush(NULL);
	}

#endif
	wall_clock timerMaster;

	timerMaster.tic();

	// Main program
	if (world.rank() == master) {
		master_diffract(comm, vm);
	} else {
		slave_diffract(comm, vm);
	}

#ifdef COMPILE_WITH_CUDA

	world.barrier();
	if (world.rank() != master) {
		shmdt(shmval);
		shmctl(shmid, IPC_RMID, 0);
	}
#endif

	world.barrier();
	if (world.rank() == master) {
		cout << "Finished: " << timerMaster.toc() <<" seconds."<<endl;
	}

  	return 0;
}

static void master_diffract(mpi::communicator* comm, opt::variables_map vm) {

	int pmiStartID = vm["pmiStartID"].as<int>();
	int pmiEndID = vm["pmiEndID"].as<int>();
	int numDP = vm["numDP"].as<int>();

	int numProcesses = comm->size();

  	int ntasks = (pmiEndID-pmiStartID+1)*numDP;

	fmat myQuaternions;
	if (numProcesses==1)
	{
		string rotationAxis = vm["rotationAxis"].as<string>();
		bool uniformRotation = vm["uniformRotation"].as<bool>();
		generateRotations(uniformRotation, rotationAxis, ntasks, \
	                  &myQuaternions);
	}

	for (int ntask = 0; ntask < ntasks; ntask++) {
		if (numProcesses==1)
		{
            string outputName = "diffr_out_0000001.h5";
            int success = prepH5(vm, outputName);
            assert(success == 0);
			make1Diffr(myQuaternions,ntask,vm, outputName);
		}
		else
		{
			int tmp;
		  	boost::mpi::status status = comm->recv(boost::mpi::any_source, 0, tmp);
			comm->send(status.source(), 0, &ntask, 1);
		}
	}

// final send
	int ntask=-1;
	for (int np = 0; np < numProcesses; np++) {
		if (np!= master)
			{
				comm->send(np, 0, &ntask, 1);
			}
	}

}


static void slave_diffract(mpi::communicator* comm, opt::variables_map vm) {

	string rotationAxis = vm["rotationAxis"].as<string>();
	bool uniformRotation = vm["uniformRotation"].as<bool>();


    // Get required input arguments.
	int pmiStartID = vm["pmiStartID"].as<int>();
	int pmiEndID = vm["pmiEndID"].as<int>();
	int numDP = vm["numDP"].as<int>();
	string inputDir = vm["inputDir"].as<std::string>();
	string outputDir = vm["outputDir"].as<string>();

	int ntasks = (pmiEndID-pmiStartID+1)*numDP;

	// Setup rotations
	fmat myQuaternions;
	generateRotations(uniformRotation, rotationAxis, ntasks, \
	                  &myQuaternions);

    // Init a local counter.
	int counter;

    // Setup IO.
    // Output file
    stringstream sstm;
	sstm.str("");
	sstm << outputDir << "/diffr_out_" << setfill('0') << setw(7) \
			      << comm->rank() << ".h5";
	string outputName = sstm.str();
	if ( boost::filesystem::exists( outputName ) ) {
		boost::filesystem::remove( outputName );
	}


	// Wave to master, we're good to go.
	comm->send(master, 0, &counter, 1);

    // Start event loop and process the diffraction images.
	while (true){
		comm->recv(master, 0, counter);
		if (counter < 0) return;
		make1Diffr(myQuaternions,counter,vm,outputName);
		comm->send(master, 0, &counter, 1);
	}
}

int prepH5(opt::variables_map vm, string outputFile) {

    //Get directories from input arguments.
    string inputDir = vm["inputDir"].as<std::string>();
    string outputDir = vm["outputDir"].as<string>();
    string configFile = vm["configFile"].as<string>();

    //// Check if output file exists, backup.
    //if (H5Fis_hdf5(outputFile.c_str()) > 0) { // file does not exist or is invalid
        //throw;
    //}

    // Create output file.
    hid_t file_id = H5Fcreate(outputFile.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    // Generate top level directories.
    //
    hid_t data_group_id = H5Gcreate( file_id, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t params_group_id = H5Gcreate( file_id, "params", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t misc_group_id = H5Gcreate( file_id, "misc", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t info_group_id = H5Gcreate( file_id, "info", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	hid_t props = H5Pcreate (H5P_DATASET_CREATE);
    hid_t str_type, dataspace, dataset;
	hsize_t rank,dimens_1d[1];

    // Write metadata.
    // Package format version.
	string text="SingFEL v0.2.0";
	str_type = H5Tcopy (H5T_C_S1);
	H5Tset_size (str_type, text.size()+1);
	dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	dataset = H5Dcreate(info_group_id, "package_version",str_type,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	H5Dwrite(dataset, str_type, H5S_ALL,H5S_ALL,H5P_DEFAULT,text.c_str());
	H5Sclose (dataspace);

    // Data format version.
	text="0.2";
	str_type = H5Tcopy (H5T_C_S1);
	H5Tset_size (str_type, text.size()+1);
	dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	dataset = H5Dcreate(file_id, "version",str_type,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	H5Dwrite(dataset, str_type, H5S_ALL,H5S_ALL,H5P_DEFAULT,text.c_str());
	H5Sclose (dataspace);

    // Contact
	text="Carsten Fortmann-Grote <carsten.grote@xfel.eu>";
	str_type = H5Tcopy (H5T_C_S1);
	H5Tset_size (str_type, text.size()+1);
	dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	dataset = H5Dcreate(info_group_id, "contact",str_type,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	H5Dwrite(dataset, str_type, H5S_ALL,H5S_ALL,H5P_DEFAULT,text.c_str());
	H5Sclose (dataspace);

    // Data Description.
   	text="This dataset contains diffraction patterns generated using SingFEL.";
	str_type = H5Tcopy (H5T_C_S1);
	H5Tset_size (str_type, text.size()+1);
	dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	dataset = H5Dcreate(info_group_id, "data_description",str_type,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	H5Dwrite(dataset, str_type, H5S_ALL,H5S_ALL,H5P_DEFAULT,text.c_str());
	H5Sclose (dataspace);

    // Method Description.
    text="Form factors of the radiation damaged molecules are calculated in time slices. At each time slice, the coherent scattering is calculated and incoherently added to the final diffraction pattern (/data/???????/diffr). Finally, Poissonian noise is added to the diffraction pattern (/data/???????/data).";
	str_type = H5Tcopy (H5T_C_S1);
	H5Tset_size (str_type, text.size()+1);
	dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	dataset = H5Dcreate(info_group_id, "method_description",str_type,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	H5Dwrite(dataset, str_type, H5S_ALL,H5S_ALL,H5P_DEFAULT,text.c_str());
	H5Sclose (dataspace);

    //// Parameters.
    // TODO: Store python script or "pickled" diffractor class.
    //text = vm["inputDir"].as<string>() + "\n" +\
           //vm["outputDir"].as<string>() + "\n" +\
           //vm["configFile"].as<string>() + "\n" +\
           //vm["beamFile"].as<string>() + "\n" +\
           //vm["geomFile"].as<string>() + "\n" +\
           //vm["prepHDF5File"].as<string>() + "\n" +\
           //vm["rotationAxis"].as<string>() + "\n" +\
           //vm["numSlices"].as<string>() + "\n" +\
           //vm["sliceInterval"].as<string>() + "\n" +\
           //vm["pmiStartID"].as<string>() + "\n" +\
           //vm["pmiEndID"].as<string>() + "\n" +\
           //vm["numDP"].as<string>() + "\n" +\
           //vm["calculateCompton"].as<string>() + "\n" +\
           //vm["uniformRotation"].as<string>() + "\n" +\
           //vm["saveSlices"].as<string>() + "\n" +\
           //vm["gpu"].as<string>();

    //str_type = H5Tcopy (H5T_C_S1);
	//H5Tset_size (str_type, text.size()+1);
	//dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	//dataset = H5Dcreate(params_group_id, "info",str_type,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	//H5Dwrite(dataset, str_type, H5S_ALL,H5S_ALL,H5P_DEFAULT,text.c_str());
	//H5Sclose (dataspace);

    H5Fclose(file_id);

    std::clog << "Created output file " << outputFile << std::endl;
    return 0;
}


void make1Diffr(const fmat& myQuaternions,int counter,opt::variables_map vm, string outputName ) {


	int sliceInterval = vm["sliceInterval"].as<int>();
	string inputDir = vm["inputDir"].as<std::string>();
	string outputDir = vm["outputDir"].as<string>();
	string configFile = vm["configFile"].as<string>();
	string beamFile = vm["beamFile"].as<string>();

	string geomFile = vm["geomFile"].as<string>();
	int numSlices = vm["numSlices"].as<int>();
	int saveSlices = vm["saveSlices"].as<int>();
	bool calculateCompton = vm["calculateCompton"].as<bool>();

	int pmiStartID = vm["pmiStartID"].as<int>();
	int numDP = vm["numDP"].as<int>();


	int pmiID = pmiStartID + counter/numDP;
	int diffrID = (pmiStartID-1)*numDP+1 + (pmiID-1)*numDP+counter%numDP;


	// Set up beam and detector from file
	CDetector det = CDetector();
	CBeam beam = CBeam();
	beam.readBeamFile(beamFile);
	det.readGeomFile(geomFile);

	bool givenFluence = false;
	if (beam.get_photonsPerPulse() > 0) {
		givenFluence = true;
	}
	bool givenPhotonEnergy = false;
	if (beam.get_photon_energy() > 0) {
		givenPhotonEnergy = true;
	}
	bool givenFocusRadius = false;
	if (beam.get_focus() > 0) {
		givenFocusRadius = true;
	}

	int px = det.get_numPix_x();
	int py = px;

	fmat photon_field(py,px);
	fmat detector_intensity(py,px);
	umat detector_counts(py,px);
	fmat F_hkl_sq(py,px);
	fmat Compton(py,px);
	fmat myPos;

	fvec quaternion(4);
	quaternion = trans(myQuaternions.row(counter));


	// input file
	stringstream sstm;
	sstm << inputDir << "/pmi_out_" << setfill('0') << setw(7) \
		     << pmiID << ".h5";

	string inputName = sstm.str();
	if ( !boost::filesystem::exists( inputName ) ) {
		cout << inputName << " does not exist!" << endl;
		exit(0);
	}

	// Set up diffraction geometry
	if (givenPhotonEnergy == false) {
		setEnergyFromFile(inputName, &beam);
	}
	if (givenFocusRadius == false) {
		setFocusFromFile(inputName, &beam);
	}
	det.init_dp(&beam);

	double total_phot = 0;
	photon_field.zeros(py,px);
	detector_intensity.zeros(py,px);
	detector_counts.zeros(py,px);
	int done = 0;
	int timeSlice = 0;
	int isFirstSlice = 1;
	while(!done) {	// sum up time slices
		setTimeSliceInterval(numSlices, &sliceInterval, &timeSlice, \
		                     &done);
		// Particle //
			CParticle particle = CParticle();
			loadParticle(vm, inputName, timeSlice, &particle);
		// Apply random rotation to particle
		rotateParticle(&quaternion, &particle);
		// Beam // FIXME: Check that these fields exist
		if (givenFluence == false) {
			setFluenceFromFile(inputName, timeSlice, sliceInterval, \
			                   &beam);
		}
		total_phot += beam.get_photonsPerPulse();
			// Coherent contribution
		CDiffraction::calculate_atomicFactor(&particle, &det);
			// Incoherent contribution
		if (calculateCompton) {
			getComptonScattering(vm, &particle, &det, &Compton);
		}
		#ifdef COMPILE_WITH_CUDA
		if (localRank < 2*deviceCount){
			float* F_mem = F_hkl_sq.memptr();
			// f_hkl: py x px x numAtomTypes
			float* f_mem = CDiffraction::f_hkl.memptr();
			float* q_mem = det.q_xyz.memptr();
			float* p_mem = particle.atomPos.memptr();
			int*   i_mem = particle.xyzInd.memptr();
			cuda_structureFactor(F_mem, f_mem, q_mem, p_mem, i_mem, \
			               det.numPix, particle.numAtoms,  \
			               particle.numAtomTypes,localRank%deviceCount);
			if (calculateCompton) {
				photon_field += (F_hkl_sq+Compton) % det.solidAngle \
				                % det.thomson \
				                * beam.get_photonsPerPulsePerArea();
			} else {
				photon_field += F_hkl_sq % det.solidAngle \
				                % det.thomson \
				                * beam.get_photonsPerPulsePerArea();
			}
		}else
		#endif
		{
			F_hkl_sq = CDiffraction::calculate_molecularFormFactorSq(&particle, &det);
			if (calculateCompton) {
				photon_field = (F_hkl_sq + Compton) % det.solidAngle \
				               % det.thomson \
				               * beam.get_photonsPerPulsePerArea();
			} else {
				photon_field = F_hkl_sq % det.solidAngle \
				               % det.thomson \
				               * beam.get_photonsPerPulsePerArea();
			}
		}
		detector_intensity += photon_field;

		//if (saveSlices) {
			//savePhotonField(outputName, counter, isFirstSlice, timeSlice, \
			                //&photon_field);
		//}
		isFirstSlice = 0;
	}// end timeSlice

	// Apply badpixelmap
	CDetector::apply_badPixels(&detector_intensity);
	// Poisson noise
	detector_counts = CToolbox::convert_to_poisson(&detector_intensity);

	// Save to HDF5
	saveAsDiffrOutFile(outputName, inputName, counter, &detector_counts, \
	                   &detector_intensity, &quaternion, &det, &beam, \
	                   total_phot);


}// end of diffract

void generateRotations(const bool uniformRotation, const string rotationAxis, \
                       const int numQuaternions, fmat* myQuaternions) {
	fmat& _myQuaternions = myQuaternions[0];

	_myQuaternions.zeros(numQuaternions,4);
	if (uniformRotation) { // uniform rotations
		if (rotationAxis == "y" || rotationAxis == "z") {
			_myQuaternions = CToolbox::pointsOn1Sphere(numQuaternions, \
			                                           rotationAxis);
		} else if (rotationAxis == "xyz") {
			_myQuaternions = CToolbox::pointsOn4Sphere(numQuaternions);
		}
	} else { // random rotations
		for (int i = 0; i < numQuaternions; i++) {
			_myQuaternions.row(i) = trans( \
			                         CToolbox::getRandomRotation(rotationAxis));
		}
	}
}

void setTimeSliceInterval(const int numSlices, int* sliceInterval, \
                          int* timeSlice, int* done) {
	if (*timeSlice + *sliceInterval >= numSlices) {
		*sliceInterval = numSlices - *timeSlice;
		*done = 1;
	}
	*timeSlice += *sliceInterval;
}

void loadParticle(const opt::variables_map vm, const string filename, \
                  const int timeSlice, CParticle* particle) {
	bool calculateCompton = vm["calculateCompton"].as<bool>();

	string datasetname;
	stringstream ss;
	ss << "/data/snp_" << setfill('0') << setw(7) << timeSlice;
	datasetname = ss.str();
	// rowvec atomType
	particle->load_atomType(filename,datasetname+"/T");
	// mat pos
	particle->load_atomPos(filename,datasetname+"/r");
	// rowvec ion list
	particle->load_ionList(filename,datasetname+"/xyz");
	// mat ffTable (atomType x qSample)
	particle->load_ffTable(filename,datasetname+"/ff");
	// rowvec q vector sin(theta)/lambda
	particle->load_qSample(filename,datasetname+"/halfQ");
	// Particle's inelastic properties
	if (calculateCompton) {
		// rowvec q vector sin(theta)/lambda
		particle->load_compton_qSample(filename,datasetname+"/Sq_halfQ");
		// rowvec static structure factor
		particle->load_compton_sBound(filename,datasetname+"/Sq_bound");
		// rowvec Number of free electrons
		particle->load_compton_nFree(filename,datasetname+"/Sq_free");
	}
}

void rotateParticle(fvec* quaternion, CParticle* particle) {
	fvec& _quat = quaternion[0];

	// Rotate particle
	fmat rot3D = CToolbox::quaternion2rot3D(_quat);
	fmat myPos = particle->get_atomPos();
	myPos = myPos * trans(rot3D); // rotate atoms
	particle->set_atomPos(&myPos);
}

void setFluenceFromFile(const string filename, const int timeSlice, \
                        const int sliceInterval, CBeam* beam) {
	double n_phot = 0;
	for (int i = 0; i < sliceInterval; i++) {
		string datasetname;
		stringstream ss;
		ss << "/data/snp_" << setfill('0') << setw(7) << timeSlice-i;
		datasetname = ss.str();
// SY: new format for fluence
//		double myNph = hdf5readConst<double>(filename,datasetname+"/Nph");
		vec vecNph;
		vecNph = hdf5read<vec>(filename,datasetname+"/Nph");
		if (vecNph.n_elem !=1)
		{
			cerr << "setFluenceFromFile: Wrong fluence format in : " << filename << endl;
			exit(0);
		}
		beam->set_photonsPerPulse(vecNph[0]);
		n_phot += beam->get_photonsPerPulse();	// number of photons per pulse
	}
	beam->set_photonsPerPulse(n_phot);
}

void setEnergyFromFile(const string filename, CBeam* beam) {
	// Read in photon energy
	double photon_energy = double(hdf5readScalar<float>(filename, \
	                             "/history/parent/detail/params/photonEnergy"));
	beam->set_photon_energy(photon_energy);
}

void setFocusFromFile(const string filename, CBeam* beam) {
	// Read in focus size
	double focus_xFWHM = double(hdf5readScalar<float>(filename,\
	                                      "/history/parent/detail/misc/xFWHM"));
	double focus_yFWHM = double(hdf5readScalar<float>(filename,\
	                                      "/history/parent/detail/misc/yFWHM"));
	beam->set_focus(focus_xFWHM, focus_yFWHM, "ellipse");
}

void getComptonScattering(const opt::variables_map vm, CParticle* particle, \
                          CDetector* det, fmat* Compton) {
	bool calculateCompton = vm["calculateCompton"].as<bool>();

	if (calculateCompton) {
		CDiffraction::calculate_compton(particle, det, Compton); // get S_hkl
	} else {
		Compton->zeros(det->py,det->px);
	}
}

void savePhotonField(const string filename, const int isFirstSlice, \
                     const int timeSlice, fmat* photon_field) {
	fmat& _photon_field = photon_field[0];


	std::stringstream ss;
	ss << "photonField_" << setfill('0') << setw(7) << timeSlice;
	string fieldName = ss.str();
	int success = hdf5writeVector(filename,  "/misc/photonField", fieldName, _photon_field );
	assert(success == 0);
}

void saveAsDiffrOutFile(const string outputName,\
                        const string inputName,\
                        int count,\
                        umat* detector_counts,\
                        fmat* detector_intensity,\
                        fvec* quaternion,\
                        CDetector* det,\
                        CBeam* beam,\
                        double total_phot) {

            // Data group.
            // Detector counts.
            stringstream sstm;
            sstm.str("");
            sstm << "data/" << setfill('0') << setw(7) << count << "/";
            string group_name = sstm.str();

            // Open output file.
            hid_t output_id = H5Fopen(outputName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

			int success = hdf5writeVector(outputName, group_name, "data", *detector_counts);

            // Detector intensity.
			success = hdf5writeVector(outputName,group_name, "diffr", *detector_intensity);

            // Quaternion
			success = hdf5writeVector(outputName, group_name, "angle", *quaternion);

            // History
            string history(group_name+"history");
            hid_t history_group_id = H5Gcreate( output_id, history.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            string parent(group_name+"history/parent");
            hid_t parent_group_id = H5Gcreate( output_id, parent.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            string detail(group_name+"history/parent/detail");
            hid_t detail_group_id = H5Gcreate( output_id, detail.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            success = linkHistory(output_id, inputName, group_name);

            // Parameters.
            // Geometry.
			double dist = det->get_detector_dist();
			success = hdf5writeScalar(outputName,"params/geom", "detectorDist", dist);
			double pixelWidth = det->get_pix_width();
			success = hdf5writeScalar(outputName,"params/geom", "pixelWidth", pixelWidth);
			double pixelHeight = det->get_pix_height();
			success = hdf5writeScalar(outputName,"params/geom", "pixelHeight", pixelHeight);
			fmat mask = ones<fmat>(det->py,det->px);
			success = hdf5writeVector(outputName,"params/geom", "mask", mask);
			double focusArea = beam->get_focus_area();
			success = hdf5writeScalar(outputName,"params/beam", "focusArea", focusArea);

            // Photons.
			double photonEnergy = beam->get_photon_energy();
			success = hdf5writeScalar(outputName,"params/beam", "photonEnergy", photonEnergy);
            //success = hdf5writeScalar(outputName,"params", "/params/beam/photons", total_phot); // Not needed.
            //
            H5Fclose( output_id );
}

opt::variables_map parse_input( int argc, char* argv[], \
                                mpi::communicator* comm ) {

    // Constructing an options describing variable and giving it a
    // textual description "All options"
    opt::options_description desc("All options");

    // When we are adding options, first parameter is a name
    // to be used in command line. Second parameter is a type
    // of that option, wrapped in value<> class. Third parameter
    // must be a short description of that option
    desc.add_options()
        ("inputDir", opt::value<std::string>(), \
                     "Input directory for finding /pmi and /diffr")
        ("outputDir", opt::value<string>(), \
                      "Output directory for saving diffraction")
        ("configFile", opt::value<string>(), \
                       "Absolute path to the config file")
        ("beamFile", opt::value<string>(), "Beam file defining X-ray beam")
        ("geomFile", opt::value<string>(), \
                     "Geometry file defining diffraction geometry")

        ("prepHDF5File", opt::value<string>(), \
                     "Absolute path to the prepHDF5.py script")

        ("rotationAxis", opt::value<string>()->default_value("xyz"), \
                         "Euler rotation convention")
        ("numSlices", opt::value<int>(), \
                      "Number of time-slices to use from \
                      Photon Matter Interaction (PMI) file")
        ("sliceInterval", opt::value<int>()->default_value(1), \
                          "Calculates photon field at every slice interval")
        ("pmiStartID", opt::value<int>()->default_value(1), \
                       "First Photon Matter Interaction (PMI) file ID to use")
        ("pmiEndID", opt::value<int>()->default_value(1), \
                     "Last Photon Matter Interaction (PMI) file ID to use")
        ("numDP", opt::value<int>()->default_value(1), \
                  "Number of diffraction patterns per PMI file")
        ("calculateCompton", opt::value<bool>()->default_value(0), \
                 "If 1, includes Compton scattering in the diffraction pattern")
        ("uniformRotation", opt::value<bool>()->default_value(0), \
                            "If 1, rotates the sample uniformly in SO(3)")
        ("saveSlices", opt::value<int>()->default_value(0), \
                       "If 1, saves time-slices of the photon field \
                        in hdf5 under /misc/photonField")
        ("gpu", opt::value<int>()->default_value(0), \
                "If 1, uses NVIDIA CUDA for faster calculation")
        ("help", "produce help message")
    ;

    // Variable to store our command line arguments
    opt::variables_map vm;

    // Parsing and storing arguments
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
	opt::notify(vm);

	// Print input arguments
    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(0);
    }

	if (comm->rank() == master) {
		if (vm.count("inputDir"))
    		cout << "inputDir: " << vm["inputDir"].as<string>() << endl;
		if (vm.count("outputDir"))
    		cout << "outputDir: " << vm["outputDir"].as<string>() << endl;
		if (vm.count("configFile"))
    		cout << "configFile: " << vm["configFile"].as<string>() << endl;
		if (vm.count("beamFile"))
    		cout << "beamFile: " << vm["beamFile"].as<string>() << endl;
		if (vm.count("geomFile"))
    		cout << "geomFile: " << vm["geomFile"].as<string>() << endl;
		if (vm.count("prepHDF5File"))
    		cout << "prepHDF5File: " << vm["prepHDF5File"].as<string>() << endl;
		if (vm.count("rotationAxis"))
    		cout << "rotationAxis: " << vm["rotationAxis"].as<string>() << endl;
		if (vm.count("numSlices"))
    		cout << "numSlices: " << vm["numSlices"].as<int>() << endl;
		if (vm.count("sliceInterval"))
    		cout << "sliceInterval: " << vm["sliceInterval"].as<int>() << endl;
		if (vm.count("pmiStartID"))
    		cout << "pmiStartID: " << vm["pmiStartID"].as<int>() << endl;
		if (vm.count("pmiEndID"))
    		cout << "pmiEndID: " << vm["pmiEndID"].as<int>() << endl;
		if (vm.count("numDP"))
    		cout << "numDP: " << vm["numDP"].as<int>() << endl;
		if (vm.count("calculateCompton"))
    		cout << "calculateCompton: " << vm["calculateCompton"].as<bool>() \
    		     << endl;
		if (vm.count("uniformRotation"))
    		cout << "uniformRotation: " << vm["uniformRotation"].as<bool>() \
    		     << endl;
		if (vm.count("saveSlices"))
    		cout << "saveSlices: " << vm["saveSlices"].as<int>() << endl;
		if (vm.count("gpu"))
    		cout << "gpu: " << vm["gpu"].as<int>() << endl;
	}
	return vm;
} // end of parse_input

/*
 * Link history from input pmi file into output diffr file
 */
int linkHistory(hid_t out_id, const string inputName, const string group_name) {

    // Data.
    hid_t link_id = H5Lcreate_external( inputName.c_str(), "/data", out_id, (group_name+"/history/parent/detail/data").c_str(), H5P_DEFAULT, H5P_DEFAULT);
    // History
    link_id =       H5Lcreate_external( inputName.c_str(), "/history/parent", out_id, (group_name+"/history/parent/parent").c_str(), H5P_DEFAULT, H5P_DEFAULT);
    // Info
    link_id =       H5Lcreate_external( inputName.c_str(), "/info", out_id, (group_name+"/history/parent/detail/info").c_str(), H5P_DEFAULT, H5P_DEFAULT);
    // Misc
    link_id =       H5Lcreate_external( inputName.c_str(), "/misc", out_id, (group_name+"/history/parent/detail/misc").c_str(), H5P_DEFAULT, H5P_DEFAULT);
    // Parameters
    link_id =       H5Lcreate_external( inputName.c_str(), "/params",  out_id, (group_name+"/history/parent/detail/params").c_str(), H5P_DEFAULT, H5P_DEFAULT);
    // Version
    link_id =       H5Lcreate_external( inputName.c_str(), "/version", out_id, (group_name+"/history/parent/detail/version").c_str(), H5P_DEFAULT, H5P_DEFAULT);

    return 0;
}




