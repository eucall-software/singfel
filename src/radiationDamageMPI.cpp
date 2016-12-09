/*
 * MPI enabled program for simulating diffraction patterns
 */

#include "hdf5.h"
#include "hdf5_hl.h"
#include <algorithm>
#include <armadillo>
#include <boost/algorithm/string.hpp>
#include <boost/mpi.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/string.hpp>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/time.h>

// SingFEL library
#include "beam.h"
#include "detector.h"
#include "diffraction.h"
#include "io.h"
#include "particle.h"
#include "toolbox.h"

// Check if code is compiled for execution on GPU (CUDA).
#ifdef COMPILE_WITH_CUDA
#include "diffraction.cuh"
#endif

// A couple of useful namespaces.
namespace mpi = boost::mpi;
namespace opt = boost::program_options;
using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

// Useful tags
#define QTAG 1	// quaternion
#define DPTAG 2	// diffraction pattern
#define DIETAG 3 // die signal
#define DONETAG 4 // done signal

#define MPI_SHMKEY 0x6FB1407

// Define master node and message length
const int master = 0;
const int msgLength = 4;

// Only if CUDA.
#ifdef COMPILE_WITH_CUDA
int localRank=0;
int deviceCount=cuda_getDeviceCount();
#endif

// Local declarations.
/* @brief: Master process, steers distribution of tasks over slave processes.
 * @param comm: MPI communicator
 * @param vm: Map of command line arguments
 */
static void master_diffract(mpi::communicator* comm, opt::variables_map vm);

/* @brief: Slave process for calculation of diffraction patterns.
 * @param comm: Pointer to MPI communicator.
 * @param vm: Map of command line arguments
 */
static void slave_diffract(mpi::communicator* comm, opt::variables_map vm);

/* @brief: Parse command line arguments and setup argument map.
 * @param argc: Number of command line arguments.
 * @param argv: Command line arguments list.
 * @param comm: Pointer to MPI communicator.
 */
opt::variables_map parse_input(int argc, char* argv[], mpi::communicator* comm);

/* @brief Calculates a single diffraction pattern and stores in an hdf5 file.
 * @param myQuaternions: Quaternion to use for this pattern.
 * @param counter: Global pattern count for this run.
 * @param vm: Map of command line arguments.
 * @param outputName: Name of output hdf5 file.
 */
void make1Diffr(const fmat& myQuaternions,int counter,opt::variables_map vm, string outputName);

/* @brief: Generate the rotations for this run.
 * @param uniformRotation: Whether to sample the rotation space uniformly.
 * @param rotationAxis: xyz or Euler convention.
 * @param numQuaternions: Number of quaternions (rotations).
 * @param myQuaternions: Pointer to matrix holding the quaternions.
 */
void generateRotations(const bool uniformRotation, \
                       const string rotationAxis, const int numQuaternions, \
                       fmat* myQuaternions);

/* @brief: Loads the atom coordinates, structure factors, and form factors for given time slice.
 * @param vm: Map of command line arguments.
 * @param filename: Name of hdf5 file that holds the particle data.
 * @param timeslice: Which time slice to calculate.
 * @param particle: The particle to calculate.
 */
void loadParticle(const opt::variables_map vm, const string filename, \
                  const int timeSlice, CParticle* particle);

/* @brief: Set the interval of time slices to use in the calculations.
 * @param numSlices: How many slices to use.
 * @param: sliceInterval: From which interval to take the time slices.
 * @param timeSlice: The time slice to use.
 * @param done: Flag indicating success of the operation.
 */
void setTimeSliceInterval(const int numSlices, int* sliceInterval, \
                          int* timeSlice, int* done);

/* @brief: Rotate the particle
 * @param Quaternion: The quaternion to apply to the atom coordinates.
 * @param particle: The particle to rotate.
 */
void rotateParticle(fvec* quaternion, CParticle* particle);

/* @brief: Extract the photon fluence from the input file and set in calculation.
 * @param filename: The filename to read the fluence from.
 * @param timeSlice: For which timeSlice to set the fluence.
 * @sliceInterval: The time slice interval
 * @beam: The beam information struct.
 */
void setFluenceFromFile(const string filename, const int timeSlice, \
                        const int sliceInterval, CBeam* beam);

/* @brief: Extract photon energy from a file and use in calculation.
 * @param filename: The file to extract the energy from.
 * @param beam: The beam struct.
 */
void setEnergyFromFile(const string filename, CBeam* beam);

/* @brief: Extract focus size from a file and use in calculation.
 * @param filename: File to extract focus from.
 * @param beam: The beam info struct.
 */
void setFocusFromFile(const string filename, CBeam* beam);

/* @brief: Calculate Compton scattering.
 * @param vm: Map of command line arguments.
 * @param particle: The particle to calculate.
 * @param det: The detector struct.
 * @param [out] Compton: Matrix holding the result.
 */
void getComptonScattering(const opt::variables_map vm, CParticle* particle, \
                          CDetector* det, fmat* Compton);

/* @brief: Save the photon field in the output file.
 * @param filename: File to save field in.
 * @param isFirstSlice: Whether it's the first time slice.
 * @param timeSlice: The time slice of this calculation.
 * @param photon_field: The field to save.
 */
void savePhotonField(const string filename, const int isFirstSlice, \
                     const int timeSlice, fmat* photon_field);

/* @brief: Save diffraction pattern in hdf5 file.
 * @param outputName: Name of the output file.
 * @param inputName: Name of the input file to extract history and metadata from.
 * @param counter: Running counter of diffraction patterns.
 * @param detector_counts: Matrix (image) of detector counts (Poissonized).
 * @param detector_intensity: Matrix (image) of detector intensity (before Poissonization).
 * @param quaternion: The quaternion used to rotate the sample for this pattern.
 * @param det: The detector info struct.
 * @param beam: beam: The beam info struct.
 * @param total_phot: Total photon count (obsolete).
 */
void saveAsDiffrOutFile(const string outputName, const string inputName, int counter, umat* detector_counts, fmat* detector_intensity, fvec* quaternion, CDetector* det, CBeam* beam, double total_phot);

/* @brief: Prepare the output file.
 * @param vm: Map of command line arguments.
 * @outputName: Name of output hdf5 file.
 */
int prepH5(opt::variables_map vm, string outputName);

/* @brief: Link data from preceeding calculator into output file.
 * @param output_file_id: H5 file id for output file (must be open).
 * @param inputName: Link target.
 * @param group_name: Name of hdf5 group inside source file where to save the links.
 */
int linkHistory( hid_t output_file_id, const string inputName, const string group_name);


// Main
int main( int argc, char* argv[] ){

	// Initialize MPI
  	mpi::environment env;
  	mpi::communicator world;
	mpi::communicator* comm = &world;

	// All processes parse the input
	opt::variables_map vm = parse_input(argc, argv, comm);

	// Set random seed if CUDA.
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

    // Init the time.
	wall_clock timerMaster;

    // Tic.
	timerMaster.tic();

	// Main program
	if (world.rank() == master) {
		master_diffract(comm, vm);
	} else {
		slave_diffract(comm, vm);
	}

    // Only on GPU.
#ifdef COMPILE_WITH_CUDA

	world.barrier();
	if (world.rank() != master) {
		shmdt(shmval);
		shmctl(shmid, IPC_RMID, 0);
	}
#endif

    // Sync.
	world.barrier();
    // Time.
	if (world.rank() == master) {
		cout << "Finished: " << timerMaster.toc() <<" seconds."<<endl;
	}

    // Clean up hdf5.
    int h5_closed = H5close();

    // Finalize MPI.
    int finalized = MPI_Finalize();

  	return 0;
}

static void master_diffract(mpi::communicator* comm, opt::variables_map vm) {

    // Get some required command line arguments.
	int pmiStartID = vm["pmiStartID"].as<int>();
	int pmiEndID = vm["pmiEndID"].as<int>();
	int numDP = vm["numDP"].as<int>();

    // Number of processes.
	int numProcesses = comm->size();

    // Number of tasks.
  	int ntasks = (pmiEndID-pmiStartID+1)*numDP;

    // Setup quaternions.
	fmat myQuaternions;

    // Setup output filename.
    string outputName = "";

    // If only one process.
	if (numProcesses==1)
	{
        // Some more command line arguments.
		string rotationAxis = vm["rotationAxis"].as<string>();
		bool uniformRotation = vm["uniformRotation"].as<bool>();

        // Generate the rotations.
		generateRotations(uniformRotation, rotationAxis, ntasks, \
	                  &myQuaternions);

        // Prepare one hdf5 file for output.
        outputName = vm["outputDir"].as<string>() + "/diffr_out_0000001.h5";
        if ( boost::filesystem::exists( outputName ) ) {
	    	boost::filesystem::remove( outputName );
	    }
        int success = prepH5(vm, outputName);
        assert(success == 0);
	}

    // Loop over all tasks.
	for (int ntask = 0; ntask < ntasks; ntask++) {
        // If only one process, each pattern is calculated on master.
		if (numProcesses==1)
		{
			make1Diffr(myQuaternions,ntask,vm, outputName);
		}
		else
		{
            // Get status from slave.
			int tmp;
		  	boost::mpi::status status = comm->recv(boost::mpi::any_source, 0, tmp);
            // Trigger calculation on slave.
			comm->send(status.source(), 0, &ntask, 1);
		}

        // Print progress.
        std::cout << "Completed " << ntask+1 << " of " << ntasks << std::endl;
	}

    // Final send.
	int ntask=-1;
	for (int np = 0; np < numProcesses; np++) {
		if (np!= master)
			{
				comm->send(np, 0, &ntask, 1);
			}
	}
}


static void slave_diffract(mpi::communicator* comm, opt::variables_map vm) {

    // Get some command line arguments.
	bool uniformRotation = vm["uniformRotation"].as<bool>();
	int numDP = vm["numDP"].as<int>();
	int pmiEndID = vm["pmiEndID"].as<int>();
	int pmiStartID = vm["pmiStartID"].as<int>();
	string inputDir = vm["inputDir"].as<std::string>();
	string outputDir = vm["outputDir"].as<string>();
	string rotationAxis = vm["rotationAxis"].as<string>();

    // Number of tasks.
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

    // Prepare the output file.
    int success = prepH5(vm, outputName);

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

    // Create output file.
    hid_t file_id = H5Fcreate(outputFile.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    // Generate top level directories.
    //
    hid_t data_group_id = H5Gcreate( file_id, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t params_group_id = H5Gcreate( file_id, "params", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t misc_group_id = H5Gcreate( file_id, "misc", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t info_group_id = H5Gcreate( file_id, "info", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Needed h5 resources.
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
    text="Form factors of the radiation damaged molecules are calculated in time slices. At each time slice, the coherent scattering is calculated and incoherently added to the final diffraction pattern (/data/nnnnnnn/diffr). Finally, Poissonian noise is added to the diffraction pattern (/data/nnnnnnn/data).";
	str_type = H5Tcopy (H5T_C_S1);
	H5Tset_size (str_type, text.size()+1);
	dataspace = H5Screate_simple(rank, dimens_1d, NULL);
	dataset = H5Dcreate(info_group_id, "method_description",str_type,dataspace,H5P_DEFAULT,props,H5P_DEFAULT);
	H5Dwrite(dataset, str_type, H5S_ALL,H5S_ALL,H5P_DEFAULT,text.c_str());
	H5Sclose (dataspace);

    // Close file.
    H5Fclose(file_id);

    return 0;
}


void make1Diffr(const fmat& myQuaternions,int counter,opt::variables_map vm, string outputName ) {
    // Get required command line arguments.
	bool calculateCompton = vm["calculateCompton"].as<bool>();
	int numDP = vm["numDP"].as<int>();
	int numSlices = vm["numSlices"].as<int>();
	int pmiStartID = vm["pmiStartID"].as<int>();
	int pmiID = pmiStartID + counter/numDP;
	int diffrID = (pmiStartID-1)*numDP+1 + (pmiID-1)*numDP+counter%numDP;
	int saveSlices = vm["saveSlices"].as<int>();
	int sliceInterval = vm["sliceInterval"].as<int>();
	string beamFile = vm["beamFile"].as<string>();
	string configFile = vm["configFile"].as<string>();
	string geomFile = vm["geomFile"].as<string>();
	string inputDir = vm["inputDir"].as<std::string>();
	string outputDir = vm["outputDir"].as<string>();

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

    // Detector geometry.
	int px = det.get_numPix_x();
	int py = px;

    // Init matrices.
	fmat photon_field(py,px);
	fmat detector_intensity(py,px);
	umat detector_counts(py,px);
	fmat F_hkl_sq(py,px);
	fmat Compton(py,px);
	fmat myPos;

    // Setup quaternions.
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

    // Zero out the data fields.
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
		// Particle
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
        std::clog << timeSlice << "\t" <<  detector_intensity.max() << std::endl;

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
}

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

            std::clog << "in saveAsDiffrOutFile " << detector_intensity->max() << std::endl;

			int success = hdf5writeVector(outputName, group_name, "data", *detector_counts);
            std::cout << "Data written with success = " << success << std::endl;

            // Detector intensity.
			success = hdf5writeVector(outputName,group_name, "diffr", *detector_intensity);
            std::cout << "Diffr written with success = " << success << std::endl;

            // Quaternion
			success = hdf5writeVector(outputName, group_name, "angle", *quaternion);
            std::cout << "Angle written with success = " << success << std::endl;

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
            std::cout << "Detector dist. written with success = " << success << std::endl;
			double pixelWidth = det->get_pix_width();
			success = hdf5writeScalar(outputName,"params/geom", "pixelWidth", pixelWidth);
            std::cout << "pixel width written with success = " << success << std::endl;
			double pixelHeight = det->get_pix_height();
			success = hdf5writeScalar(outputName,"params/geom", "pixelHeight", pixelHeight);
            std::cout << "pixel height written with success = " << success << std::endl;
			fmat mask = ones<fmat>(det->py,det->px);
			success = hdf5writeVector(outputName,"params/geom", "mask", mask);
            std::cout << "mask written with success = " << success << std::endl;
			double focusArea = beam->get_focus_area();
			success = hdf5writeScalar(outputName,"params/beam", "focusArea", focusArea);
            std::cout << "focus area written with success = " << success << std::endl;

            // Photons.
			double photonEnergy = beam->get_photon_energy();
			success = hdf5writeScalar(outputName,"params/beam", "photonEnergy", photonEnergy);
            std::cout << "photon energy written with success = " << success << std::endl;
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




