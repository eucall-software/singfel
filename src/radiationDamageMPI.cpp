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
// Armadillo library
#include <armadillo>
// Boost library
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
// HDF5 library
#include "hdf5.h"
#include "hdf5_hl.h"
// SingFEL library
#include "detector.h"
#include "beam.h"
#include "particle.h"
#include "diffraction.h"
#include "toolbox.h"
#include "diffraction.cuh"
#include "io.h"

namespace mpi = boost::mpi;
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

static void master_diffract(mpi::communicator* comm, int pmiStartID, \
                            int pmiEndID, int numDP, int sliceInterval, \
                            string rotationAxis, int uniformRotation);
static void slave_diffract(mpi::communicator* comm, string inputDir, \
                           string outputDir, string configName, \
                           string beamFile, string geomFile, int numSlices, \
                           int calculateCompton, int saveSlices);

int main( int argc, char* argv[] ){

	// Initialize MPI
  	mpi::environment env;
  	mpi::communicator world;
	mpi::communicator* comm = &world;
	const int master = 0;
	
	// All processes do this
	string beamFile;
	string geomFile;
	int gpu = 0; // not used
	string output;
    
	string inputDir;
	string outputDir;
	string configName;
	string rotationAxis = "xyz";
	
	int sliceInterval;
	int numSlices;
	int pmiStartID;
	int pmiEndID;
	int dpID;
	int numDP;
	int calculateCompton = 0;
	int uniformRotation = 0;
	int saveSlices = 0;
	
	// Let's parse input
	for (int n = 1; n < argc; n++) {
		cout << argv [ n ] << endl;
		if(boost::algorithm::iequals(argv[ n ], "--sliceInterval")) {
		    sliceInterval = atoi(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--input_dir")) {
		    inputDir = argv[ n+2 ];
		} else if (boost::algorithm::iequals(argv[ n ], "--output_dir")) {
		    outputDir = argv[ n+2 ];
		} else if (boost::algorithm::iequals(argv[ n ], "--config_file")) {
		    configName = argv[ n+2 ];
		} else if (boost::algorithm::iequals(argv[ n ], "-b")) {
		    beamFile = argv[ n+1 ];
		} else if (boost::algorithm::iequals(argv[ n ], "-g")) {
		    geomFile = argv[ n+1 ];
		} else if (boost::algorithm::iequals(argv[ n ], "--gpu")) {
		    gpu = 1;
		} else if (boost::algorithm::iequals(argv[ n ], "--numSlices")) {
		    numSlices = atoi(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--pmiStartID")) {
		    pmiStartID = atoi(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--pmiEndID")) {
		    pmiEndID = atoi(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--dpID")) {
		    dpID = atoi(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--numDP")) {
		    numDP = atoi(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--rotationAxis")) {
		    rotationAxis = argv[ n+2 ];
		} else if (boost::algorithm::iequals(argv[ n ], "--uniformRotation")) {
		    uniformRotation = atoi(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--calculateCompton")) {
		    calculateCompton = atoi(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--saveSlices")) {
		    saveSlices = atoi(argv[ n+2 ]);
		}
	}
  	
	wall_clock timerMaster;

	timerMaster.tic();

	world.barrier();

	srand( pmiStartID + world.rank() + (unsigned)time(NULL) );

	// Main program
	if (world.rank() == master) {
		/* initialize random seed: */
		master_diffract(comm, pmiStartID, pmiEndID, numDP, sliceInterval, \
                        rotationAxis, uniformRotation);
	} else {
		slave_diffract(comm, inputDir, outputDir, configName, beamFile, \
                       geomFile, numSlices, calculateCompton, saveSlices);
	}
	world.barrier();
	if (world.rank() == master) {
		cout << "Finished: " << timerMaster.toc() <<" seconds."<<endl;
	}

  	return 0;
}

static void master_diffract(mpi::communicator* comm, int pmiStartID, int pmiEndID, int numDP, int sliceInterval, string rotationAxis, int uniformRotation) {

  	int ntasks, rank, numProcesses, numSlaves;
  	int numTasksDone = 0;
  	boost::mpi::status status;
  	float msg[4];

	ntasks = (pmiEndID-pmiStartID+1)*numDP;
	numProcesses = comm->size();
	numSlaves = comm->size()-1;

	if (numSlaves > ntasks) {
		cout << "Reduce number of slaves and restart" << endl;
		for (rank = 1; rank < numProcesses; ++rank) {
			comm->send(rank, DIETAG, msg);
			cout << "Killing: " << rank << endl;
		}
		return;
	}

	// Send
	// 1) pmiID
	// 2) diffrID
	// 3) sliceInterval
	 
	int diffrID = (pmiStartID-1)*numDP+1;
	int pmiID = pmiStartID;
	int dpID = 1;
	fvec quaternion(4);
	
	// Setup rotations
	fmat myQuaternions;
	myQuaternions.zeros(ntasks,4);
	if (uniformRotation) { // uniform rotations
		if (rotationAxis == "y" || rotationAxis == "z") {
			myQuaternions = CToolbox::pointsOn1Sphere(ntasks, rotationAxis);		
		} else if (rotationAxis == "xyz") {
			myQuaternions = CToolbox::pointsOn4Sphere(ntasks);
		}
	} else { // random rotations
		for (int i = 0; i < ntasks; i++) {
			myQuaternions.row(i) = trans(CToolbox::getRandomRotation(rotationAxis));
		}
	}

	int counter = 0;
	int done = 0;
	for (rank = 1; rank < numProcesses; ++rank) {
		if (pmiID > pmiEndID) {
			cout << "Error!!" << endl;
			return;	
		}
		// Tell the slave how to rotate the particle
		quaternion = trans(myQuaternions.row(counter));
		counter++;
		float* quat = &quaternion[0];
		comm->send(rank, QTAG, quat,4);
		// Tell the slave to compute DP
		fvec id;
		id << pmiID << diffrID << sliceInterval << endr;
		float* id1 = &id[0];
		comm->send(rank, DPTAG, id1, 3);
				
		diffrID++;
		dpID++;
		numTasksDone++;
		if (dpID > numDP) {
			dpID = 1;
			pmiID++;
		}
	}

	// Listen for slaves
	int msgDone = 0;
	
	if (numTasksDone >= ntasks) done = 1;
	while (!done) {
		status = comm->recv(boost::mpi::any_source, boost::mpi::any_tag, msgDone);
		// Tell the slave how to rotate the particle
		quaternion = trans(myQuaternions.row(counter));
		float* quat = &quaternion[0];
		counter++;
		comm->send(status.source(), QTAG, quat, 4);
		// Tell the slave to compute DP
		fvec id;
		id << pmiID << diffrID << sliceInterval << endr;
		float* id1 = &id[0];
		comm->send(status.source(), DPTAG, id1, 3);
		
		diffrID++;
		dpID++;
		numTasksDone++;
		if (dpID > numDP) {
			dpID = 1;
			pmiID++;
		}
		if (numTasksDone >= ntasks) {
			done = 1;
		}
	}
	
  	// Wait for status update of slaves.
	for (rank = 1; rank < numProcesses; ++rank) {
		status = comm->recv(rank, boost::mpi::any_tag, msgDone);
	}
    
	// KILL SLAVES
  	// Tell all the slaves to exit by sending an empty message with the DIETAG.
	for (rank = 1; rank < numProcesses; ++rank) {
		comm->send(rank, DIETAG, 1);
	}
}

static void slave_diffract(mpi::communicator* comm, string inputDir, \
                           string outputDir, string configName, \
                           string beamFile, string geomFile, int numSlices, \
                           int calculateCompton, int saveSlices) {
	
	wall_clock timer;
	boost::mpi::status status;
	const int master = 0;

	/****** Beam ******/
	// Let's read in our beam file
	double photon_energy = 0;
	double n_phot = 0;
	double focus_radius = 0;
	int givenPhotonEnergy = 0;
	int givenFluence = 0;
	int givenFocusRadius = 0;
	string line;
	ifstream myFile(beamFile.c_str());
	while (getline(myFile, line)) {
		if (line.compare(0,1,"#") && line.compare(0,1,";") && line.length() > 0) {
			// line now contains a valid input
			typedef boost::tokenizer<boost::char_separator<char> > Tok;
			boost::char_separator<char> sep(" ="); // default constructed
			Tok tok(line, sep);
			for(Tok::iterator tok_iter = tok.begin(); tok_iter != tok.end(); ++tok_iter){
				if ( boost::algorithm::iequals(*tok_iter,"beam/photon_energy") ) {            
					string temp = *++tok_iter;
					photon_energy = atof(temp.c_str()); // photon energy to wavelength
					givenPhotonEnergy = 1;
					break;
				} else if ( boost::algorithm::iequals(*tok_iter,"beam/fluence") ) {            
					string temp = *++tok_iter;
					n_phot = atof(temp.c_str()); // number of photons per pulse
					givenFluence = 1;
					break;
				} else if ( boost::algorithm::iequals(*tok_iter,"beam/radius") ) {            
					string temp = *++tok_iter;
					focus_radius = atof(temp.c_str()); // focus radius
					givenFocusRadius = 1;
					break;
				}
			}
		}
	}
	CBeam beam = CBeam();

	/****** Detector ******/
	double d = 0;					// (m) detector distance
	double pix_width = 0;			// (m)
	int px_in = 0;                  // number of pixel along x
	string badpixmap = ""; // this information should go into the detector class
	// Parse the geom file
	ifstream myGeomFile(geomFile.c_str());
	while (getline(myGeomFile, line)) {
		if (line.compare(0,1,"#") && line.compare(0,1,";") && line.length() > 0) {
			// line now contains a valid input 
			typedef boost::tokenizer<boost::char_separator<char> > Tok;
			boost::char_separator<char> sep(" ="); // default constructed
			Tok tok(line, sep);
			for(Tok::iterator tok_iter = tok.begin(); tok_iter != tok.end(); ++tok_iter){
				if ( boost::algorithm::iequals(*tok_iter,"geom/d") ) {            
					string temp = *++tok_iter;
					d = atof(temp.c_str());
					break;
				} else if ( boost::algorithm::iequals(*tok_iter,"geom/pix_width") ) {            
					string temp = *++tok_iter;
					pix_width = atof(temp.c_str());
					break;
				} else if ( boost::algorithm::iequals(*tok_iter,"geom/px") ) {            
					string temp = *++tok_iter;
					px_in = atof(temp.c_str());
					break;
				} else if ( boost::algorithm::iequals(*tok_iter,"geom/badpixmap") ) {            
					string temp = *++tok_iter;
					badpixmap = temp;
					break;
 				}
			}
		}
	}
	double pix_height = pix_width;		// (m)
	const int px = px_in;				// number of pixels in x
	const int py = px;					// number of pixels in y	
	double cx = ((double) px-1)/2;		// this can be user defined
	double cy = ((double) py-1)/2;		// this can be user defined

	CDetector det = CDetector();
	det.set_detector_dist(d);	
	det.set_pix_width(pix_width);	
	det.set_pix_height(pix_height);
	det.set_numPix(py,px);
	det.set_center_x(cx);
	det.set_center_y(cy);
	det.set_pixelMap(badpixmap);
	uvec goodpix = det.get_goodPixelMap();
	
	float msg[4]; //std::vector<float> msg;
	fmat rot3D(3,3);
	fvec quaternion(4);
	fmat photon_field(py,px);
	fmat detector_intensity(py,px);
	umat detector_counts(py,px);
	fmat F_hkl_sq(py,px);
	fmat Compton(py,px);
	fmat myPos;

	while (1) {
		// Receive a message from the master
    	status = comm->recv(master, boost::mpi::any_tag, msg, 4);

    	if (status.tag() == QTAG) {
    		quaternion << msg[0] << msg[1] << msg[2] << msg[3] << endr;
    	}

		// Receive how many slices assigned to this slave
		if (status.tag() == DPTAG) {

			timer.tic();

    		int pmiID = (int) msg[0];
    		int diffrID = (int) msg[1];
    		int sliceInterval = (int) msg[2];
    		
    		//TO DO: Check pmiID exists in the workflow
		
			// input file
			string filename;
			stringstream sstm;
			sstm << inputDir << "/pmi/pmi_out_" << setfill('0') << setw(7) << pmiID << ".h5";
			filename = sstm.str();

			// output file
			stringstream sstm3;
			sstm3 << outputDir << "/diffr_out_" << setfill('0') << setw(7) << diffrID << ".h5";
			string outputName;
			outputName = sstm3.str();
			if ( boost::filesystem::exists( outputName ) ) {
				boost::filesystem::remove( outputName );
			}

			// Run prepHDF5
			string scriptName;
			stringstream sstm2;
			sstm2 << inputDir << "/prepHDF5.py";
			scriptName = sstm2.str();
			string myCommand = string("python ") + scriptName + " " + filename + " " + outputName + " " + configName;
			int i = system(myCommand.c_str());
			
			// Rotate single particle			
			rot3D = CToolbox::quaternion2rot3D(quaternion);

			double total_phot = 0;
			photon_field.zeros(py,px);
			detector_intensity.zeros(py,px);
			detector_counts.zeros(py,px);	
			int done = 0;
			int timeSlice = 0;
			int isFirstSlice = 1;
			while(!done) {	// sum up time slices
				if (timeSlice+sliceInterval >= numSlices) {
					sliceInterval = numSlices - timeSlice;
					done = 1;
				}
				timeSlice += sliceInterval;

				string datasetname;
				stringstream sstm0;
				sstm0 << "/data/snp_" << setfill('0') << setw(7) << timeSlice;
				datasetname = sstm0.str();
		
				// Particle //
				CParticle particle = CParticle();
				particle.load_atomType(filename,datasetname+"/T"); 	// rowvec atomType
				particle.load_atomPos(filename,datasetname+"/r");		// mat pos
				particle.load_ionList(filename,datasetname+"/xyz");		// rowvec ion list
				particle.load_ffTable(filename,datasetname+"/ff");	// mat ffTable (atomType x qSample)
				particle.load_qSample(filename,datasetname+"/halfQ");	// rowvec q vector sin(theta)/lambda
				// Particle's inelastic properties
				if (calculateCompton) {
					particle.load_compton_qSample(filename,datasetname+"/Sq_halfQ");	// rowvec q vector sin(theta)/lambda
					particle.load_compton_sBound(filename,datasetname+"/Sq_bound");	// rowvec static structure factor
					particle.load_compton_nFree(filename,datasetname+"/Sq_free");	// rowvec Number of free electrons
				}
				// Rotate atom positions
				int numAtoms = particle.get_numAtoms();
				myPos.zeros(numAtoms,3);
				myPos = particle.get_atomPos();
//cout << "myPos: " << myPos(1) << endl;
				myPos = myPos * trans(rot3D); // rotate atoms
				particle.set_atomPos(&myPos);

				// Beam //
				if (givenFluence == 0) {
					n_phot = 0;
					for (int i = 0; i < sliceInterval; i++) {
						string datasetname;
						stringstream sstm0;
						sstm0 << "/data/snp_" << setfill('0') << setw(7) << timeSlice-i;
						datasetname = sstm0.str(); 					
						vec myNph = hdf5readT<vec>(filename,datasetname+"/Nph");
						beam.set_photonsPerPulse(myNph[0]);
						n_phot += beam.get_photonsPerPulse();	// number of photons per pulse
					}
					total_phot += n_phot;
					beam.set_photonsPerPulse(n_phot);
				} else {
					total_phot = n_phot;
					beam.set_photonsPerPulse(n_phot);
				}
				
				if (givenPhotonEnergy == 0) {
					// Read in photon energy
					photon_energy = double(hdf5readScalar<float>(filename,"/history/parent/detail/params/photonEnergy"));
					beam.set_photon_energy(photon_energy);
				} else {
					beam.set_photon_energy(photon_energy);
				}
	
				if (givenFocusRadius == 0) {
					// Read in focus size
					double focus_xFWHM = double(hdf5readScalar<float>(filename,"/history/parent/detail/misc/xFWHM"));
					double focus_yFWHM = double(hdf5readScalar<float>(filename,"/history/parent/detail/misc/yFWHM"));
					beam.set_focus(focus_xFWHM,focus_yFWHM,"ellipse");
cout << "focus: " << focus_xFWHM << " " << focus_yFWHM << endl;
				} else {
					beam.set_focus(focus_radius*2);
				}
				////////////////////////

				det.init_dp(&beam);
				CDiffraction::calculate_atomicFactor(&particle, &det); // get f_hkl
				Compton.zeros(py,px);
				if (calculateCompton) {
					Compton = CDiffraction::calculate_compton(&particle, &det); // get S_hkl
				}

				#ifdef COMPILE_WITH_CUDA
				if (!USE_CHUNK) {
					//cout<< "USE_CUDA && NO_CHUNK" << endl;

					CDiffraction::get_atomicFormFactorList(&particle,&det);		

					//fmat F_hkl_sq(py,px);		
				 	float* F_mem = F_hkl_sq.memptr();
					float* f_mem = CDiffraction::f_hkl_list.memptr();
					float* q_mem = det.q_xyz.memptr();
					float* p_mem = particle.atomPos.memptr();
					cuda_structureFactor(F_mem, f_mem, q_mem, p_mem, det.numPix, particle.numAtoms);
		
					detector_intensity += (F_hkl_sq + Compton) % det.solidAngle % det.thomson * beam.get_photonsPerPulsePerArea();
			
				} else if (USE_CHUNK) {
					//cout<< "USE_CUDA && USE_CHUNK" << endl;
					int max_chunkSize = 100;
					int chunkSize = 0;

					//fmat F_hkl_sq(py,px); // F_hkl_sq: py x px
			 
					float* f_mem = CDiffraction::f_hkl.memptr(); // f_hkl: py x px x numAtomTypes
					float* q_mem = det.q_xyz.memptr(); // q_xyz: py x px x 3

					fmat pad_real;
					fmat pad_imag;
					fmat sumDr;
					sumDr.zeros(py*px,1);
					fmat sumDi;
					sumDi.zeros(py*px,1);
		
					int first_ind = 0;
					int last_ind = 0;
					while (first_ind < particle.numAtoms) {			 
						last_ind = min((last_ind + max_chunkSize),particle.numAtoms);

						chunkSize = last_ind-first_ind;

						pad_real.zeros(py*px,chunkSize);
					 	float* pad_real_mem = pad_real.memptr();

						pad_imag.zeros(py*px,chunkSize);
					 	float* pad_imag_mem = pad_imag.memptr();
					 	
						// xyzInd & pos are chunked
						// particle.xyzInd // 1 x chunk
						// particle.pos // chunk x 3
						irowvec xyzInd_sub = particle.xyzInd.subvec( first_ind,last_ind-1 );
						int* i_mem = xyzInd_sub.memptr();	
						fmat pos_sub = particle.atomPos( span(first_ind,last_ind-1), span::all );
						float* p_mem = pos_sub.memptr();

						cuda_structureFactorChunkParallel(pad_real_mem, pad_imag_mem, f_mem, q_mem, i_mem, p_mem, particle.numAtomTypes, det.numPix, chunkSize);

						sumDr += sum(pad_real,1);
						sumDi += sum(pad_imag,1);
			
						first_ind += max_chunkSize;
					}
		
					sumDr.reshape(py,px);
					sumDi.reshape(py,px);
					F_hkl_sq = sumDr % sumDr + sumDi % sumDi;

					detector_intensity += (F_hkl_sq + Compton) % det.solidAngle % det.thomson * beam.get_photonsPerPulsePerArea();
				}
				#else
				cout<< "USE_CPU" << endl;
				CDiffraction::get_atomicFormFactorList(&particle, &det);

				F_hkl_sq = CDiffraction::calculate_intensity(&particle, &det);
				
				photon_field = (F_hkl_sq + Compton) % det.solidAngle % det.thomson * beam.get_photonsPerPulsePerArea();
cout << "F_hkl_sq:" << F_hkl_sq(30,30) << endl;
cout << "Compton:" << Compton(40,40) << endl;
cout << "photons:" << beam.get_photonsPerPulsePerArea() << endl;
				detector_intensity += photon_field;
				#endif
				if (saveSlices) {
					int createSubgroup;
					if (isFirstSlice == 1) {
						createSubgroup = 1;
					} else {
						createSubgroup = 0;
					}
					std::stringstream sstm0;
		  			sstm0 << "/misc/photonField/photonField_" << setfill('0') << setw(7) << timeSlice;
					string fieldName = sstm0.str();
					cout << fieldName << endl;
					int success = hdf5writeVector(outputName, "misc", "/misc/photonField", fieldName, photon_field, createSubgroup);
				}
				isFirstSlice = 0;
			}// end timeSlice

			// Apply badpixelmap
			CDetector::apply_badPixels(&detector_intensity);
			// Poisson noise
			detector_counts = CToolbox::convert_to_poisson(&detector_intensity);
			
			// Save to HDF5
			int createSubgroup = 0;
			int success = hdf5writeVector(outputName,"data","","/data/data", detector_counts, createSubgroup); // FIXME: groupname and subgroupname are redundant
			success = hdf5writeVector(outputName,"data","","/data/diffr", detector_intensity, createSubgroup);
			createSubgroup = 0;
			fvec angle = quaternion;
			success = hdf5writeVector(outputName,"data","","/data/angle", angle,createSubgroup);
			createSubgroup = 1;
			double dist = det.get_detector_dist();
			success = hdf5writeScalar(outputName,"params","params/geom","/params/geom/detectorDist", dist,createSubgroup);
			createSubgroup = 0;
			double pixelWidth = det.get_pix_width();
			success = hdf5writeScalar(outputName,"params","params/geom","/params/geom/pixelWidth", pixelWidth,createSubgroup);
			double pixelHeight = det.get_pix_height();
			success = hdf5writeScalar(outputName,"params","params/geom","/params/geom/pixelHeight", pixelHeight,createSubgroup);
			fmat mask = ones<fmat>(px_in,px_in);
			success = hdf5writeVector(outputName,"params","params/geom","/params/geom/mask", mask,createSubgroup);
			createSubgroup = 1;
			double photonEnergy = beam.get_photon_energy();
			success = hdf5writeScalar(outputName,"params","params/beam","/params/beam/photonEnergy", photonEnergy,createSubgroup);
			createSubgroup = 0;
			double photons = total_phot;
			success = hdf5writeScalar(outputName,"params","params/beam","/params/beam/photons", photons,createSubgroup);			
			createSubgroup = 0;
			double focusArea = beam.get_focus_area();
			success = hdf5writeScalar(outputName,"params","params/beam","/params/beam/focusArea", focusArea,createSubgroup);

			if (comm->rank() == 1) {
				double thetaMax = atan((px/2*pix_height)/d);
				double qmax = 2*sin(thetaMax/2)/beam.get_wavelength();
				double dmin = 1/(2*qmax);
				cout << "px, pix_height, d, thetaMax: " << px << "," << pix_height << "," << d << "," << thetaMax << endl;
				cout << "wavelength: " << beam.get_wavelength() << endl;
				cout << "max q to the edge: " << qmax * 1e-10 << " A^-1" << endl;
				cout << "Half period resolution: " << dmin * 1e10 << " Angstroms" << endl;
			}

			float msgDone = 0; //std::vector<float> msgDone;
    		comm->send(master, DONETAG, msgDone);
    		
			cout << "DP took: " << timer.toc() <<" seconds."<<endl;
    	}
		
		if (status.tag() == DIETAG) {
			return;
		}
	} // end of while
}

