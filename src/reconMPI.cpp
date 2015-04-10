/*
 * Program for merging diffraction patterns based on maximum likelihood
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
#include <limits>
// Armadillo library
#include <armadillo>
// Boost library
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/program_options.hpp>
// SingFEL library
#include "detector.h"
#include "beam.h"
#include "particle.h"
#include "diffraction.h"
#include "toolbox.h"
#include "diffraction.cuh"
#include "io.h"

namespace mpi = boost::mpi;
namespace opt = boost::program_options;
using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

const int master = 0;
const int msgLength = 2000;
#define USE_CUDA 0

#define MODELTAG 1	// mySlices matrix
#define DPTAG 2	// diffraction pattern
#define DIETAG 3 // die signal
#define SAVESLICESTAG 4 // save slices signal
#define SAVELSETAG 5 // save LSE signal
#define GOODPIXTAG 6 // goodpixelmap
#define DONETAG 7 // done signal

#define CHUNKTAG 8
#define PROBTAG 9
static void master_recon(mpi::communicator* comm, opt::variables_map vm, \
                         fcube* myRot, fmat* pix, uvec* goodpix, float pix_max,\
                         fcube* myIntensity, fcube* myWeight, int numImages, \
                         int numSlices, int iter);

static void slave_recon(mpi::communicator* comm, opt::variables_map vm, \
                        int iter);

opt::variables_map parse_input(int argc, char* argv[], mpi::communicator* comm);

static int expansion(opt::variables_map vm, arma::fcube* myRot, arma::fmat* pix, arma::uvec* goodpix, float pix_max, arma::fcube* myIntensity, int numSlices, int iter);

static int maximization(mpi::communicator* comm, opt::variables_map vm, int numSlaves, arma::uvec* goodpix, int numProcesses, int numCandidates, int numImages, int numSlices, int iter);

static int compression(opt::variables_map vm, fcube* myIntensity, fcube* myWeight, fmat* pix, float pix_max, fcube* myRot, int numSlices, int iter);

static int saveDiffractionVolume(opt::variables_map vm, fcube* myIntensity, fcube* myWeight, int iter);

uvec numJobsPerSlave(int numImages, int numSlaves);

void normalizeCondProb(fvec* condProb, int numCandidates, fvec* normCondProb, uvec* candidatesInd);

void updateExpansionSlice(opt::variables_map vm, fmat* updatedSlice, uvec* goodpix, fvec* normCondProb, uvec* candidatesInd);

void sendJobsToSlaves(boost::mpi::communicator* comm, int numProcesses, uvec* numJobsForEachSlave, int expansionInd);

void receiveProbsFromSlaves(boost::mpi::communicator* comm, int numProcesses, uvec* numJobsForEachSlave, fvec* condProb);

void saveCondProb2File(opt::variables_map vm, int iter, int expansionInd, fvec* myProb);

void saveExpansionUpdate(opt::variables_map vm, int iter, int expansionInd, fmat* updatedSlice);

void loadUpdatedExpansion(opt::variables_map vm, int iter, int sliceInd, fcube* myDPnPixmap);

void loadDP(opt::variables_map vm, int ind, fmat* myDP);

void load_readNthLine(opt::variables_map vm, int N, fmat* img);

int main( int argc, char* argv[] ){

	wall_clock timerMaster;

	// Initialize MPI
  	mpi::environment env;
  	mpi::communicator world;
	mpi::communicator* comm = &world;

	// All processes parse the input
	opt::variables_map vm = parse_input(argc, argv, comm);
	// Set random seed
	srand( world.rank() );

	string input = vm["input"].as<string>();
	string beamFile = vm["beamFile"].as<string>();
	string geomFile = vm["geomFile"].as<string>();
	int volDim = vm["volDim"].as<int>();
	string format = vm["format"].as<string>();
	string initialVolume = vm["initialVolume"].as<string>();
	string rotationAxis = vm["rotationAxis"].as<string>();
	int startIter = vm["startIter"].as<int>();
	int numIterations = vm["numIterations"].as<int>();
	string numImagesStr = vm["numImages"].as<string>();
	ivec numImages = str2ivec(numImagesStr);
	string numSlicesStr = vm["numSlices"].as<string>();
	ivec numSlices = str2ivec(numSlicesStr);
	string hdfField = vm["hdfField"].as<string>();

	// Check the input makes sense
	int numSlaves = comm->size()-1;
	int maxNumImages = max(numImages);
	if (maxNumImages < numSlaves) {
		cerr << "Number of workers too large for this task" << endl;
		return 0;
	}
	//TODO: check length of numImages == numSlices
	//TODO: check startIter is less than length numImages
	//TODO: check startIter+numIterations is less than length numImages
	
	fcube myIntensity, myWeight;
	fmat pix;
	float pix_max;
	uvec goodpix;
	
	if (world.rank() == master) {	
		/****** Beam ******/
		// Let's read in our beam file
		double photon_energy = 0;
		double focus_radius = 0;
		double fluence = 0;
		string line;
		ifstream myFile(beamFile.c_str());
		while (getline(myFile, line)) {
			if (line.compare(0,1,"#") && line.compare(0,1,";") && line.length() > 0) {
				// line now contains a valid input
				cout << line << endl;
				typedef boost::tokenizer<boost::char_separator<char> > Tok;
				boost::char_separator<char> sep(" ="); // default constructed
				Tok tok(line, sep);
				for(Tok::iterator tok_iter = tok.begin(); tok_iter != tok.end(); ++tok_iter){
				    if ( boost::algorithm::iequals(*tok_iter,"beam/photon_energy") ) {            
				        string temp = *++tok_iter;
				        photon_energy = atof(temp.c_str()); // photon energy to wavelength
				        break;
				    } else if ( boost::algorithm::iequals(*tok_iter,"beam/fluence") ) {            
				        string temp = *++tok_iter;
				        fluence = atof(temp.c_str()); // number of photons per pulse
				        break;
				    } else if ( boost::algorithm::iequals(*tok_iter,"beam/radius") ) {            
				        string temp = *++tok_iter;
				        focus_radius = atof(temp.c_str()); // focus radius
				        break;
				    }
				}
			}
		}
		CBeam beam = CBeam();
		beam.set_photon_energy(photon_energy);

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
		det.init_dp(&beam);
		pix = det.pixSpace;
		pix_max = det.pixSpaceMax;
		goodpix = det.get_goodPixelMap();

		double theta = atan((px/2*pix_height)/d);
		double qmax = 2/beam.get_wavelength()*sin(theta/2);
		double dmin = 1/(2*qmax);
		if (world.rank() == 0) {
			cout << "max q to the edge: " << qmax*1e-10 << " A^-1" << endl;
			cout << "Half period resolution: " << dmin*1e10 << " A" << endl;
		}
	
		// Only Master initializes intensity volume
		string filename;
  		fmat myDP(py,px);
	  	fmat myR;
	  	myR.zeros(3,3);
		myWeight.zeros(volDim,volDim,volDim);
		myIntensity.zeros(volDim,volDim,volDim);

		timerMaster.tic();
		
		int active;
		string interpolate = "linear";
		
		fmat rot3D(3,3);
		fvec u(3);
		fvec quaternion(4);
	
		std::ifstream infile;
		if (format == "list") {
			cout << "Using image list: " << input << endl;
			infile.open(input.c_str());
		}
	
			if ( strcmp(initialVolume.c_str(),"randomMerge")==0 ) {
				cout << "Randomly merging diffraction volume..." << endl;
				// Setup initial diffraction volume by merging randomly
				// rotationAxis determines the random nature of the angles
				for (int r = 0; r < numImages[startIter]; r++) {
				  	// Get image
				  	if (format == "S2E") {
				  		std::stringstream sstm;
				  		sstm << input << "/diffr/diffr_out_" << setfill('0') << setw(7) << r+1 << ".h5";
						filename = sstm.str();
						myDP = hdf5readT<fmat>(filename,hdfField);
				  	} else if (format == "list") {
				  		std::getline(infile, line);
				  		myDP = load_asciiImage(line);
				  	} else {
					  	std::stringstream sstm;
				  		sstm << input << "/diffr/diffr_out_" << setfill('0') << setw(7) << r+1 << ".dat";
						filename = sstm.str();
						myDP = load_asciiImage(filename);
					}
			
					// Get rotation matrix
				  	quaternion = CToolbox::getRandomRotation(rotationAxis);
			
					myR = CToolbox::quaternion2rot3D(quaternion);
					active = 1;
					CToolbox::merge3D(&myDP, &pix, &goodpix, &myR, pix_max, &myIntensity, &myWeight, active, interpolate);
			  	}
				// Normalize here
				CToolbox::normalize(&myIntensity,&myWeight);
		  	} else { // Load pre-existing diffraction volume
				cout << "Loading diffraction volume..." << endl;
				for (int i = 0; i < volDim; i++) {
					std::stringstream sstm;
					sstm << initialVolume << "/vol_" << setfill('0') << setw(7) << i << ".dat";
					string outputName = sstm.str();
					myIntensity.slice(i) = load_asciiImage(outputName);
				}
			} // end of intial diffraction volume
	} // end of master

	world.barrier();
	if (world.rank() == master) {
		cout << "Initialization time: " << timerMaster.toc() <<" seconds."<<endl;
	}
	
	// Main iteration
	fmat myQuaternions;
	fmat myR;

	for (int iter = startIter; iter < startIter+numIterations; iter++) { // number of iterations
		if (world.rank() == master) {
			int numImagesNow = numImages[iter];
			int numSlicesNow = numSlices[iter];
			fcube myRot(3,3,numSlicesNow);
			cout << "***ITER " << iter << "***" << endl;
			if (iter == startIter) {
				// Equal distribution of quaternions
				if (rotationAxis == "y" || rotationAxis == "z") {
					myQuaternions = CToolbox::pointsOn1Sphere(numSlicesNow, rotationAxis);
				} else {
  					myQuaternions = CToolbox::pointsOn4Sphere(numSlicesNow);
  				}
				for (int i = 0; i < numSlicesNow; i++) {
					myR = CToolbox::quaternion2rot3D(trans(myQuaternions.row(i)));
					myRot.slice(i) = myR;
				}
  			}
			master_recon(comm, vm, &myRot, &pix, &goodpix, pix_max, &myIntensity, &myWeight, numImagesNow, numSlicesNow, iter);
		} else {
			slave_recon(comm, vm, iter);
		}
		world.barrier();
	}
  	return 0;
}

static void master_recon(mpi::communicator* comm, opt::variables_map vm, fcube* myRot, fmat* pix, uvec* goodpix, float pix_max, fcube* myIntensity, fcube* myWeight, int numImages, int numSlices, int iter){

	wall_clock timerMaster;

  	int rank, numProcesses, numSlaves;
  	fvec quaternion;

	numProcesses = comm->size();
	numSlaves = numProcesses-1;

	// ########### EXPANSION ##############
	cout << "Start expansion" << endl;
	timerMaster.tic();

	expansion(vm, myRot, pix, goodpix, pix_max, myIntensity, numSlices, iter);
	
	cout << "Expansion time: " << timerMaster.toc() <<" seconds."<<endl;

	// ########### MAXIMIZATION ##############	
	cout << "Start maximization" << endl;
	timerMaster.tic();

	//////////////////////
	// Number of data candidates to update expansion slice
	int numCandidates = 2;
	//////////////////////
	maximization(comm, vm, numSlaves, goodpix, numProcesses, numCandidates, numImages, numSlices, iter);

	cout << "Maximization time: " << timerMaster.toc() <<" seconds."<<endl;

	// ########### COMPRESSION ##############
	cout << "Start compression" << endl;
	timerMaster.tic();

	compression(vm, myIntensity, myWeight, pix, pix_max, myRot, numSlices, iter);

	cout << "Compression time: " << timerMaster.toc() <<" seconds."<<endl;
	
	// ########### Save diffraction volume ##############
	cout << "Saving diffraction volume..." << endl;
	timerMaster.tic();

	saveDiffractionVolume(vm, myIntensity, myWeight, iter);

	cout << "Save time: " << timerMaster.toc() <<" seconds."<<endl;

	// ########### Reset workers ##############
  	// Tell all the slaves to exit by sending an empty message with the DIETAG.
  	float msg1[1];
	for (rank = 1; rank < numProcesses; ++rank) {
		comm->send(rank, DIETAG, msg1, 1);
	}

	cout << "Done iteration: " << iter << endl;
}

static void slave_recon(mpi::communicator* comm, opt::variables_map vm, int iter) {
//wall_clock timer;	
	int volDim = vm["volDim"].as<int>();
	string output = vm["output"].as<string>();
	int useFileList = vm["useFileList"].as<int>();
	string input = vm["input"].as<string>();
	string format = vm["format"].as<string>();
	string hdfField = vm["hdfField"].as<string>();
	fvec gaussianStdDev;
	bool useGaussianProb = false;
	if (vm.count("gaussianStdDev")) {
		useGaussianProb = true;
		string gaussianStdDevStr = vm["gaussianStdDev"].as<string>();
		gaussianStdDev = str2fvec(gaussianStdDevStr);
	}
	int numChunkData = 0;
	boost::mpi::status status;
	// Expansion related variables
	fvec quaternion(4);
	// Maximization related variables
	fmat diffraction = zeros<fmat>(volDim,volDim);
	fvec condProb; // conditional probability
	fmat imgRep;
	uvec goodpixmap;
	fmat myDP;
	float msg[3];
	while (1) {

		// Receive a message from the master
		/////////////////////////////////////////////////////////
    	status = comm->recv(master, boost::mpi::any_tag, msg, 3);
		/////////////////////////////////////////////////////////   		

    	// Calculate least squared error
    	if (status.tag() == DPTAG) {
    		int startInd = (int) msg[0];
    		int endInd = (int) msg[1];
    		int expansionInd = (int) msg[2];
			numChunkData = endInd - startInd + 1; // number of measured data to process

			// Initialize
	    	condProb.zeros(numChunkData);

			//TODO: Time reading of expansion slice (all processes going for the same file)
	    	//////////////////////////
	    	// Read in expansion slice
	    	//////////////////////////
			// Get expansion image
			std::stringstream sstm;
			sstm << output << "/expansion/iter" << iter << "/expansion_" << setfill('0') << setw(7) << expansionInd << ".dat";
			string filename = sstm.str();
			fmat myExpansionSlice = load_asciiImage(filename);
			// Get expansion pixmap
			std::stringstream sstm1;
			sstm1 << output << "/expansion/iter" << iter << "/expansionPixmap_" << setfill('0') << setw(7) << expansionInd << ".dat";
			string filename1 = sstm1.str();
			fmat myPixmap = load_asciiImage(filename1);
		   
			//TODO: Time reading of data
	    	///////////////
	    	// Read in data
	    	///////////////
			string line;
			int counter = 0;
	    	for (int i = startInd; i <= endInd; i++) {
				//Read in measured diffraction data
				if (format == "S2E") {
					loadDP(vm, i+1, &myDP);
			  	} else if (format == "list") { // TODO: this needs testing
				  	load_readNthLine(vm, i, &myDP);
			  	} else {
				  	std::stringstream sstm;
			  		sstm << input << "/diffr/diffr_out_" << setfill('0') << setw(7) << i+1 << ".dat";
					filename = sstm.str();
					myDP = load_asciiImage(filename);
				}

				/////////////////////////////////////////////////////
				// Compare measured diffraction with expansion slices
				/////////////////////////////////////////////////////
				double val = 0.0;
				if (useGaussianProb) {
					val = CToolbox::calculateGaussianSimilarity(&myExpansionSlice, &myDP, &myPixmap, gaussianStdDev(iter));
				}
				
				condProb(counter) = float(val);
				counter++;
			}
			// Send back conditional probability to master
			float* msgProb = &condProb[0];
			/////////////////////////////////////////////////////
			comm->send(master, PROBTAG, msgProb, numChunkData); // send probability in chunk size
			/////////////////////////////////////////////////////
		}

		if (status.tag() == DIETAG) {
		  	return;
		}

	} // end of while
} // end of slave_recon

//TODO: Move expansion, maximization and compression to reconMPI.cpp (remove #define TAGs)
// Given a diffraction volume (myIntensity) and save 2D slices (numSlices)
int expansion(opt::variables_map vm, fcube* myRot, fmat* pix, uvec* goodpix, float pix_max, fcube* myIntensity, int numSlices, int iter) {

	int volDim = vm["volDim"].as<int>();
	string output = vm["output"].as<string>();

	int active = 1;
	string interpolate = "linear";
	fmat myR;
	myR.zeros(3,3);
	fcube myDPnPixmap; 	// first slice: diffraction pattern
						// second slice: good pixel map
	
	// Slice diffraction volume and save to file
	for (int i = 0; i < numSlices; i++) {
		myDPnPixmap.zeros(volDim,volDim,2);
		// Get rotation matrix
		myR = myRot->slice(i);
		CToolbox::slice3D(&myDPnPixmap, pix, goodpix, &myR, pix_max, myIntensity, active, interpolate);
		
		// Save expansion slice to disk
		std::stringstream sstm;
		sstm << output << "/expansion/iter" << iter << "/expansion_" << setfill('0') << setw(7) << i << ".dat";
		string outputName = sstm.str();
		myDPnPixmap.slice(0).save(outputName,raw_ascii);
		std::stringstream sstm1;
		sstm1 << output << "/expansion/iter" << iter << "/expansionPixmap_" << setfill('0') << setw(7) << i << ".dat";
		string outputName1 = sstm1.str();
		myDPnPixmap.slice(1).save(outputName1,raw_ascii);
	}

	return 0;
}

// Maximization
// goodpix: good pixel on a detector (used for beamstops and gaps)
int maximization(boost::mpi::communicator* comm, opt::variables_map vm, int numSlaves, uvec* goodpix, int numProcesses, int numCandidates, int numImages, int numSlices, int iter) {

	wall_clock timer;

	bool useGaussianProb = false;
	if (vm.count("gaussianStdDev")) {
		useGaussianProb = true;
	}
	bool saveCondProb = false;
	if (vm.count("saveCondProb")) {
		saveCondProb = vm["saveCondProb"].as<bool>();
	}
	
	uvec numJobsForEachSlave(numSlaves);
	fvec myProb(numImages);
	fvec normCondProb;
	uvec candidatesInd;
	fmat updatedSlice;
		
	// Calculate number jobs for each slave
	numJobsForEachSlave = numJobsPerSlave(numImages, numSlaves);
	
cout << "max msgProb length: " << max(numJobsForEachSlave) << endl;
	
	// Loop through all expansion slices and compare all measured data
	for (int expansionInd = 0; expansionInd < numSlices; expansionInd++) {
cout << "expansionInd: " << expansionInd << endl;
		// For each slice, each worker get a subset of measured data
		sendJobsToSlaves(comm, numProcesses, &numJobsForEachSlave, expansionInd);

//timer.tic();
		//TODO: Time accumulation. May need to write random access accumulator
		// Accumulate conditional probabilities for each expansion slice
		receiveProbsFromSlaves(comm, numProcesses, &numJobsForEachSlave, &myProb);
		
//cout << "Master: Received all probtags " <<timer.toc()<<endl;

		if (saveCondProb) {
			saveCondProb2File(vm, iter, expansionInd, &myProb);
		}
//timer.tic();
		//TODO: Time generating new weighted expansion slice
		// Pick top candidates
cout << "sort start" << endl;
		
		if (useGaussianProb) {
			normalizeCondProb(&myProb, numCandidates, &normCondProb, &candidatesInd);
		}
cout << "normVal: " << normCondProb << endl;
		// Update expansion slice
		updateExpansionSlice(vm, &updatedSlice, goodpix, &normCondProb, &candidatesInd);
		
//cout << "Master generates new expansion slice " <<timer.toc()<<endl;
//timer.tic();
		saveExpansionUpdate(vm, iter, expansionInd, &updatedSlice);

//cout << "Master save expansion slice " <<timer.toc()<<endl;
	}
	return 0;
}

// Compression
int compression(opt::variables_map vm, fcube* myIntensity, fcube* myWeight, fmat* pix, float pix_max, fcube* myRot, int numSlices, int iter) {

	int volDim = vm["volDim"].as<int>();
	string output = vm["output"].as<string>();
	string format = vm["format"].as<string>();

	myWeight->zeros(volDim,volDim,volDim);
	myIntensity->zeros(volDim,volDim,volDim);
	int active = 1;
	string interpolate = "linear";
	fcube myDPnPixmap; 	// first slice: diffraction pattern
						// second slice: good pixel map
	fmat myR;
	for (int sliceInd = 0; sliceInd < numSlices; sliceInd++) {
		// Get updated expansion slice
		loadUpdatedExpansion(vm, iter, sliceInd, &myDPnPixmap);
		// Get rotation matrix
		myR = myRot->slice(sliceInd);
		// Merge into 3D diffraction volume
		CToolbox::merge3D(&myDPnPixmap, pix, &myR, pix_max, myIntensity, myWeight, active, interpolate);
	}
	// Normalize here
	CToolbox::normalize(myIntensity, myWeight);
	return 0;
}

int saveDiffractionVolume(opt::variables_map vm, fcube* myIntensity, fcube* myWeight, int iter) {

	int volDim = vm["volDim"].as<int>();
	string output = vm["output"].as<string>();

	for (int i = 0; i < volDim; i++) {
		std::stringstream sstm;
		sstm << output << "/compression/iter" << iter << "/vol_" << setfill('0') << setw(7) << i << ".dat";
		string outputName = sstm.str();
		myIntensity->slice(i).save(outputName,raw_ascii);
		// Temporary
		std::stringstream sstm1;
		sstm1 << output << "/compression/iter" << iter << "/volWeight_" << setfill('0') << setw(7) << i << ".dat";
		string outputName1 = sstm1.str();
		myWeight->slice(i).save(outputName1,raw_ascii);
	}
	return 0;
}

uvec numJobsPerSlave(int numImages, int numSlaves) {
	int dataPerSlave = floor( (float) numImages / (float) numSlaves );
	int leftOver = numImages - dataPerSlave * numSlaves;
	uvec numJobsForEachSlave(numSlaves);
	numJobsForEachSlave.fill(dataPerSlave);
	for (int i = 0; i < numSlaves; i++) {
		if (leftOver > 0) {
			numJobsForEachSlave(i) += 1;
			leftOver--;
		}
	}
	return numJobsForEachSlave;
}

void updateExpansionSlice(opt::variables_map vm, fmat* updatedSlice, uvec* goodpix, fvec* normCondProb, uvec* candidatesInd) {
	string input = vm["input"].as<string>();
	string format = vm["format"].as<string>();
	string hdfField = vm["hdfField"].as<string>();
	int volDim = vm["volDim"].as<int>();
	
	fmat& _updatedSlice = updatedSlice[0];
	fvec& _normCondProb = normCondProb[0];
	uvec& _candidatesInd = candidatesInd[0];
	
	int numCandidates = _candidatesInd.n_elem;
	_updatedSlice.zeros(volDim,volDim);
	fmat myDP2;
	string filename;
	std::stringstream sstm;
	// Setup goodpixmap
	uvec::iterator goodBegin = goodpix->begin();
	uvec::iterator goodEnd = goodpix->end();
	
	// Load measured diffraction pattern from file
	if (format == "S2E") {
		for (int i = 0; i < numCandidates; i++) {
			loadDP(vm, _candidatesInd(i)+1, &myDP2);
			// Weighted mean
			for(uvec::iterator p=goodBegin; p!=goodEnd; ++p) {
				_updatedSlice(*p) += _normCondProb(i) * myDP2(*p);
			}
		}
	} else if (format == "list") { //TODO: this needs testing
		for (int i = 0; i < numCandidates; i++) {
			myDP2.zeros(volDim,volDim);
			load_readNthLine(vm, i, &myDP2);
			// Weighted mean
			for(uvec::iterator p=goodBegin; p!=goodEnd; ++p) {
				_updatedSlice(*p) += _normCondProb(i) * myDP2(*p);
			}
		}
	}
	cout << "calculating weighted slice" << endl;

	cout << "done weighted slice" << endl;
}

void normalizeCondProb(fvec* condProb, int numCandidates, fvec* normCondProb, uvec* candidatesInd) {
	fvec& _condProb = condProb[0];
	fvec& _normCondProb = normCondProb[0];
	uvec& _candidatesInd = candidatesInd[0];
	
	_normCondProb.zeros(numCandidates);
	_candidatesInd.zeros(numCandidates);
	uvec indices = sort_index(_condProb,"descend");
	_candidatesInd = indices.subvec(0,numCandidates-1);
	// Calculate norm cond prob
	fvec candidatesVal(numCandidates);
	// fill with smallest floating value to avoid dividing by zero
	candidatesVal.fill(std::numeric_limits<float>::min());
	for (int i = 0; i < numCandidates; i++) {
		candidatesVal(i) = _condProb(_candidatesInd(i));
	}
	_normCondProb = candidatesVal / sum(candidatesVal);
}

////////////////////////////////////////////
// Send jobs to slaves
// 1) Start index of measured diffraction pattern
// 2) End index of measured diffraction pattern
// 3) Index of expansion slice
////////////////////////////////////////////
void sendJobsToSlaves(boost::mpi::communicator* comm, int numProcesses, uvec* numJobsForEachSlave, int expansionInd) {
	uvec& _numJobsForEachSlave = numJobsForEachSlave[0];
	
	int startInd = 0;
	int endInd = 0;
	for (int rank = 1; rank < numProcesses; ++rank) {
		endInd = startInd + _numJobsForEachSlave(rank-1) - 1;
		fvec id;
		id << startInd << endInd << expansionInd << endr;
		float* id1 = &id[0];
		////////////////////////////////
		comm->send(rank, DPTAG, id1, 3); // send to slave
		////////////////////////////////
		startInd += _numJobsForEachSlave(rank-1);
	}
}

void receiveProbsFromSlaves(boost::mpi::communicator* comm, int numProcesses, uvec* numJobsForEachSlave, fvec* condProb) {
	uvec& _numJobsForEachSlave = numJobsForEachSlave[0];
	fvec& _condProb = condProb[0];
	
	int currentInd = 0;
	int numElements;
	float msgProb[max(_numJobsForEachSlave)];
	for (int rank = 1; rank < numProcesses; ++rank) {
		numElements = _numJobsForEachSlave(rank-1);
		///////////////////////////////////////////////////////
		comm->recv(rank, PROBTAG, msgProb, numElements); // receive from slave
		///////////////////////////////////////////////////////
		for (int i = 0; i < numElements; i++) {
			_condProb(currentInd) = msgProb[i];
			currentInd++;
		}
	}
	cout << "currentInd: " << currentInd << endl;
}

void saveCondProb2File(opt::variables_map vm, int iter, int expansionInd, fvec* myProb) {
	string output = vm["output"].as<string>();

	string outputName;
	stringstream sstm;
	sstm << output << "/maximization/iter" << iter << "/similarity_" << setfill('0') << setw(7) << expansionInd << ".dat";
	outputName = sstm.str();
	myProb->save(outputName,raw_ascii);
}

void saveExpansionUpdate(opt::variables_map vm, int iter, int expansionInd, fmat* updatedSlice) {
	string output = vm["output"].as<string>();
		
	string filename;
	// Save updated expansion slices
	std::stringstream sstm;
	sstm << output << "/expansion/iter" << iter << "/expansionUpdate_" << setfill('0') << setw(7) << expansionInd << ".dat";
	filename = sstm.str();
	updatedSlice->save(filename,raw_ascii);
}

void loadUpdatedExpansion(opt::variables_map vm, int iter, int sliceInd, fcube* myDPnPixmap) {
	string output = vm["output"].as<string>();
	int volDim = vm["volDim"].as<int>();
	myDPnPixmap->zeros(volDim,volDim,2);
		
	// Get image
	std::stringstream ss;
	ss << output << "/expansion/iter" << iter << "/expansionUpdate_" << setfill('0') << setw(7) << sliceInd << ".dat";
	string filename = ss.str();
	myDPnPixmap->slice(0) = load_asciiImage(filename);
	// Get goodpixmap
	ss.str("");
	ss << output << "/badpixelmap.dat";
	filename = ss.str();
	fmat pixmap = load_asciiImage(filename); // load badpixmap
	myDPnPixmap->slice(1) = CToolbox::badpixmap2goodpixmap(pixmap); // goodpixmap
}

void loadDP(opt::variables_map vm, int ind, fmat* myDP) {
	string input = vm["input"].as<string>();
	string hdfField = vm["hdfField"].as<string>();
	int volDim = vm["volDim"].as<int>();
	fmat& _myDP = myDP[0];
	
	_myDP.zeros(volDim,volDim);
	std::stringstream ss;
	ss.str("");
	ss << input << "/diffr/diffr_out_" << setfill('0') << setw(7) << ind << ".h5";
	string filename = ss.str();
	// Read in diffraction				
	_myDP = hdf5readT<fmat>(filename,hdfField);
}

void load_readNthLine(opt::variables_map vm, int N, fmat* img) {
	string input = vm["input"].as<string>();
	int volDim = vm["volDim"].as<int>();
	fmat& _img = img[0];

	_img.zeros(volDim,volDim);
	std::ifstream infile;
	infile.open(input.c_str());
	string line;
	// skip N lines
	for (int r = 0; r <= N; r++) { //reads Nth line of the file
		std::getline(infile, line);
	}
	_img = load_asciiImage(line);

}


opt::variables_map parse_input( int argc, char* argv[], mpi::communicator* comm ) {

    // Constructing an options describing variable and giving it a
    // textual description "All options"
    opt::options_description desc("All options");

    // When we are adding options, first parameter is a name
    // to be used in command line. Second parameter is a type
    // of that option, wrapped in value<> class. Third parameter
    // must be a short description of that option
    desc.add_options()
        ("input", opt::value<std::string>(), "Input directory for finding /pmi and /diffr")
        ("output", opt::value<string>(), "Output directory for saving /expansion, /maximization, /compression")
        ("useFileList", opt::value<int>()->default_value(0), "Input a list containing filenames of diffraction patterns")
        ("beamFile", opt::value<string>(), "Beam file defining X-ray beam")
        ("geomFile", opt::value<string>(), "Geometry file defining diffraction geometry")
		("rotationAxis", opt::value<string>()->default_value("xyz"), "Euler rotation convention")
        ("numIterations", opt::value<int>(), "Number of iterations to perform from startIter")
        ("numImages", opt::value<string>(), "Number of measured diffraction patterns (Comma separated list)")
        ("numSlices", opt::value<string>(), "Number of Ewald slices in the expansion step (Comma separated list)")
        ("volDim", opt::value<int>(), "Number of pixel along one dimension")
        ("startIter", opt::value<int>()->default_value(0), "Start iteration number used to index 2 vectors: numImages and numSlices (count from 0)")
        ("initialVolume", opt::value<string>()->default_value("randomMerge"), "Absolute path to initial volume")
        ("format", opt::value<string>(), "Defines file format to use")
        ("hdfField", opt::value<string>()->default_value("/data/data"), "Data field to use for reconstruction")
        ("gaussianStdDev", opt::value<string>(), "Use Gaussian likelihood for maximization with the following standard deviations (Comma separated list)")
        ("saveCondProb", opt::value<bool>(), "Optionally save conditional probabilities")
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
		if (vm.count("input"))
    		cout << "input: " << vm["input"].as<string>() << endl;
		//TODO: print all parameters
	}

	return vm;
} // end of parse_input
