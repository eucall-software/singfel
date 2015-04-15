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
#include <assert.h>
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
const int lenDPTAG = 4;

#define MODELTAG 1	// mySlices matrix
#define DPTAG 2	// diffraction pattern
#define DIETAG 3 // die signal
#define SAVESLICESTAG 4 // save slices signal
#define SAVELSETAG 5 // save LSE signal
#define GOODPIXTAG 6 // goodpixelmap
#define DONETAG 7 // done signal

#define CHUNKTAG 8 // TODO: Clean up unused TAGs
#define PROBTAG 9
static void master_recon(mpi::communicator* comm, opt::variables_map vm, \
                         fcube* myRot, CDetector* det, fcube* myIntensity, \
                         fcube* myWeight, int numImages, int numSlices, \
                         int iter);
static void slave_recon(mpi::communicator* comm, opt::variables_map vm);
opt::variables_map parse_input(int argc, char* argv[], mpi::communicator* comm);
static int expansion(opt::variables_map vm, arma::fcube* myRot, \
                     arma::fcube* myIntensity, fcube* myWeight, CDetector* det,\
                     int numSlices, int iter);
static int maximization(mpi::communicator* comm, opt::variables_map vm, \
                        CDetector* det, int numSlaves, int numProcesses, \
                        int numCandidates, int numImages, \
                        int numSlices, int iter);
static int compression(opt::variables_map vm, fcube* myIntensity, \
                       fcube* myWeight, CDetector* det, fcube* myRot,\
                       int numSlices, int iter);
static int saveDiffractionVolume(opt::variables_map vm, fcube* myIntensity, \
                                 fcube* myWeight, int iter);
uvec numJobsPerSlave(int numImages, int numSlaves);
void normalizeCondProb(fvec* condProb, int numCandidates, fvec* normCondProb, \
                       uvec* candidatesInd);
void updateExpansionSlice(opt::variables_map vm, fcube* updatedSlice_Pixmap, \
                          fvec* normCondProb, \
                          uvec* candidatesInd);
void sendJobsToSlaves(boost::mpi::communicator* comm, int numProcesses, \
                      uvec* numJobsForEachSlave, int expansionInd, int iter);
void receiveProbsFromSlaves(boost::mpi::communicator* comm, int numProcesses, \
                            uvec* numJobsForEachSlave, fvec* condProb);
void saveCondProb2File(opt::variables_map vm, int iter, int expansionInd, \
                       fvec* myProb);
void loadCondProb(opt::variables_map vm, int iter, int expansionInd, fvec* myProb);
void saveExpansionSlice(opt::variables_map vm, fcube* myDPnPixmap, int iter, \
                        int ind);
void saveExpansionUpdate(opt::variables_map vm, fcube* updatedSlice_Pixmap, int iter, \
                         int expansionInd);
void loadExpansionSlice(opt::variables_map vm, int iter, int sliceInd, \
                        fcube* myDPnPixmap);
void loadUpdatedExpansion(opt::variables_map vm, int iter, int sliceInd, \
                          fcube* myDPnPixmap);
void loadDPnPixmap(opt::variables_map vm, int ind, fcube* myDPnPixmap);
void load_readNthLine(opt::variables_map vm, int N, fmat* img);
void calculateWeightedImage(float weight, fcube* updatedSlice, \
                            fcube* myDPnPixmap);
void getRotationMatrix(fmat* myR, fcube* myRot, int sliceInd);
void generateUniformRotations(string rotationAxis, int numSlicesNow, \
                              fcube* myRot);
void getBestCandidateProb(fvec* normCondProb, fvec* candidateProb, \
                           int expansionInd);
void getGoodSlicesIndex(fvec* candidateProb, float percentile, \
                        uvec* goodSlicesInd);
void saveBestCandidateProb(opt::variables_map vm, fvec* candidateProb, int iter);
void loadCandidateProb(opt::variables_map vm, int iter, fvec* candidateProb);
void loadInitVol(opt::variables_map vm, fcube* myIntensity, fcube* myWeight);

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
	string format = vm["format"].as<string>();
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
	CDetector det = CDetector();
	
	if (world.rank() == master) {	
		CBeam beam = CBeam();
		beam.readBeamFile(beamFile);
		det.readGeomFile(geomFile);

		// diffraction geometry needs the wavelength
		det.init_dp(&beam);
	
		// optionally display resolution
		CDiffraction::displayResolution(&det,&beam);
	} // end of master

	world.barrier();
	// Main iteration
	if (world.rank() == master) {
		for (int iter = startIter; iter < startIter+numIterations; iter++) {
			int numImagesNow = numImages[iter];
			int numSlicesNow = numSlices[iter];
			fcube myRot(3,3,numSlicesNow);
			cout << "***** ITER " << iter << " *****" << endl;
			generateUniformRotations(rotationAxis, numSlicesNow, &myRot);
			master_recon(comm, vm, &myRot, &det, &myIntensity, &myWeight, numImagesNow, numSlicesNow, iter);
		}
	} else {
		slave_recon(comm, vm); // slaves keep on running
	}
	
	float msg[1];
	if (world.rank() == master) {
		int numProcesses = world.size();
		for (int rank = 1; rank < numProcesses; ++rank) {
			comm->send(rank, DIETAG, msg, 1);
		}
	}
	
  	return 0;
}

static void master_recon(mpi::communicator* comm, opt::variables_map vm, fcube* myRot, CDetector* det, fcube* myIntensity, fcube* myWeight, int numImages, int numSlices, int iter){

	wall_clock timerMaster;

	string numCandidatesStr = vm["numCandidates"].as<string>();
	ivec numCandidates = str2ivec(numCandidatesStr);
	string justDo = "EMC";
	if (vm.count("justDo")) {
		justDo = vm["justDo"].as<string>();
		boost::to_upper(justDo);
	}

  	int numProcesses, numSlaves, status;
  	fvec quaternion;

	numProcesses = comm->size();
	numSlaves = numProcesses-1;

	// ########### EXPANSION ##############
	if (justDo == "EMC" || justDo == "E") {
		cout << "Start expansion" << endl;
		timerMaster.tic();
		status = expansion(vm, myRot, myIntensity, myWeight, det, numSlices, iter);
		assert(status==0);
		cout << "Expansion time: " << timerMaster.toc() <<" seconds."<<endl;
	}
	// ########### MAXIMIZATION ##############
	if (justDo == "EMC" || justDo == "M") {
		cout << "Start maximization" << endl;
		timerMaster.tic();
		status = maximization(comm, vm, det, numSlaves, numProcesses, \
		                      numCandidates(iter), numImages, \
		                      numSlices, iter);
		assert(status==0);
		cout << "Maximization time: " << timerMaster.toc() <<" seconds."<<endl;
	}
	// ########### COMPRESSION ##############
	if (justDo == "EMC" || justDo == "C") {
		cout << "Start compression" << endl;
		timerMaster.tic();
		status = compression(vm, myIntensity, myWeight, det, myRot, numSlices, \
			                 iter);
		assert(status==0);
		cout << "Compression time: " << timerMaster.toc() <<" seconds."<<endl;
		// ########### Save diffraction volume ##############
		cout << "Saving diffraction volume..." << endl;
		status = saveDiffractionVolume(vm, myIntensity, myWeight, iter);
		assert(status==0);
		}

	cout << "Done iteration: " << iter << endl;
}

static void slave_recon(mpi::communicator* comm, opt::variables_map vm) {
wall_clock timer;

	string output = vm["output"].as<string>();
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
	float msg[lenDPTAG];
	// Maximization related variables
	fvec condProb; // conditional probability
	uvec goodpixmap;
	fcube myDPnPixmap;
	fcube modelDPnPixmap; 	// first slice: expansion slice
							// second slice: good pixel map
	while (1) {

		// Receive a message from the master
		/////////////////////////////////////////////////////////
    	status = comm->recv(master, boost::mpi::any_tag, msg, lenDPTAG);
		/////////////////////////////////////////////////////////   		

    	// Calculate least squared error
    	if (status.tag() == DPTAG) {
    		int startInd = (int) msg[0];
    		int endInd = (int) msg[1];
    		int expansionInd = (int) msg[2];
    		int iter = (int) msg[3];
			numChunkData = endInd - startInd + 1; // number of measured data to process

			// Initialize
	    	condProb.zeros(numChunkData);

			//TODO: Time reading of expansion slice (all processes going for the same file)
	    	//////////////////////////
	    	// Read in expansion slice
	    	//////////////////////////
			// Get expansion image
			loadExpansionSlice(vm, iter, expansionInd, &modelDPnPixmap);

			//TODO: Time reading of data
	    	///////////////
	    	// Read in data
	    	///////////////
			int counter = 0;
	    	for (int i = startInd; i <= endInd; i++) {
				//Read in measured diffraction data
				if (format == "S2E") {
					loadDPnPixmap(vm, i+1, &myDPnPixmap);
			  	} else if (format == "list") { // TODO: this needs testing
				  	//load_readNthLine(vm, i, &myDP); // Wrong
			  	} else {
				  	loadDPnPixmap(vm, i+1, &myDPnPixmap);
				}
/*if (expansionInd == 99 && i == endInd){
cout << "imgInd: " << i+1 << endl;
cout << "DPval: " << myDPnPixmap.slice(0) << endl;
uvec mm = find(myDPnPixmap.slice(0)>0);
cout << "myDP: " << mm.n_elem << " " << mm << endl;
}*/
				/////////////////////////////////////////////////////
				// Compare measured diffraction with expansion slices
				/////////////////////////////////////////////////////
				double val = 0.0;
				if (useGaussianProb) {
					val = CToolbox::calculateGaussianSimilarity(&modelDPnPixmap, &myDPnPixmap, gaussianStdDev(iter));
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

// Given a diffraction volume (myIntensity) and save 2D slices (numSlices)
int expansion(opt::variables_map vm, fcube* myRot, fcube* myIntensity, \
              fcube* myWeight, CDetector* det, int numSlices, int iter) {
	int volDim = vm["volDim"].as<int>();
	string initialVolume;
	if (vm.count("initialVolume")) {
		initialVolume = vm["initialVolume"].as<string>();
	}
	int startIter = vm["startIter"].as<int>();
	bool initVol = false;
	if (startIter == iter) {
		initVol = true;
	}

	int active = 1;
	string interpolate = "linear";
	fmat myR;
	myR.zeros(3,3);
	fcube myDPnPixmap; 	// first slice: diffraction pattern
						// second slice: good pixel map

	if (initVol) {
		if ( strcmp(initialVolume.c_str(),"randomStart")==0 ) {
			cout << "Random diffraction volume..." << endl;
			myIntensity->randu(volDim,volDim,volDim);
			myWeight->ones(volDim,volDim,volDim);
		} else { // Load pre-existing diffraction volume
			cout << "Loading diffraction volume..." << endl;
			loadInitVol(vm, myIntensity, myWeight);
		}
	}
	
	// Slice diffraction volume and save to file
	for (int i = 0; i < numSlices; i++) {
		myDPnPixmap.zeros(volDim,volDim,2);
		// Get rotation matrix
		myR = myRot->slice(i);
		CToolbox::slice3D(&myDPnPixmap, &myR, myIntensity, myWeight, det, active, interpolate);
		// Save expansion slice to disk
		saveExpansionSlice(vm, &myDPnPixmap, iter, i);
	}

	return 0;
}

// Maximization
// goodpix: good pixel on a detector (used for beamstops and gaps)
int maximization(boost::mpi::communicator* comm, opt::variables_map vm, \
                 CDetector* det, int numSlaves, int numProcesses, \
                 int numCandidates, int numImages, \
                 int numSlices, int iter) {

	wall_clock timer;

	bool useGaussianProb = false;
	if (vm.count("gaussianStdDev")) {
		useGaussianProb = true;
	}
	bool saveCondProb = false;
	if (vm.count("saveCondProb")) {
		saveCondProb = vm["saveCondProb"].as<bool>();
	}
	
	bool useExistingProb = false;
	if (vm.count("useExistingProb")) {
		useExistingProb = true;
	}
	
	uvec numJobsForEachSlave(numSlaves);
	fvec myProb(numImages);
	fvec normCondProb;
	uvec candidatesInd;
	fcube updatedSlice_Pixmap;
	fvec candidateProb(numSlices);
	
	// Calculate number jobs for each slave
	numJobsForEachSlave = numJobsPerSlave(numImages, numSlaves);
	
	float lastPercentDone = 0;
	// Loop through all expansion slices and compare all measured data
	for (int expansionInd = 0; expansionInd < numSlices; expansionInd++) {
		if ( !useExistingProb ) {
			// For each slice, each worker get a subset of measured data
			sendJobsToSlaves(comm, numProcesses, &numJobsForEachSlave, \
				             expansionInd, iter);

			// Accumulate conditional probabilities for each expansion slice
			receiveProbsFromSlaves(comm, numProcesses, &numJobsForEachSlave, &myProb);
		
			if (saveCondProb) {
				saveCondProb2File(vm, iter, expansionInd, &myProb);
			}
		} else {
			loadCondProb(vm, iter, expansionInd, &myProb);
		}
		
		if (useGaussianProb) {
			normalizeCondProb(&myProb, numCandidates, &normCondProb, &candidatesInd);
		} else {
			// TODO: There must be a default option
		}

		// get best candidate probability for each expansion slice
		getBestCandidateProb(&normCondProb, &candidateProb, expansionInd);

		// Update expansion slice
		updateExpansionSlice(vm, &updatedSlice_Pixmap, &normCondProb, &candidatesInd);
		
		// Save updated expansion slice
		saveExpansionUpdate(vm, &updatedSlice_Pixmap, iter, expansionInd);
		
		// Display status
		CToolbox::displayStatusBar(expansionInd+1,numSlices,&lastPercentDone);
	}
	// save best candidate probabilities for all expansion slices
	saveBestCandidateProb(vm, &candidateProb, iter);
	
	return 0;
}

// Compression
int compression(opt::variables_map vm, fcube* myIntensity, fcube* myWeight, \
                CDetector* det, fcube* myRot, int numSlices, \
                int iter) {

	int volDim = vm["volDim"].as<int>();
	bool usePercentile = false;
	fvec percentile;
	if (vm.count("percentile")) {
		usePercentile = true;
		string percentileStr = vm["percentile"].as<string>();
		percentile = str2fvec(percentileStr);
	}
	
	myWeight->zeros(volDim,volDim,volDim);
	myIntensity->zeros(volDim,volDim,volDim);
	int active = 1;
	string interpolate = "linear";
	fcube myDPnPixmap; 	// first slice: diffraction pattern
						// second slice: good pixel map
	fmat myR;
	uvec goodSlicesInd;
	fvec candidateProb;
	if (usePercentile) {
		loadCandidateProb(vm, iter, &candidateProb);
		getGoodSlicesIndex(&candidateProb, percentile(iter), &goodSlicesInd);
		uvec::iterator a = goodSlicesInd.begin();
		uvec::iterator b = goodSlicesInd.end();
		for(uvec::iterator sliceInd = a; sliceInd != b; ++sliceInd) {
			// Get updated expansion slice
			loadUpdatedExpansion(vm, iter, *sliceInd, &myDPnPixmap);
			// Get rotation matrix
			getRotationMatrix(&myR, myRot, *sliceInd);
			// Merge into 3D diffraction volume
			CToolbox::merge3D(&myDPnPixmap, &myR, myIntensity, myWeight, det, active, interpolate);
		}
	} else {
		for (int sliceInd = 0; sliceInd < numSlices; sliceInd++) {
			// Get updated expansion slice
			loadUpdatedExpansion(vm, iter, sliceInd, &myDPnPixmap);
			// Get rotation matrix
			getRotationMatrix(&myR, myRot, sliceInd);
			// Merge into 3D diffraction volume
			CToolbox::merge3D(&myDPnPixmap, &myR, myIntensity, myWeight, det, active, interpolate);
		}
	}
	// Normalize here
	CToolbox::normalize(myIntensity, myWeight);
	return 0;
}

void generateUniformRotations(string rotationAxis, int numSlicesNow, fcube* myRot) {
	assert(myRot);
	fcube& _myRot = myRot[0];
	
	fmat myQuaternions;
	fmat myR;
	// Equal distribution of quaternions
	if (rotationAxis == "y" || rotationAxis == "z") {
		myQuaternions = CToolbox::pointsOn1Sphere(numSlicesNow, rotationAxis);
	} else {
		myQuaternions = CToolbox::pointsOn4Sphere(numSlicesNow);
	}
	for (int i = 0; i < numSlicesNow; i++) {
		myR = CToolbox::quaternion2rot3D(trans(myQuaternions.row(i)));
		_myRot.slice(i) = myR;
	}
}

void getRotationMatrix(fmat* myR, fcube* myRot, int sliceInd) {
	fmat& _myR = myR[0];
	_myR.zeros(3,3);
	_myR = myRot->slice(sliceInd);
}

int saveDiffractionVolume(opt::variables_map vm, fcube* myIntensity, fcube* myWeight, int iter) {
	int volDim = vm["volDim"].as<int>();
	string output = vm["output"].as<string>();

	std::stringstream ss;
	string filename;
	for (int i = 0; i < volDim; i++) {
		// save volume intensity
		ss.str("");
		ss << output << "/compression/iter" << iter << "/vol_" << setfill('0') << setw(7) << i << ".dat";
		filename = ss.str();
		myIntensity->slice(i).save(filename,raw_ascii);
		// save volume weights
		ss.str("");
		ss << output << "/compression/iter" << iter << "/volWeight_" << setfill('0') << setw(7) << i << ".dat";
		filename = ss.str();
		myWeight->slice(i).save(filename,raw_ascii);
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

void updateExpansionSlice(opt::variables_map vm, fcube* updatedSlice_Pixmap, fvec* normCondProb, uvec* candidatesInd) {
	string input = vm["input"].as<string>();
	string format = vm["format"].as<string>();
	string hdfField = vm["hdfField"].as<string>();
	int volDim = vm["volDim"].as<int>();
	
	uvec& _candidatesInd = candidatesInd[0];
	//uvec goodpix = det->get_goodPixelMap();		// good detector pixels
	
	int numCandidates = _candidatesInd.n_elem;
	updatedSlice_Pixmap->zeros(volDim,volDim,2);
	fcube myDPnPixmap;
	// Load measured diffraction pattern from file
	if (format == "S2E") {
		for (int i = 0; i < numCandidates; i++) {
			// load measured diffraction pattern
			loadDPnPixmap(vm, _candidatesInd(i)+1, &myDPnPixmap);
			// calculate weighted image and add to updatedSlice
			calculateWeightedImage(normCondProb->at(i), updatedSlice_Pixmap, &myDPnPixmap);
		}
	} else if (format == "list") { //TODO: this needs testing
		for (int i = 0; i < numCandidates; i++) {
			//load_readNthLine(vm, i, &myDP);
			// calculate weighted image and add to updatedSlice
			//calculateWeightedImage(&goodpix, normCondProb->at(i), updatedSlice, &myDP);
		}
	}
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
void sendJobsToSlaves(boost::mpi::communicator* comm, int numProcesses, uvec* numJobsForEachSlave, int expansionInd, int iter) {
	uvec& _numJobsForEachSlave = numJobsForEachSlave[0];
	
	int startInd = 0;
	int endInd = 0;
	for (int rank = 1; rank < numProcesses; ++rank) {
		endInd = startInd + _numJobsForEachSlave(rank-1) - 1;
		fvec id;
		id << startInd << endInd << expansionInd << iter << endr; // number of elements == lenDPTAG
		assert(id.n_elem == lenDPTAG);
		float* id1 = &id[0];
		////////////////////////////////
		comm->send(rank, DPTAG, id1, lenDPTAG); // send to slave
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
}

void saveCondProb2File(opt::variables_map vm, int iter, int expansionInd, fvec* myProb) {
	string output = vm["output"].as<string>();

	string filename;
	stringstream sstm;
	sstm << output << "/maximization/iter" << iter << "/similarity_" << setfill('0') << setw(7) << expansionInd << ".dat";
	filename = sstm.str();
	myProb->save(filename,raw_ascii);
}

void loadCondProb(opt::variables_map vm, int iter, int expansionInd, fvec* myProb) {
	string output = vm["output"].as<string>();

	string filename;
	stringstream sstm;
	sstm << output << "/maximization/iter" << iter << "/similarity_" << setfill('0') << setw(7) << expansionInd << ".dat";
	filename = sstm.str();
	myProb->load(filename,raw_ascii);
}

void saveExpansionSlice(opt::variables_map vm, fcube* myDPnPixmap, int iter, int ind) {
	string output = vm["output"].as<string>();

	std::stringstream ss;
	string filename;
	ss << output << "/expansion/iter" << iter << "/expansion_" << setfill('0') << setw(7) << ind << ".dat";
	filename = ss.str();
	myDPnPixmap->slice(0).save(filename,raw_ascii);
	ss.str("");
	ss << output << "/expansion/iter" << iter << "/expansionPixmap_" << setfill('0') << setw(7) << ind << ".dat";
	filename = ss.str();
	myDPnPixmap->slice(1).save(filename,raw_ascii);
}

void saveExpansionUpdate(opt::variables_map vm, fcube* updatedSlice_Pixmap, int iter, int expansionInd) {
	string output = vm["output"].as<string>();
	fcube& _updatedSlice_Pixmap = updatedSlice_Pixmap[0];

	string filename;
	// Save updated expansion slices
	std::stringstream sstm;
	sstm << output << "/expansion/iter" << iter << "/expansionUpdate_" << setfill('0') << setw(7) << expansionInd << ".dat";
	filename = sstm.str();
	_updatedSlice_Pixmap.slice(0).save(filename,raw_ascii);
	sstm.str("");
	sstm << output << "/expansion/iter" << iter << "/expansionUpdatePixmap_" << setfill('0') << setw(7) << expansionInd << ".dat";
	filename = sstm.str();
	_updatedSlice_Pixmap.slice(1).save(filename,raw_ascii);
}

void loadExpansionSlice(opt::variables_map vm, int iter, int sliceInd, fcube* modelDPnPixmap) {		
	string output = vm["output"].as<string>();
	int volDim = vm["volDim"].as<int>();
	modelDPnPixmap->zeros(volDim,volDim,2);
	
	std::stringstream ss;
	ss << output << "/expansion/iter" << iter << "/expansion_" << setfill('0') << setw(7) << sliceInd << ".dat";
	string filename = ss.str();
	modelDPnPixmap->slice(0) = load_asciiImage(filename); // load expansion slice
	// Get expansion pixmap
	ss.str("");
	ss << output << "/expansion/iter" << iter << "/expansionPixmap_" << setfill('0') << setw(7) << sliceInd << ".dat";
	filename = ss.str();
	modelDPnPixmap->slice(1) = load_asciiImage(filename); // load goodpixmap
}

void loadUpdatedExpansion(opt::variables_map vm, int iter, int sliceInd, fcube* myDPnPixmap) {
	string output = vm["output"].as<string>();
	int volDim = vm["volDim"].as<int>();
	myDPnPixmap->zeros(volDim,volDim,2);
		
	// Get image
	std::stringstream ss;
	string filename;
	ss << output << "/expansion/iter" << iter << "/expansionUpdate_" << setfill('0') << setw(7) << sliceInd << ".dat";
	filename = ss.str();
	myDPnPixmap->slice(0) = load_asciiImage(filename);
	// Get photon count pixmap
	ss.str("");
	ss << output << "/expansion/iter" << iter << "/expansionUpdatePixmap_" << setfill('0') << setw(7) << sliceInd << ".dat";
	filename = ss.str();
	myDPnPixmap->slice(1) = load_asciiImage(filename);
}

void loadDPnPixmap(opt::variables_map vm, int ind, fcube* myDPnPixmap) {
	string input = vm["input"].as<string>();
	string output = vm["output"].as<string>();
	string format = vm["format"].as<string>();
	string hdfField = vm["hdfField"].as<string>();
	int volDim = vm["volDim"].as<int>();
	
	myDPnPixmap->zeros(volDim,volDim,2);
	string filename;
	std::stringstream ss;
	if (format == "S2E") {
		ss << input << "/diffr/diffr_out_" << setfill('0') << setw(7) << ind << ".h5";
		filename = ss.str();
		// Read in diffraction			
		myDPnPixmap->slice(0) = hdf5readT<fmat>(filename,hdfField);
	} else {
		ss << input << "/diffr/diffr_out_" << setfill('0') << setw(7) << ind << ".dat";
		filename = ss.str();
		myDPnPixmap->slice(0) = load_asciiImage(filename);
	}
	//det->get_goodPixelMap();	
	ss.str("");
	ss << output << "/badpixelmap.dat";
	filename = ss.str();
	fmat pixmap = load_asciiImage(filename); // load badpixmap
	myDPnPixmap->slice(1) = CToolbox::badpixmap2goodpixmap(pixmap); // goodpixmap
}
/*
void loadDP(opt::variables_map vm, int ind, fmat* myDP) {
	string input = vm["input"].as<string>();
	string format = vm["format"].as<string>();
	string hdfField = vm["hdfField"].as<string>();
	int volDim = vm["volDim"].as<int>();
	fmat& _myDP = myDP[0];
	
	_myDP.zeros(volDim,volDim);
	string filename;
	std::stringstream ss;
	if (format == "S2E") {
		ss << input << "/diffr/diffr_out_" << setfill('0') << setw(7) << ind << ".h5";
		filename = ss.str();
		// Read in diffraction				
		_myDP = hdf5readT<fmat>(filename,hdfField);
	} else {
		ss << input << "/diffr/diffr_out_" << setfill('0') << setw(7) << ind << ".dat";
		filename = ss.str();
		_myDP = load_asciiImage(filename);
	}
}
*/
// Reads Nth line of a file containing names of diffraction patterns
// TODO: check
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

void calculateWeightedImage(float weight, fcube* updatedSlice_Pixmap, fcube* myDPnPixmap) {
	fcube& _updatedSlice_Pixmap = updatedSlice_Pixmap[0];
	
	fmat myDP = myDPnPixmap->slice(0);
	uvec photonCountPixmap = find(myDP > 0);
	// Setup goodpixmap
	uvec::iterator goodBegin = photonCountPixmap.begin();
	uvec::iterator goodEnd = photonCountPixmap.end();
	
	for(uvec::iterator p=goodBegin; p!=goodEnd; ++p) {
		_updatedSlice_Pixmap.slice(0)(*p) += weight * myDP(*p);
		_updatedSlice_Pixmap.slice(1)(*p) = 1;
	}
}

// Used to decide how good each expansion update is
void getBestCandidateProb(fvec* normCondProb, fvec* candidateProb, \
                           int expansionInd) {
	fvec& _normCondProb = normCondProb[0];
	fvec& _candidateProb = candidateProb[0];
	
	_candidateProb(expansionInd) = max(_normCondProb);
}

void saveBestCandidateProb(opt::variables_map vm, fvec* candidateProb, int iter) {
	string output = vm["output"].as<string>();
	
	string filename;
	std::stringstream sstm;
	sstm << output << "/maximization/iter" << iter << "/bestCandidateProb.dat";
	filename = sstm.str();
	candidateProb->save(filename,raw_ascii);
}

void loadCandidateProb(opt::variables_map vm, int iter, fvec* candidateProb) {
	string output = vm["output"].as<string>();
	
	string filename;
	std::stringstream sstm;
	sstm << output << "/maximization/iter" << iter << "/bestCandidateProb.dat";
	filename = sstm.str();
	candidateProb->load(filename,raw_ascii);
}

void getGoodSlicesIndex(fvec* candidateProb, float percentile, uvec* goodSlicesInd) {
	fvec& _candidateProb = candidateProb[0];
	uvec& _goodSlicesInd = goodSlicesInd[0];
	int numElem = _candidateProb.n_elem;
	int numChosen = round(numElem * percentile/100.); //1000 * 0.8
	uvec indices = sort_index(_candidateProb,"descend");
	_goodSlicesInd = indices.subvec(0,numChosen-1);
}

void loadInitVol(opt::variables_map vm, fcube* myIntensity, fcube* myWeight) {
	string initialVolume = vm["initialVolume"].as<string>();
	int volDim = vm["volDim"].as<int>();
	
	myIntensity->zeros(volDim,volDim,volDim);
	myWeight->zeros(volDim,volDim,volDim);
	std::stringstream ss;
	string filename;
	for (int i = 0; i < volDim; i++) {
		ss.str("");
		ss << initialVolume << "/vol_" << setfill('0') << setw(7) << i << ".dat";
		filename = ss.str();
		myIntensity->slice(i) = load_asciiImage(filename);
		ss.str("");
		ss << initialVolume << "/volWeight_" << setfill('0') << setw(7) << i << ".dat";
		filename = ss.str();
		myWeight->slice(i) = load_asciiImage(filename);
	}
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
        ("input,i", opt::value<std::string>(), "Input directory for finding /pmi and /diffr")
        ("output,o", opt::value<string>(), "Output directory for saving /expansion, /maximization, /compression")
        ("useFileList", opt::value<int>()->default_value(0), "Input a list containing filenames of diffraction patterns")
        ("beamFile", opt::value<string>(), "Beam file defining X-ray beam")
        ("geomFile", opt::value<string>(), "Geometry file defining diffraction geometry")
		("rotationAxis", opt::value<string>()->default_value("xyz"), "Euler rotation convention")
        ("numIterations", opt::value<int>(), "Number of iterations to perform from startIter")
        ("numImages", opt::value<string>(), "Number of measured diffraction patterns (Comma separated list)")
        ("numSlices", opt::value<string>(), "Number of Ewald slices in the expansion step (Comma separated list)")
        ("numCandidates", opt::value<string>(), "Number of best fitting images to update expansion slice (Comma separated list)")
        ("volDim", opt::value<int>(), "Number of pixel along one dimension")
        ("startIter", opt::value<int>()->default_value(0), "Start iteration number used to index 2 vectors: numImages and numSlices (count from 0)")
        ("initialVolume", opt::value<string>()->default_value("randomStart"), "Absolute path to initial volume")
        ("format", opt::value<string>(), "Defines file format to use")
        ("hdfField", opt::value<string>()->default_value("/data/data"), "Data field to use for reconstruction")
        ("gaussianStdDev", opt::value<string>(), "Use Gaussian likelihood for maximization with the following standard deviations (Comma separated list)")
        ("saveCondProb", opt::value<bool>(), "Optionally save conditional probabilities")
        ("justDo", opt::value<string>(), "Choose which E,M,C step to perform")
        ("percentile", opt::value<string>(), "Top percentile expansion slices to use for compression (Comma separated list)")
        ("useExistingProb", opt::value<bool>(), "Use the conditional probabilities that have already been computed")
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
		if (!vm.count("input")) {
    		cout << "NOTICE: input field is required" << endl;
    		exit(0);
    	}
		if (!vm.count("beamFile")) {
    		cout << "NOTICE: beamFile field is required" << endl;
    		exit(0);
    	}
    	if (!vm.count("geomFile")) {
    		cout << "NOTICE: geomFile field is required" << endl;
    		exit(0);
    	}
		//TODO: print all parameters
	}

	return vm;
} // end of parse_input
