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
#include "diffractionVolume.h"
#include "diffractionPattern.h"

namespace mpi = boost::mpi;
namespace opt = boost::program_options;
using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;
using namespace diffractionVolume;
using namespace diffractionPattern;

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
                         fcube* myRot, CDetector* det, CDiffrVol* diffrVol, \
                         int numImages, int numSlices, int iter);
static void slave_recon(mpi::communicator* comm, opt::variables_map vm);
opt::variables_map parse_input(int argc, char* argv[], mpi::communicator* comm);
static int expansion(opt::variables_map vm, arma::fcube* myRot, \
                     CDiffrVol* diffrVol, CDetector* det,\
                     int numSlices, int iter);
static int maximization(mpi::communicator* comm, opt::variables_map vm, \
                        CDetector* det, int numSlaves, int numProcesses, \
                        int numCandidates, int numImages, \
                        int numSlices, int iter);
static int compression(opt::variables_map vm, CDiffrVol* diffrVol, \
                       CDetector* det, fcube* myRot,\
                       int numSlices, int iter);
uvec numJobsPerSlave(int numImages, int numSlaves);
void normalizeCondProb(fmat* condProb, int numCandidates, fmat* normCondProb, \
                       umat* candidatesInd);
void sendJobsToSlaves(boost::mpi::communicator* comm, int numProcesses, \
                      uvec* numJobsForEachSlave, int numSlices, int iter);
void receiveProbsFromSlaves(boost::mpi::communicator* comm, int numProcesses, \
                            uvec* numJobsForEachSlave, fmat* condProb, \
                            int numSlices);
void saveCondProb2File(opt::variables_map vm, int iter, \
                       fmat* myProb);
void loadCondProb(opt::variables_map vm, int iter, fmat* myProb);
void load_readNthLine(opt::variables_map vm, int N, fmat* img);
void getRotationMatrix(fmat* myR, fcube* myRot, int sliceInd);
void generateUniformRotations(string rotationAxis, int numSlicesNow, \
                              fcube* myRot);
//void getBestCandidateProb(fvec* normCondProb, fvec* candidateProb, \
                           int expansionInd);
void getGoodSlicesIndex(fvec* candidateProb, float percentile, \
                        uvec* goodSlicesInd);
void saveBestCandidateProbPerSlice(opt::variables_map vm, fmat* condProb, \
                                   umat* candidateInd, int iter);
void loadCandidateProbPerSlice(opt::variables_map vm, int iter, \
                               fvec* candidateProb);

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
	int volDim = vm["volDim"].as<int>();

	// Check the input makes sense
	int numSlaves = comm->size()-1;
	int maxNumImages = max(numImages);
	if (maxNumImages < numSlaves) {
		cerr << "Number of workers too large for this task: " << maxNumImages << "<" << numSlaves << endl;
		return 0;
	}
	//TODO: check length of numImages == numSlices
	//TODO: check startIter is less than length numImages
	//TODO: check startIter+numIterations is less than length numImages

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
		CDiffrVol diffrVol = CDiffrVol(volDim);
		for (int iter = startIter; iter < startIter+numIterations; iter++) {
			int numImagesNow = numImages[iter];
			int numSlicesNow = numSlices[iter];
			fcube myRot(3,3,numSlicesNow);
			cout << "***** ITER " << iter << " *****" << endl;
			generateUniformRotations(rotationAxis, numSlicesNow, &myRot);
			master_recon(comm, vm, &myRot, &det, &diffrVol, numImagesNow, numSlicesNow, iter);
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

static void master_recon(mpi::communicator* comm, opt::variables_map vm, \
                         fcube* myRot, CDetector* det, CDiffrVol* diffrVol, \
                         int numImages, int numSlices, int iter){

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
		status = expansion(vm, myRot, diffrVol, det, numSlices, iter);
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
		status = compression(vm, diffrVol, det, myRot, numSlices, \
			                 iter);
		assert(status==0);
		cout << "Compression time: " << timerMaster.toc() <<" seconds."<<endl;
		// ########### Save diffraction volume ##############
		cout << "Saving diffraction volume..." << endl;
		diffrVol->saveDiffractionVolume(vm, iter); //status = saveDiffractionVolume(vm, diffrVol, iter);
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
	string useProbType = vm["useProbType"].as<string>();
	boost::to_upper(useProbType);
	
	fvec gaussianStdDev;
	if (useProbType == "GAUSSIAN") {
		if (vm.count("gaussianStdDev")) {
			string gaussianStdDevStr = vm["gaussianStdDev"].as<string>();
			gaussianStdDev = str2fvec(gaussianStdDevStr);
		} else {
			cout << "gaussianStdDev is missing" << endl;
			exit(0);
		}
	}
	
	int numChunkData;
	boost::mpi::status status;
	float msg[lenDPTAG];
	// Maximization related variables
	fvec condProb; // conditional probability (numChunk x numSlices)
	uvec goodpixmap;
	CDiffrPat myDP, mySlice;
	fmat dataBlock, sliceBlock;
	fvec dataSizes, val;
	umat pixmapRecord;
	int maxElements = 600;
	urowvec s;
	while (1) {

		// Receive a message from the master
		/////////////////////////////////////////////////////////
    	status = comm->recv(master, boost::mpi::any_tag, msg, lenDPTAG);
		/////////////////////////////////////////////////////////   		

    	// Calculate least squared error
    	if (status.tag() == DPTAG) {    	
    		int startInd = (int) msg[0];
    		int endInd = (int) msg[1];
    		int numSlices = (int) msg[2];
    		int iter = (int) msg[3];
			numChunkData = endInd - startInd + 1; // number of measured data to process
			// Initialize
	    	condProb.zeros(numChunkData*numSlices);
			dataSizes.zeros(numChunkData);
			dataBlock.zeros(numChunkData,maxElements);
			sliceBlock.zeros(numChunkData,maxElements);
			pixmapRecord.zeros(numChunkData,maxElements);
			val.zeros(numChunkData);
			int condProbPos = 0;
			
			////////////////////////////////////////
			// Read in data into dataBlock only once
			////////////////////////////////////////
			
			if (format == "S2E") {
				int counter = 0;
				for (int i = startInd; i <= endInd; i++) {
					myDP.loadPhotonCount(vm, i+1);
					dataSizes(counter) = myDP.photonpixmap.n_elem;
					// save into block
					if (dataSizes(counter) > maxElements) {
						dataBlock( counter,span(0,599) ) = trans(myDP.photonCount(myDP.photonpixmap.subvec(0,599)));
						pixmapRecord( counter,span(0,599) ) = trans(myDP.photonpixmap.subvec(0,599));			
					} else {
						dataBlock( counter,span(0,dataSizes(counter)-1) ) = trans(myDP.photonCount(myDP.photonpixmap));
						pixmapRecord( counter,span(0,dataSizes(counter)-1) ) = trans(myDP.photonpixmap);		
					}
					counter++;
			  	}
			}
			
			float lastPercentDone = 0;
			for (int j = 0; j < numSlices; j++) {
				//////////////////////////
				// Read in expansion slice
				//////////////////////////
				mySlice.loadExpansionSlice(vm, iter, j);

				////////////////////////
				// Shape into sliceBlock
				////////////////////////
				s.zeros(maxElements);
				if (format == "S2E") {
					for (int i = 0; i < numChunkData; i++) {
						// save into block
						if (dataSizes(i) > maxElements) {
							sliceBlock( i,span(0,599) ) = trans(mySlice.photonCount(pixmapRecord.row(i)));					
						} else {
							s = pixmapRecord.row(i);
							sliceBlock( i,span(0,dataSizes(i)-1) ) = trans(mySlice.photonCount(s.subvec(0,dataSizes(i)-1)));
						}
				  	}
				}
				
				/////////////////////////////////////////////////////
				// Compare measured diffractions with expansion slice
				/////////////////////////////////////////////////////
				if (useProbType == "POISSON") {
					//val = CToolbox::calculatePoissonianSimilarity(&mySlice, &myDP);
				} else {
					val = CToolbox::calculateGaussianSimilarityBlock(&sliceBlock, &dataBlock, gaussianStdDev(iter));
				}

				condProb.subvec(condProbPos,condProbPos+numChunkData-1) = val; // val (numChunkData x 1)
				condProbPos += numChunkData;
				if (comm->rank() == 1) {
					CToolbox::displayStatusBar(j+1, numSlices, &lastPercentDone);
				}
			}
			// Send back conditional probability to master
			float* msgProb = &condProb[0];
			/////////////////////////////////////////////////////
			comm->send(master, PROBTAG, msgProb, condProb.n_elem); // send probability in chunk size
			/////////////////////////////////////////////////////
		}

		if (status.tag() == DIETAG) {
		  	return;
		}

	} // end of while
} // end of slave_recon

// Given a diffraction volume (myIntensity) and save 2D slices (numSlices)
int expansion(opt::variables_map vm, fcube* myRot, CDiffrVol* diffrVol, \
              CDetector* det, int numSlices, int iter) {
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
	CDiffrPat myDP;

	if (initVol) {
		if ( strcmp(initialVolume.c_str(),"randomStart")==0 ) {
			cout << "Random diffraction volume..." << endl;
			diffrVol->randVol();
		} else { // Load pre-existing diffraction volume
			cout << "Loading diffraction volume...";
			diffrVol->loadInitVol(vm);//loadInitVol(vm, diffrVol);
			cout << "Done" << endl;
		}
	}
	
	// Get a slice of the diffraction volume and save to file
	cout << "Generating slices...";
	for (int i = 0; i < numSlices; i++) {		
		myDP.init(volDim);
		// Get rotation matrix
		myR = myRot->slice(i);
		CToolbox::slice3D(&myDP, &myR, diffrVol, det, active, interpolate);
		// Save expansion slice to disk
		myDP.saveExpansionSlice(vm, iter, i);
	}
	cout << "Done" << endl;

	return 0;
}

// Maximization
int maximization(boost::mpi::communicator* comm, opt::variables_map vm, \
                 CDetector* det, int numSlaves, int numProcesses, \
                 int numCandidates, int numImages, \
                 int numSlices, int iter) {

	wall_clock timer;

	bool saveCondProb = false;
	if (vm.count("saveCondProb")) {
		saveCondProb = vm["saveCondProb"].as<bool>();
	}
	
	bool useExistingProb = false;
	if (vm.count("useExistingProb")) {
		useExistingProb = vm["useExistingProb"].as<bool>();
	}

	// Initialize
	cout << "Initializing... ";
	uvec numJobsForEachSlave(numSlaves);
	fmat myProb(numImages,numSlices);
	fmat normCondProb(numCandidates,numSlices);
	umat candidatesInd(numCandidates,numSlices);
	fvec candidateProb(numSlices);
	CDiffrPat updatedSlice;
	cout << "Done" << endl;
	// Calculate number jobs for each slave
	numJobsForEachSlave = numJobsPerSlave(numImages, numSlaves);

	// Each slave get a set of data to compare against all slices
	if ( !useExistingProb ) {
		// For each slice, each worker get a subset of measured data
		cout << "Assigning jobs to slaves... ";
		sendJobsToSlaves(comm, numProcesses, &numJobsForEachSlave, \
			             numSlices, iter);
		cout << "Done" << endl;
		// Accumulate conditional probabilities for each expansion slice
		receiveProbsFromSlaves(comm, numProcesses, &numJobsForEachSlave, \
		                       &myProb, numSlices);

		if (saveCondProb) {
			cout << "Saving conditional probabilities... ";
			saveCondProb2File(vm, iter, &myProb);
			cout << "Done" << endl;
		}
	} else {
		loadCondProb(vm, iter, &myProb);
	}
	// normalize conditional probabilities
	cout << "Normalizing conditional probabilities... ";
	normalizeCondProb(&myProb, numCandidates, &normCondProb, &candidatesInd);
	cout << "Done" << endl;
	
	// update all expansion slices
	fvec myWeights(numCandidates);
	uvec myCandidatesInd(numCandidates);
	cout << "Saving updated expansion slices... ";
	for (int i = 0; i < numSlices; i++) {
		myWeights = normCondProb.col(i);
		myCandidatesInd = candidatesInd.col(i);
		// Update expansion slice
		updatedSlice.updateExpansionSlice(vm, &myWeights, &myCandidatesInd);
		// Save updated expansion slice
		updatedSlice.saveExpansionUpdate(vm, iter, i);
	}
	cout << "Done" << endl;
	// save best candidate probabilities for all expansion slices
	saveBestCandidateProbPerSlice(vm, &myProb, &candidatesInd, iter);

	return 0;
}

// Compression
int compression(opt::variables_map vm, CDiffrVol* diffrVol, \
                CDetector* det, fcube* myRot, int numSlices, \
                int iter) {

	bool usePercentile = false;
	fvec percentile;
	if (vm.count("percentile")) {
		usePercentile = true;
		string percentileStr = vm["percentile"].as<string>();
		percentile = str2fvec(percentileStr);
	}
	
	diffrVol->initVol();//intensity.zeros(volDim,volDim,volDim);
	//diffrVol->weight.zeros(volDim,volDim,volDim);
	int active = 1;
	string interpolate = "linear";
	CDiffrPat myUpdatedSlice; 	// first slice: diffraction pattern
						// second slice: good pixel map
	fmat myR;
	uvec goodSlicesInd;
	fvec candidateProb;
	if (usePercentile) {
		loadCandidateProbPerSlice(vm, iter, &candidateProb);
//cout << "candidateProb: " << candidateProb << endl;
		getGoodSlicesIndex(&candidateProb, percentile(iter), &goodSlicesInd);
//cout << "goodSlicesInd: " << goodSlicesInd << endl;
		uvec::iterator a = goodSlicesInd.begin();
		uvec::iterator b = goodSlicesInd.end();
		for(uvec::iterator sliceInd = a; sliceInd != b; ++sliceInd) {
			// Get updated expansion slice
			myUpdatedSlice.loadUpdatedExpansion(vm, iter, *sliceInd);
			// Get rotation matrix
			getRotationMatrix(&myR, myRot, *sliceInd);
			// Merge into 3D diffraction volume
			CToolbox::merge3D(&myUpdatedSlice, &myR, diffrVol, det, active, interpolate);
		}
	} else {
		for (int sliceInd = 0; sliceInd < numSlices; sliceInd++) {
			// Get updated expansion slice
			myUpdatedSlice.loadUpdatedExpansion(vm, iter, sliceInd);
			// Get rotation matrix
			getRotationMatrix(&myR, myRot, sliceInd);
			// Merge into 3D diffraction volume
			CToolbox::merge3D(&myUpdatedSlice, &myR, diffrVol, det, active, interpolate);
		}
	}
	// Normalize here
	diffrVol->normalize();
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

void normalizeCondProb(fmat* condProb, int numCandidates, fmat* normCondProb, umat* candidatesInd) {
	fmat& _condProb = condProb[0];
	fmat& _normCondProb = normCondProb[0];
	umat& _candidatesInd = candidatesInd[0];

	int numImages = _condProb.n_rows;
	int numSlices = _condProb.n_cols;
	_normCondProb.zeros(numCandidates,numSlices);
	_candidatesInd.zeros(numCandidates,numSlices);
	fvec prob;
	prob.zeros(numImages);
	uvec indices;
	indices.zeros(numImages);
	uvec candidates;
	candidates.zeros(numCandidates);
	fvec candidatesVal;
	candidatesVal.zeros(numCandidates);
	for (int i = 0; i < numSlices; i++) {
		prob = _condProb.col(i);
		indices = sort_index(prob,"descend");
		candidates = indices.subvec(0,numCandidates-1);
		_candidatesInd.col(i) = candidates;
		// Calculate norm cond prob
		for (int j = 0; j < numCandidates; j++) {
			candidatesVal(j) = prob(candidates(j));
		}
		_normCondProb.col(i) = candidatesVal / sum(candidatesVal);
	}
//_normCondProb.print("normCondProb");
//_candidatesInd.print("candidatesInd"); 
}

////////////////////////////////////////////
// Send jobs to slaves
// 1) Start index of measured diffraction pattern
// 2) End index of measured diffraction pattern
// 3) Index of expansion slice
////////////////////////////////////////////
void sendJobsToSlaves(boost::mpi::communicator* comm, int numProcesses, uvec* numJobsForEachSlave, int numSlices, int iter) {
	uvec& _numJobsForEachSlave = numJobsForEachSlave[0];
	
	int startInd = 0;
	int endInd = 0;
	for (int rank = 1; rank < numProcesses; ++rank) {
		endInd = startInd + _numJobsForEachSlave(rank-1) - 1;
		fvec id;
		id << startInd << endInd << numSlices << iter << endr; // number of elements == lenDPTAG
		assert(id.n_elem == lenDPTAG);
		float* id1 = &id[0];
		////////////////////////////////
		comm->send(rank, DPTAG, id1, lenDPTAG); // send to slave
		////////////////////////////////
		startInd += _numJobsForEachSlave(rank-1);
//id.print("id:");		
	}
}

void receiveProbsFromSlaves(boost::mpi::communicator* comm, int numProcesses, uvec* numJobsForEachSlave, fmat* masterCondProb, int numSlices) {
	uvec& _numJobsForEachSlave = numJobsForEachSlave[0];
	fmat& _masterCondProb = masterCondProb[0];

	int counter, numChunkData, prevChunkData;
	float* msgProb = new float[max(_numJobsForEachSlave)*numSlices];
	prevChunkData = 0;
	float lastPercentDone = 0.;
	for (int rank = 1; rank < numProcesses; ++rank) {
		numChunkData = _numJobsForEachSlave(rank-1);
		///////////////////////////////////////////////////////
		comm->recv(rank, PROBTAG, msgProb, numChunkData*numSlices); // receive from slave
		///////////////////////////////////////////////////////
		counter = 0;
		for (int j = 0; j < numSlices; j++) {
		for (int i = 0; i < numChunkData; i++) { //FIXME
			_masterCondProb(i+prevChunkData,j) = msgProb[counter];
			counter++;
		}
		}
		prevChunkData += numChunkData;
		CToolbox::displayStatusBar(rank, numProcesses, &lastPercentDone);
	}
	delete[] msgProb;
}

void saveCondProb2File(opt::variables_map vm, int iter, fmat* myProb) {
	string output = vm["output"].as<string>();

	string filename;
	stringstream sstm;
	sstm << output << "/maximization/iter" << iter << "/similarity.dat";
	filename = sstm.str();
	myProb->save(filename,raw_ascii);
}

void loadCondProb(opt::variables_map vm, int iter, fmat* myProb) {
	string output = vm["output"].as<string>();

	string filename;
	stringstream sstm;
	sstm << output << "/maximization/iter" << iter << "/similarity.dat";
	filename = sstm.str();
	myProb->load(filename,raw_ascii);
}

// Reads Nth line of a file containing names of diffraction patterns
// FIXME
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
/*
// Used to decide how good each expansion update is
void getBestCandidateProb(fmat* normCondProb, fvec* candidateProb, \
                           int expansionInd) {
	fmat& _normCondProb = normCondProb[0];
	fvec& _candidateProb = candidateProb[0];
	
	int numSlices = _normCondProb.n_cols;
	for (int i = 0; i < numSlices; i++) {
		//_candidateProb(i) = max(_normCondProb);
	}
}
*/
void saveBestCandidateProbPerSlice(opt::variables_map vm, fmat* condProb, umat* candidateInd, int iter) {
	string output = vm["output"].as<string>();
	
	int numSlices = condProb->n_cols;
	fvec candidateProb(numSlices);
	for (int i = 0; i < numSlices; i++) {
		candidateProb(i) = condProb->at(candidateInd->at(0,i),i);
	}
	string filename;
	std::stringstream sstm;
	sstm << output << "/maximization/iter" << iter << "/bestCandidateProb.dat";
	filename = sstm.str();
	candidateProb.save(filename,raw_ascii);
}

void loadCandidateProbPerSlice(opt::variables_map vm, int iter, fvec* candidateProb) {
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
        ("useProbType", opt::value<string>()->default_value("POISSON"), "Use Poisson or Gaussian likelihood for maximization")
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
