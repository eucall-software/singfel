/*
 * Program for merging diffraction patterns based on maximum cross correlations
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
#include "detector.h"
#include "beam.h"
#include "particle.h"
#include "diffraction.h"
#include "toolbox.h"
#include "diffraction.cuh"
#include "io.h"
#include <fstream>
#include <string>

#include <armadillo>

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>

namespace mpi = boost::mpi;

using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

#define USE_CUDA 0

#define MODELTAG 1	// mySlices matrix
#define DPTAG 2	// diffraction pattern
#define DIETAG 3 // die signal
#define SAVESLICESTAG 4 // save slices signal
#define SAVELSETAG 5 // save LSE signal
#define GOODPIXTAG 6 // goodpixelmap
#define DONETAG 7 // done signal
static void master_expansion(mpi::communicator* comm, fmat* quaternions, fcube* myIntensity, fmat* pix, float pix_max, uvec* goodpix, int numImages, int mySize, int iter, int numIterations, int numSlices, string output);
static void slave_expansion(mpi::communicator* comm, int numImages, int mySize, string output, int useFileList, string input);

int main( int argc, char* argv[] ){

	wall_clock timerMaster;

	// Initialize MPI
  	mpi::environment env;
  	mpi::communicator world;
	mpi::communicator* comm = &world;

	// Everyone parses input
	int master = 0;
	string imageList;
	int useFileList = 0;
	string quaternionList;
	string beamFile;
	string geomFile;	
	int numIterations = 0;
	int numImages = 0;
	int mySize = 0;
	int numSlices = 0;
	int startIter = 0;
	string output;
	string intialVolume = "randomMerge";
	// Let's parse input
	for (int n = 1; n < argc; n++) {
	cout << argv [ n ] << endl;
	    if(boost::algorithm::iequals(argv[ n ], "-i")) {
	        imageList = argv[ n+1 ];
	    } else if (boost::algorithm::iequals(argv[ n ], "-f")) {
	        useFileList = atoi(argv[ n+1 ]);
	    } else if (boost::algorithm::iequals(argv[ n ], "-q")) {
	        quaternionList = argv[ n+1 ];
	    } else if (boost::algorithm::iequals(argv[ n ], "-b")) {
	        beamFile = argv[ n+1 ];
	    } else if (boost::algorithm::iequals(argv[ n ], "-g")) {
	        geomFile = argv[ n+1 ];
	    } else if (boost::algorithm::iequals(argv[ n ], "--num_iterations")) {
        	numIterations = atoi(argv[ n+2 ]);
	    } else if (boost::algorithm::iequals(argv[ n ], "--num_images")) {
	        numImages = atoi(argv[ n+2 ]);
	    } else if (boost::algorithm::iequals(argv[ n ], "--num_slices")) {
	        numSlices = atoi(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--vol_dim")) {
		    mySize = atof(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--output_name")) {
		    output = argv[ n+2 ];
		} else if (boost::algorithm::iequals(argv[ n ], "--start_iter_from")) {
		    startIter = atoi(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--initial_volume")) {
		    intialVolume = argv[ n+2 ];
		}
	}
	
	fcube myIntensity;
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
		        //cout << line << endl; 
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
		                //cout << temp << endl;
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

		goodpix = det.get_goodPixelMap();
		//cout << "Good pix:" << goodpix << endl;

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
		fcube myWeight;
		myWeight.zeros(mySize,mySize,mySize);
		myIntensity.zeros(mySize,mySize,mySize);
		//int numPixels = mySize * mySize;
		//fmat myImages;
		//myImages.zeros(numImages,numPixels);
	
		int counter = 0;
		pix.zeros(det.numPix,3);
		for (int i = 0; i < px; i++) {
			for (int j = 0; j < py; j++) { // column-wise
				pix(counter,0) = det.q_xyz(j,i,0);
				pix(counter,1) = det.q_xyz(j,i,1);
				pix(counter,2) = det.q_xyz(j,i,2);
				counter++;
			}
		}
		fvec pix_mod;
		pix = pix * 1e-10; // (nm)
		pix_mod = sqrt(sum(pix%pix,1));
		pix_max = max(pix_mod);
		float inc_res = (mySize-1)/(2*pix_max/sqrt(2));
		pix = pix * inc_res;
		pix_mod = sqrt(sum(pix%pix,1));		
		pix_max = cx;//max(pix_mod);

		timerMaster.tic();
		
		int active;
		string interpolate = "linear";
		
		fmat rot3D(3,3);
		fvec u(3);
		fvec quaternion(4);
	
		std::ifstream infile;
		if (useFileList) {
			cout << "Using image list: " << imageList << endl;
			infile.open(imageList.c_str());
		}
	
		if ( strcmp(intialVolume.c_str(),"randomMerge")==0 ) {
			cout << "Randomly merging diffraction volume..." << endl;
			// Setup initial diffraction volume by merging randomly
			for (int r = 0; r < numImages; r++) {
			  	// Get image
			  	if (useFileList) {
			  		std::getline(infile, line);
			  		myDP = load_asciiImage(line);
			  	} else {
				  	std::stringstream sstm;
			  		sstm << imageList << setfill('0') << setw(7) << r << ".dat";
					filename = sstm.str();
					myDP = load_asciiImage(filename);
				}
			
				// Get rotation matrix
			  	u = randu<fvec>(3); // uniform random distribution in the [0,1] interval
				// generate uniform random quaternion on SO(3)
				quaternion << sqrt(1-u(0)) * sin(2*datum::pi*u(1)) << sqrt(1-u(0)) * cos(2*datum::pi*u(1))
						   << sqrt(u(0)) * sin(2*datum::pi*u(2)) << sqrt(u(0)) * cos(2*datum::pi*u(2));
			
				myR = CToolbox::quaternion2rot3D(quaternion);
				active = 1;
				CToolbox::merge3D(&myDP, &pix, &goodpix, &myR, pix_max, &myIntensity, &myWeight, active, interpolate);
			
				// Save diffraction pattern
				//myImages.row(r) = reshape(myDP,1,numPixels); // read along fs
		  	}
			// Normalize here
			CToolbox::normalize(&myIntensity,&myWeight);
	  	} else {
			cout << "Loading diffraction volume..." << endl;
			for (int i = 0; i < mySize; i++) {
				std::stringstream sstm;
				sstm << intialVolume << setfill('0') << setw(7) << i << ".dat";
				string outputName = sstm.str();
				myIntensity.slice(i) = load_asciiImage(outputName);
			}
		}
	}

	world.barrier();
	if (world.rank() == master) {
		cout << "Initialization time: " << timerMaster.toc() <<" seconds."<<endl;
	}
	
	// Main iteration
	fmat myQuaternions;
	for (int iter = startIter; iter < numIterations; iter++) { // number of iterations
		if (world.rank() == master) {
			if (iter == startIter) {
				// Equal distribution of quaternions
  				myQuaternions = CToolbox::pointsOn4Sphere(numSlices);
  			}
  			cout << "iteration: " << iter << endl;
			master_expansion(comm, &myQuaternions, &myIntensity, &pix, pix_max, &goodpix, numImages, mySize, iter, numIterations, numSlices, output);
		} else {
			slave_expansion(comm, numImages, mySize, output, useFileList, imageList);
		}
		world.barrier();
		if (world.rank() == master) {
			cout << "Barrier passed" << endl;
		}
	}
  	return 0;
}

static void master_expansion(mpi::communicator* comm, fmat* quaternions, fcube* myIntensity, fmat* pix, float pix_max, uvec* goodpix, int numImages, int mySize, int iter, int numIterations, int numSlices, string output) {

	wall_clock timerMaster;
	
	// ########### EXPANSION ##############
	cout << "Start expansion" << endl;
	timerMaster.tic();

	//int numImages = myImages->n_rows;
	
  	int ntasks, rank, numProcesses, numSlaves;
  	const int numMaster = 1;
  	fvec quaternion;
  	boost::mpi::status status;

	numProcesses = comm->size();
	numSlaves = numProcesses-1;
	ntasks = quaternions->n_rows;
	if (ntasks < numSlaves) {
		numProcesses = ntasks + numMaster;
	}
	//cout << "numProcesses: " << numProcesses << endl;
	// Find out how many processes there are in the default communicator
	//http://en.wikipedia.org/wiki/ANSI_escape_code#graphics
	// 033: Escape
	// 1: Effect = Bold
	// 30+1: Color = Red
	//cout << "\033[1;31mhow many processes?: \033[0m" << ntasks << endl;
  	//MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

	// Vectorize my diffraction volume
	fvec intensityVector(myIntensity->n_elem);
    for(unsigned int i = 0; i < myIntensity->n_elem; i++) {
    	intensityVector(i) =  myIntensity->at(i); // column-wise
	}
	std::vector<float> data = conv_to< std::vector<float> >::from(intensityVector);

	int active = 1;
	string interpolate = "linear";
	//int numPixels = mySize*mySize;
	fmat myR;
	myR.zeros(3,3);
	fmat myDP;
	//fmat mySlice(1,numPixels);
	for (int i = 0; i < numSlices; i++) {
		myDP.zeros(mySize,mySize);
		// Get rotation matrix
		myR = CToolbox::quaternion2rot3D(trans(quaternions->row(i)));
		CToolbox::slice3D(&myDP, pix, goodpix, &myR, pix_max, myIntensity, active, interpolate);
		// Save expansion slice to disk
		std::stringstream sstm;
		sstm << output << "myExpansion_" << setfill('0') << setw(7) << i << ".dat";
		string outputName = sstm.str();
		myDP.save(outputName,raw_ascii);
	}
	
	cout << "Done mySlices" << endl;
	
	// Send
	// 1) Start and end indices of model slices
	// 2) Subset of model slices
	// 3) Compute signal
	int slicesPerSlave = floor( (float) numSlices / (float) (numProcesses-1) );
	int leftOver = numSlices - slicesPerSlave * numSlaves;
	
	//cout << "slicesPerSlave: " << slicesPerSlave << endl;
	//cout << "leftOver: " << leftOver << endl;
	
	uvec s(numSlaves);
	s.fill(slicesPerSlave);
	for (int i = 0; i < numSlaves; i++) {
		if (leftOver > 0) {
			s(i) += 1;
			leftOver--;
		}
	}
	//cout << "s: " << s << endl;
	
	// Send good pixel map to slaves
	uvec& myGoodpix = goodpix[0];
	std::vector<unsigned int> goodpixelmap = conv_to< std::vector<unsigned int> >::from(myGoodpix);
	//for( std::vector<unsigned int>::const_iterator i = goodpixelmap.begin(); i != goodpixelmap.end(); ++i) {
    //	std::cout << *i << ' ';
    //}
	for (rank = 1; rank < numProcesses; ++rank) {
		comm->send(rank, GOODPIXTAG, goodpixelmap);
	}
	
	int startInd = 0;
	int endInd = 0;
	std::vector<float> msg;
	for (rank = 1; rank < numProcesses; ++rank) {
		endInd = startInd + s(rank-1) - 1;

		std::vector<int> id(2);
		id.at(0) = startInd;
		id.at(1) = endInd;
			
		comm->send(rank, DPTAG, id);//comm->send(rank, DPTAG, msg);
		
		//comm->send(rank, SAVESLICESTAG, msg);
		
		//comm->send(rank, SAVELSETAG, msg);
		
		startInd += s(rank-1);
  	}

	cout << "Expansion time: " << timerMaster.toc() <<" seconds."<<endl;

	// SAVE SLICES FOR VIEWING
	// Tell all the slaves to save slices by sending an empty message with the SAVESLICESTAG.
//	for (rank = 1; rank < numProcesses; ++rank) {
//		comm->send(rank, SAVESLICESTAG, msg);
//	}

	// ########### MAXIMIZATION ##############
	cout << "Start maximization" << endl;
	
	timerMaster.tic();
	
	std::vector<float> msgLSE;
	fmat myTable(numImages,numSlices);
	int numWorkerSlices;
	int currentCol = 0;
  	// Receive chunks of LSE from the slaves.
	for (rank = 1; rank < numProcesses; ++rank) {
    	status = comm->recv(rank, boost::mpi::any_tag, msgLSE);
		fvec lse = conv_to< fvec >::from(msgLSE);
		numWorkerSlices = s(rank-1);
		myTable.cols(currentCol,currentCol+s(rank-1)-1) = reshape(lse,numImages,numWorkerSlices);
		currentCol += s(rank-1);
	}
	
	// SAVE LSE FOR VIEWING
	// Tell all the slaves to save LSE by sending an empty message with the SAVELSETAG.
	//for (rank = 1; rank < numProcesses; ++rank) {
	//	comm->send(rank, SAVELSETAG, msg);
	//}
	
	// Master calculates the minimum for each diffraction pattern
	uvec bestFit(numImages);
	float lowest;
	for (int i = 0; i < numImages; i++) {
		lowest = myTable(i,0);
		bestFit(i) = 0;
		for (int j = 0; j < numSlices; j++) {
			if (lowest > myTable(i,j)) {
				lowest = myTable(i,j);
				bestFit(i) = j; // minimum lse index
			}
		}
	}
	cout << "Maximization time: " << timerMaster.toc() <<" seconds."<<endl;
	
	// ########### COMPRESSION ##############
	cout << "Start compression" << endl;
	timerMaster.tic();
	
	fcube myWeight;
	myWeight.zeros(mySize,mySize,mySize);
	fcube myIntensityOld(mySize,mySize,mySize);
	for (int r = 0; r < mySize; r++) {
		myIntensityOld.slice(r) = myIntensity->slice(r);
	}
	myIntensity->zeros(mySize,mySize,mySize);
	active = 1;
	interpolate = "linear";
	string filename;
	for (int r = 0; r < numImages; r++) {
		// Get image
		std::stringstream sstm;
		sstm << output << "myExpansion_" << setfill('0') << setw(7) << r << ".dat";
		filename = sstm.str();
		myDP = load_asciiImage(filename);
		// Get rotation matrix
		myR = CToolbox::quaternion2rot3D(trans(quaternions->row(bestFit(r))));
		CToolbox::merge3D(&myDP, pix, goodpix, &myR, pix_max, myIntensity, &myWeight, active, interpolate);
	}
	// Normalize here
	CToolbox::normalize(myIntensity,&myWeight);
	
	// Introduce memory to myIntensity
	for (int r = 0; r < mySize; r++) {
		myIntensity->slice(r) = 0.1*myIntensityOld.slice(r) + 0.9*myIntensity->slice(r);
	}
	
	
	cout << "Compression time: " << timerMaster.toc() <<" seconds."<<endl;
	
	// ########### Save diffraction volume ##############
	cout << "Saving diffraction volume..." << endl;
	for (int i = 0; i < mySize; i++) {
		std::stringstream sstm;
		sstm << output << "vol" << iter << "_" << setfill('0') << setw(7) << i << ".dat";
		string outputName = sstm.str();
		myIntensity->slice(i).save(outputName,raw_ascii);
	}

// KILL SLAVES
  	// Tell all the slaves to exit by sending an empty message with the DIETAG.
  	//if (iter == numIterations) {
		for (rank = 1; rank < numProcesses; ++rank) {
			comm->send(rank, DIETAG, msg);
			//cout << "Finsh working: " << rank << endl;
		}
	//}
}

static void slave_expansion(mpi::communicator* comm, int numImages, int mySize, string output, int useFileList, string input) {

	//int numImages = myImages->n_rows;
	int numChunkSlices = 0;
	int numPixels = mySize*mySize;
	boost::mpi::status status;
	const int master = 0;
	// Expansion related variables
	fvec quaternion(4);
	std::vector<int> msg;
	// Maximization related variables
	fmat diffraction = zeros<fmat>(mySize,mySize);
	fmat lse;
	fmat imgRep;
	fmat myExpansionChunk;
	uvec goodpixmap;
	int useRep = 0;
	fmat myDP;
	uvec::iterator goodBegin;
	uvec::iterator goodEnd;
	while (1) {

		// Receive a message from the master
    	status = comm->recv(master, boost::mpi::any_tag, msg);

		//cout << "slave rank: Got something " << comm->rank() << endl;
		if (status.tag() == GOODPIXTAG) {
    		goodpixmap.zeros(msg.size());
    		for(unsigned int i = 0; i < msg.size(); i++) {
    			goodpixmap.at(i) =  msg.at(i);
			}
			goodBegin = goodpixmap.begin();
			goodEnd = goodpixmap.end();
    	}
    		
    	// Calculate least squared error
    	if (status.tag() == DPTAG) {
    		fvec id = conv_to< fvec >::from(msg);
    		int startInd = (int) id(0);
    		int endInd = (int) id(1);
    		numChunkSlices = endInd - startInd + 1;
    		
    		myExpansionChunk.zeros(numChunkSlices,numPixels);
    		lse.zeros(numImages,numChunkSlices);
    					
    		std::ifstream infile;
			if (useFileList) {
				infile.open(input.c_str());
			}
		
			string line;
			string filename;
    		for (int i = 0; i < numImages; i++) {
				if (comm->rank() == 1) {
					cout << i << "/" << numImages << endl;
				}
				//Read in diffraction data
				if (useFileList) {
			  		std::getline(infile, line);
			  		myDP = load_asciiImage(line);
			  	} else {
				  	std::stringstream sstm;
			  		sstm << input << setfill('0') << setw(7) << i << ".dat";
					filename = sstm.str();
					myDP = load_asciiImage(filename);
				}
				
				if (useRep) {
					// replicate measured diffraction pattern
					imgRep.zeros(numChunkSlices,numPixels);
					imgRep = repmat(reshape(myDP,1,numPixels), numChunkSlices, 1);
					
					// read chunk of expansion slices
					int counter = 0;
					for(int j = startInd; j <= endInd; j++) {
						// Get image
						std::stringstream sstm;
						sstm << output << "myExpansion_" << setfill('0') << setw(7) << j << ".dat";
						string filename = sstm.str();
						fmat myExpansionSlice = load_asciiImage(filename);
						myExpansionChunk.row(counter) = reshape(myExpansionSlice, 1, numPixels);
						counter++;
					}
					
					// Compare measured diffraction with expansion slices
					lse.row(i) = trans(sum(pow(imgRep-myExpansionChunk,2),1));
				} else {
					// Compare measured diffraction with expansion slices
					int counter = 0;
					float sim;
					for(int j = startInd; j <= endInd; j++) {
						sim = 0;
						// Get image
						std::stringstream sstm;
						sstm << output << "myExpansion_" << setfill('0') << setw(7) << j << ".dat";
						string filename = sstm.str();
						fmat myExpansionSlice = load_asciiImage(filename);
						for(uvec::iterator k=goodBegin; k!=goodEnd; ++k) {
							sim += pow( myDP(*k) - myExpansionSlice(*k) ,2);
						}
						lse(i,counter) = sim;
						counter++;
					}
				}
			}
			// Send back LSE to master
			std::vector<float> msgLSE = conv_to< std::vector<float> >::from(vectorise(lse));
			comm->send(master, DONETAG, msgLSE);
    	}
	
		if (status.tag() == SAVESLICESTAG) {
			if (comm->rank() == 1) {
				cout << "Saving slices to file" << endl;
				// Get image
				std::stringstream sstm;
				sstm << output << "mySlice_" << setfill('0') << setw(7) << 1 << ".dat";
				string filename = sstm.str();
				for(uvec::iterator i=goodBegin; i!=goodEnd; ++i) {
					myDP(*i) = -1;//cout << myDP(*i) << endl;
				}
				myDP.save(filename,raw_ascii);
			}
		}

		if (status.tag() == SAVELSETAG) {
			cout << "Saving LSE to file" << endl;
			string outputName;
			stringstream sstm3;
			sstm3 << output << "lse_" << setfill('0') << setw(3) << comm->rank() << ".dat";
			outputName = sstm3.str();
			lse.save(outputName,raw_ascii);
		}

		if (status.tag() == DIETAG) {
			//cout << comm->rank() << ": I'm told to exit from my while loop" << endl;
		  	return;
		}

	}
}

