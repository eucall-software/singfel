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
static void master_recon(mpi::communicator* comm, fcube* myRot, fcube* myIntensity, fcube* myWeight, fmat* pix, float pix_max, uvec* goodpix, int numImages, int mySize, int iter, int startIter, int numIterations, int numSlices, string intput, string output, string format);
static void slave_recon(mpi::communicator* comm, int numImages, int mySize, string output, int useFileList, string input, string format, int iter);

int main( int argc, char* argv[] ){

	wall_clock timerMaster;

	// Initialize MPI
  	mpi::environment env;
  	mpi::communicator world;
	mpi::communicator* comm = &world;

	srand( world.rank() );

	// Every worker parses input
	int master = 0;
	string input;
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
	string format;
	string rotationAxis = "xyz";
	// Let's parse input
	for (int n = 1; n < argc; n++) {
		if (world.rank() == master) {
			cout << argv [ n ] << endl;
		}
	    if(boost::algorithm::iequals(argv[ n ], "-i")) {
	        input = argv[ n+1 ];
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
		} else if (boost::algorithm::iequals(argv[ n ], "--format")) {
            format = argv[ n+2 ];
        } else if (boost::algorithm::iequals(argv[ n ], "--rotationAxis")) {
            rotationAxis = argv[ n+2 ];
        }
	}

	int numSlaves = comm->size()-1;
	if (numImages < numSlaves) {
		cerr << "Number of workers too large for this task" << endl;
		return 0;
	}
	
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
		myWeight.zeros(mySize,mySize,mySize);
		myIntensity.zeros(mySize,mySize,mySize);

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
	
		if (world.rank() == master) {
			if ( strcmp(intialVolume.c_str(),"randomMerge")==0 ) {
				cout << "Randomly merging diffraction volume..." << endl;
				// Setup initial diffraction volume by merging randomly
				// rotationAxis determines the random nature of the angles
				for (int r = 0; r < numImages; r++) {
				  	// Get image
				  	if (format == "S2E") {
				  		std::stringstream sstm;
				  		sstm << input << "diffr/diffr_out_" << setfill('0') << setw(7) << r+1 << ".h5";
						filename = sstm.str();
						myDP = hdf5readT<fmat>(filename,"/data/data");
				  	} else if (format == "list") {
				  		std::getline(infile, line);
				  		myDP = load_asciiImage(line);
				  	} else {
					  	std::stringstream sstm;
				  		sstm << input << "diffr/diffr_out_" << setfill('0') << setw(7) << r+1 << ".dat";
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
				for (int i = 0; i < mySize; i++) {
					std::stringstream sstm;
					sstm << intialVolume << setfill('0') << setw(7) << i << ".dat";
					string outputName = sstm.str();
					myIntensity.slice(i) = load_asciiImage(outputName);
				}
			}	
		}
	}

	world.barrier();
	if (world.rank() == master) {
		cout << "Initialization time: " << timerMaster.toc() <<" seconds."<<endl;
	}
	
	// Main iteration
	fmat myQuaternions;
	fcube myRot(3,3,numSlices);
	fmat myR;
	for (int iter = startIter; iter < numIterations; iter++) { // number of iterations
		if (world.rank() == master) {
			if (iter == startIter) {
				// Equal distribution of quaternions
				if (rotationAxis == "y" || rotationAxis == "z") {
					myQuaternions = CToolbox::pointsOn1Sphere(numSlices, rotationAxis);
				} else {
  					myQuaternions = CToolbox::pointsOn4Sphere(numSlices);
  				}
				for (int i = 0; i < numSlices; i++) {
					myR = CToolbox::quaternion2rot3D(trans(myQuaternions.row(i)));
					myRot.slice(i) = myR;
				}
  			}
			master_recon(comm, &myRot, &myIntensity, &myWeight, &pix, pix_max, &goodpix, numImages, mySize, iter, startIter, numIterations, numSlices, input, output, format);
		} else {
			slave_recon(comm, numImages, mySize, output, useFileList, input, format, iter);
		}
		world.barrier();
	}
  	return 0;
}

static void master_recon(mpi::communicator* comm, fcube* myRot, fcube* myIntensity, fcube* myWeight, fmat* pix, float pix_max, uvec* goodpix, int numImages, int mySize, int iter, int startIter, int numIterations, int numSlices, string input, string output, string format) {

	wall_clock timerMaster;

  	int rank, numProcesses, numSlaves;
  	fvec quaternion;
  	//boost::mpi::status status;

	numProcesses = comm->size();
	numSlaves = numProcesses-1;

	// ########### EXPANSION ##############
	cout << "Start expansion" << endl;
	timerMaster.tic();
	
	CToolbox::expansion(numSlices, myRot, mySize, pix, goodpix, pix_max, myIntensity, output, iter);
	
	cout << "Expansion time: " << timerMaster.toc() <<" seconds."<<endl;

	// ########### MAXIMIZATION ##############	
	cout << "Start maximization" << endl;
	timerMaster.tic();

	//////////////////////
	// Number of data candidates to update expansion slice
	int numCandidates = 2;
	//////////////////////
	CToolbox::maximization(comm, numImages, numSlaves, goodpix, numSlices, numProcesses, numCandidates, output, mySize, format, input, iter);

	cout << "Maximization time: " << timerMaster.toc() <<" seconds."<<endl;

	// ########### COMPRESSION ##############
	cout << "Start compression" << endl;
	timerMaster.tic();

	CToolbox::compression(mySize, myIntensity, myWeight, numSlices, output, pix, pix_max, myRot, format, iter);

	cout << "Compression time: " << timerMaster.toc() <<" seconds."<<endl;
	
	// ########### Save diffraction volume ##############
	cout << "Saving diffraction volume..." << endl;
	timerMaster.tic();

	CToolbox::saveDiffractionVolume(mySize, output, myIntensity, myWeight, numProcesses, iter);

	cout << "Save time: " << timerMaster.toc() <<" seconds."<<endl;

	// ########### Reset workers ##############
  	// Tell all the slaves to exit by sending an empty message with the DIETAG.
  	std::vector<int> msg1;
	for (rank = 1; rank < numProcesses; ++rank) {
		comm->send(rank, DIETAG, msg1);
	}

	cout << "Done iteration: " << iter << endl;
}

static void slave_recon(mpi::communicator* comm, int numImages, int mySize, string output, int useFileList, string input, string format, int iter) {
	int numChunkData = 0;
	boost::mpi::status status;
	const int master = 0;
	// Expansion related variables
	fvec quaternion(4);
	std::vector<int> msg;
	// Maximization related variables
	fmat diffraction = zeros<fmat>(mySize,mySize);
	fvec condProb; // conditional probability
	fmat imgRep;
	uvec goodpixmap;
	fmat myDP;
	while (1) {

		// Receive a message from the master
    	status = comm->recv(master, boost::mpi::any_tag, msg);
    		
    	// Calculate least squared error
    	if (status.tag() == DPTAG) {
	    	fvec id = conv_to< fvec >::from(msg);
	    	int startInd = (int) id(0); // start index of measured data
	    	int endInd = (int) id(1); // end index of measured data
	    	numChunkData = endInd - startInd + 1; // number of measured data to process
	    	int expansionInd = (int) id(2);
	    		
	    	// Initialize
	    	condProb.zeros(numChunkData);
    		
	    	//////////////////////////
	    	// Read in expansion slice
	    	//////////////////////////
			// Get expansion image
			std::stringstream sstm;
			sstm << output << "expansion/myExpansion" << iter << "_" << setfill('0') << setw(7) << expansionInd << ".dat";
			string filename = sstm.str();
			fmat myExpansionSlice = load_asciiImage(filename);
			// Get expansion pixmap
			std::stringstream sstm1;
			sstm1 << output << "expansion/myExpansionPixmap" << iter << "_" << setfill('0') << setw(7) << expansionInd << ".dat";
			string filename1 = sstm1.str();
			fmat myPixmap = load_asciiImage(filename1);
						    		
	    	///////////////
	    	// Read in data
	    	///////////////
			string line;
			int counter = 0;
	    	for (int i = startInd; i <= endInd; i++) {
				//Read in measured diffraction data
				if (format == "S2E") {
			  		std::stringstream sstm;
			  		sstm << input << "diffr/diffr_out_" << setfill('0') << setw(7) << i+1 << ".h5";
					filename = sstm.str();
					myDP = hdf5readT<fmat>(filename,"/data/data");
			  	} else if (format == "list") {
				  	myDP = load_readNthLine(input, i);
			  	} else {
				  	std::stringstream sstm;
			  		sstm << input << "diffr/diffr_out_" << setfill('0') << setw(7) << i+1 << ".dat";
					filename = sstm.str();
					myDP = load_asciiImage(filename);
				}
				/////////////////////////////////////////////////////
				// Compare measured diffraction with expansion slices
				/////////////////////////////////////////////////////
				string type = "euclidean";
				//string type = "gaussian";
				double val = 0.0;
				//if (expansionInd == 0 && i == 0) { 
					val = CToolbox::calculateSimilarity(&myExpansionSlice, &myDP, &myPixmap, type);
				//}
				
				condProb(counter) = float(val);
				counter++;
			}
			// Send back conditional probability to master
			std::vector<float> msgProb = conv_to< std::vector<float> >::from(vectorise(condProb));
			comm->send(master, DONETAG, msgProb);
    	}

/*	
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
*/
/*
		if (status.tag() == SAVELSETAG) {
			cout << "Saving LSE to file" << endl;
			string outputName;
			stringstream sstm3;
			sstm3 << output << "similarity/lse_" << setfill('0') << setw(7) << comm->rank() << ".dat";
			outputName = sstm3.str();
			condProb.save(outputName,raw_ascii);
		}
*/
		if (status.tag() == DIETAG) {
			//cout << comm->rank() << ": I'm told to exit from my while loop" << endl;
		  	return;
		}

	}
}

