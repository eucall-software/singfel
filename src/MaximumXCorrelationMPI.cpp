/*
 * Program for merging diffraction patterns based on maximum cross correlations
 */
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <armadillo>
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
#define INDTAG 6 // image index signal
#define DONETAG 7 // done signal
static void master_expansion(mpi::communicator* comm, fmat* quaternions, fcube* myIntensity, fmat* pix, float pix_max, uvec* goodpix, fmat* myImages, int mySize, int iter, int numIterations, int numSlices, string output);
static void slave_expansion(mpi::communicator* comm, fmat* pix, float pix_max, uvec* goodpix, fmat* myDP, fmat* myImages, int mySize, string output);

int main( int argc, char* argv[] ){

	wall_clock timerMaster;

	// Initialize MPI
  	mpi::environment env;
  	mpi::communicator world;
	mpi::communicator* comm = &world;
	
	// All processes do this
		string imageList;
		string quaternionList;
		string beamFile;
		string geomFile;
		int master = 0;
		int numIterations = 0;
		int numImages = 0;
		int mySize = 0;
		int numSlices = 0;
		string output;
		// Let's parse input
		// image.lst and euler.lst are not neccessarily in the same order! So can not use like this. 		
		// Perhaps use hdf5 to combine the two.
		for (int n = 1; n < argc; n++) {
		cout << argv [ n ] << endl; 
		    if(boost::algorithm::iequals(argv[ n ], "-i")) {
		        imageList = argv[ n+1 ];
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
		    }
		}
		
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

		uvec goodpix;
		goodpix = det.get_goodPixelMap();
		//cout << "Good pix:" << goodpix << endl;

		double theta = atan((px/2*pix_height)/d);
		double qmax = 2/beam.get_wavelength()*sin(theta/2);
		double dmin = 1/(2*qmax);
		if (world.rank() == 0) {
			cout << "max q to the edge: " << qmax*1e-10 << " A^-1" << endl;
			cout << "Half period resolution: " << dmin*1e10 << " A" << endl;
		}
	
	// Initialize intensity volume
	string filename;
  	fmat myDP(py,px);
  	fmat myR;
  	myR.zeros(3,3);
	fcube myWeight;
	myWeight.zeros(mySize,mySize,mySize);
	fcube myIntensity;
	myIntensity.zeros(mySize,mySize,mySize);
	int numPixels = mySize * mySize;
	fmat myImages;
	myImages.zeros(numImages,numPixels);
	//fmat mySlices;
	//mySlices.zeros(numSlices,numPixels);
	
	float pix_max;
	int counter = 0;
	fmat pix;
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
    float inc_res = (mySize-1)/(2*pix_max);
    pix = pix * inc_res;
    pix_mod = sqrt(sum(pix%pix,1));		
	pix_max = max(pix_mod);

	timerMaster.tic();
		
	int active;
	string interpolate = "linear";
		
	fmat rot3D(3,3);
	fvec u(3);
	fvec quaternion(4);
	// Setup initial diffraction volume by merging randomly
	for (int r = 0; r < numImages; r++) {
	  	// Get image
	  	std::stringstream sstm;
  		sstm << imageList << setfill('0') << setw(6) << r << ".dat";
		filename = sstm.str();
		myDP = load_asciiImage(filename);
		//cout << "myDP: " << myDP.n_rows << "x" << myDP.n_cols << endl;
		//cout << "myDP: " << myDP(35,24) << endl;
			
		if (world.rank() == master) {
		    // Get rotation matrix
	  		u = randu<fvec>(3); // uniform random distribution in the [0,1] interval
			// generate uniform random quaternion on SO(3)
			quaternion << sqrt(1-u(0)) * sin(2*datum::pi*u(1)) << sqrt(1-u(0)) * cos(2*datum::pi*u(1))
					   << sqrt(u(0)) * sin(2*datum::pi*u(2)) << sqrt(u(0)) * cos(2*datum::pi*u(2));
			
			myR = CToolbox::quaternion2rot3D(quaternion);
			active = 1;
			CToolbox::merge3D(&myDP, &pix, &goodpix, &myR, pix_max, &myIntensity, &myWeight, active, interpolate);
		}
		myImages.row(r) = reshape(myDP,1,numPixels); // read along fs
		//cout << "myImageRow: " << myImages(r,0) << " " << myImages(r,1) << " " << myImages(r,2) << endl;
  	}
  	if (world.rank() == master) {
		cout << "Done random merge" << endl;
		// Normalize here
		CToolbox::normalize(&myIntensity,&myWeight);
		cout << "Done normalize" << endl;
  	}
	world.barrier();	
	if (world.rank() == master) {
		cout << "Random merge time: " << timerMaster.toc() <<" seconds."<<endl;
	}
	
	// Main iteration
	fmat myQuaternions;
	for (int iter = 0; iter < numIterations; iter++) { // number of iterations
		if (world.rank() == master) {
			if (iter == 0) {
				// Equal distribution of quaternions
  				myQuaternions = CToolbox::pointsOn4Sphere(numSlices);
  			}
  			//myQuaternions.print("myQuaternions:");
  			cout << "iteration: " << iter << endl;
			master_expansion(comm, &myQuaternions, &myIntensity, &pix, pix_max, &goodpix, &myImages, mySize, iter, numIterations, numSlices, output);
		} else {
			slave_expansion(comm, &pix, pix_max, &goodpix, &myDP, &myImages, mySize, output);
		}
		world.barrier();
		if (world.rank() == master) {
			cout << "Barrier passed" << endl; // " << mode << endl;
		}
	}

  	return 0;
}

static void master_expansion(mpi::communicator* comm, fmat* quaternions, fcube* myIntensity, fmat* pix, float pix_max, uvec* goodpix, fmat* myImages, int mySize, int iter, int numIterations, int numSlices, string output) {

	
	wall_clock timerMaster;
	
	// ########### EXPANSION ##############
	cout << "Start expansion" << endl;
	timerMaster.tic();

	int numImages = myImages->n_rows;
	
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
	int numPixels = mySize*mySize;
	fmat mySlices;
	mySlices.zeros(numSlices,numPixels);
	fmat myR;
	myR.zeros(3,3);
	fmat myDP(mySize,mySize);
	cout << "numSlices: " << numSlices << endl;
	cout << "numImages: " << numImages << endl;
	for (int s = 0; s < numSlices; s++) {
		// Get rotation matrix
		myR = CToolbox::quaternion2rot3D(trans(quaternions->row(s)));
		CToolbox::slice3D(&myDP, pix, goodpix, &myR, pix_max, myIntensity, active, interpolate);
		mySlices.row(s) = reshape(myDP,1,numPixels);
	}
	
	//cout << "Done mySlices" << endl;
	
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
	int startInd = 0;
	int endInd = 0;
	std::vector<float> msg;
	for (rank = 1; rank < numProcesses; ++rank) {
		endInd = startInd + s(rank-1) - 1;
		
		//cout << "rank:startInd:endInd " << rank << " " << startInd << " " << endInd << endl;
		
		// Vectorize my model slices
		int numElem = numPixels * s(rank-1);
		fvec modelVector( numElem );
		fmat myChunkSlices = mySlices.rows(startInd,endInd);
		for(int i = 0; i < numElem; i++) {
			modelVector(i) =  myChunkSlices.at(i); // column-wise
		}
		std::vector<float> model = conv_to< std::vector<float> >::from(modelVector);
		
		std::vector<float> id(2);
		id.at(0) = (float) startInd;
		id.at(1) = (float) endInd;
			
		comm->send(rank, INDTAG, id);
		comm->send(rank, MODELTAG, model);
		comm->send(rank, DPTAG, msg);
		
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
		numWorkerSlices = s(rank-1);//lse.n_elem / numImages;
		myTable.cols(currentCol,currentCol+s(rank-1)-1) = reshape(lse,numImages,numWorkerSlices);
		currentCol += s(rank-1);
	}
	
	// SAVE LSE FOR VIEWING
	// Tell all the slaves to save LSE by sending an empty message with the SAVELSETAG.
	//for (rank = 1; rank < numProcesses; ++rank) {
	//	comm->send(rank, SAVELSETAG, msg);
	//}
	
	// Master calculates the maximum for each diffraction pattern
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
	myIntensity->zeros(mySize,mySize,mySize);
	active = 1;
	interpolate = "linear";
	for (int r = 0; r < numImages; r++) {
		// Get image
		myDP = reshape(myImages->row(r),mySize,mySize);
		// Get rotation matrix
		myR = CToolbox::quaternion2rot3D(trans(quaternions->row(bestFit(r))));
		CToolbox::merge3D(&myDP, pix, goodpix, &myR, pix_max, myIntensity, &myWeight, active, interpolate);
	}
	// Normalize here
	CToolbox::normalize(myIntensity,&myWeight);
	
	cout << "Compression time: " << timerMaster.toc() <<" seconds."<<endl;
	
	// ########### Save diffraction volume ##############
	for (int i = 0; i < mySize; i++) {
		std::stringstream sstm;
		sstm << output << "vol" << iter << "_" << setfill('0') << setw(6) << i << ".dat";
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

static void slave_expansion(mpi::communicator* comm, fmat* pix, float pix_max, uvec* goodpix, fmat* myDP, fmat* myImages, int mySize, string output) {

	int numImages = myImages->n_rows;
	int numChunkSlices = 0;
	int numPixels = myDP->n_elem;
	//float work;
	//float results;
	boost::mpi::status status;//MPI_Status status;
	const int master = 0;
	// Expansion related variables
	fvec quaternion(4);
	std::vector<float> msg; //std::string msg;
	// Maximization related variables
	fmat diffraction = zeros<fmat>(myDP->n_rows,myDP->n_cols);
	fmat lse;
	fmat imgRep;
	fmat myChunkSlices;
	fcube myIntensity;
	
	while (1) {

		// Receive a message from the master
    	status = comm->recv(master, boost::mpi::any_tag, msg);

		// Receive how many slices assigned to this slave
    	if (status.tag() == INDTAG) {
    		fvec id = conv_to< fvec >::from(msg);
    		int startInd = (int) id(0);
    		int endInd = (int) id(1);
    		numChunkSlices = endInd - startInd + 1;
    	}
  	
		// Receive a subset of model slices
    	if (status.tag() == MODELTAG) {
    		myChunkSlices.zeros(numChunkSlices,numPixels);
    		for(unsigned int i = 0; i < msg.size(); i++) {
    			myChunkSlices.at(i) =  msg.at(i); // column-wise
			}
    	}
 	
    	// Calculate least squared error
    	if (status.tag() == DPTAG) {
    		imgRep.zeros(numChunkSlices,numPixels);
    		lse.zeros(numImages,numChunkSlices);
    		for (int i = 0; i < numImages; i++) {
				imgRep = repmat(myImages->row(i), numChunkSlices, 1);
				lse.row(i) = trans(sum(pow(imgRep-myChunkSlices,2),1));
			}
			// Send back LSE to master
			std::vector<float> msgLSE = conv_to< std::vector<float> >::from(vectorise(lse));
			comm->send(master, DONETAG, msgLSE);
    	}
    	
    	//cout << "\033[3;32mSlave: \033[0m" << msg << endl;
    	/*if (comm->rank() == 1) {
			std::cout << "The contents of msg:";
	  		for (std::vector<float>::iterator it = msg.begin(); it != msg.end(); ++it)
				std::cout << ' ' << *it;
	  		std::cout << '\n';    	
			std::cout.flush();
    	}*/
	
		if (status.tag() == SAVESLICESTAG) {
			cout << "Saving slices to file" << endl;
			string outputName;
			stringstream sstm3;
			sstm3 << output << "mySlices_" << setfill('0') << setw(3) << comm->rank() << ".dat";
			outputName = sstm3.str();
			myChunkSlices.save(outputName,raw_ascii);
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

