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

#define QUATERNIONTAG 1 // quaternion
#define VOLTAG 3	// intensity volume
#define PIXTAG 4	// qx,qy,qz for all pixels
#define GOODPIXTAG 5 // good pixels
#define DPTAG 6	// diffraction pattern
#define DIETAG 2 // die signal
#define DONETAG 7 // done signal
//static void master(mpi::communicator* comm, int mode);
//static void slave(mpi::communicator* comm, int mode);
static void master_expansion(mpi::communicator* comm, fmat* quaternions, fcube* myIntensity, fmat* pix, float pix_max, uvec* goodpix);
static void slave_expansion(mpi::communicator* comm, fcube* myIntensity, fmat* pix, float pix_max, uvec* goodpix, fmat* myDP);
static void master_maximization(mpi::communicator* comm);
static void slave_maximization(mpi::communicator* comm);
static void master_compression(mpi::communicator* comm);
static void slave_compression(mpi::communicator* comm);
static int get_next_work_item(fmat* workPool, int workID);
static void process_results(float result);
static int do_work(fmat* pix, uvec* goodpix, int pix_max, fcube* myIntensity, fvec* quaternion, fmat* myDP);

int main( int argc, char* argv[] ){

	// Initialize MPI
  	mpi::environment env;
  	mpi::communicator world;
	mpi::communicator* comm = &world;
	
	// All processes do this
		string imageList;
		string quaternionList;
		string beamFile;
		string geomFile;
		int numIterations = 0;
		int numImages = 0;
		int mySize = 0;
		int numSlices = 0;
		string output;
		// Let's parse input
		// image.lst and euler.lst are not neccessarily in the same order! So can not use like this. 		// Perhaps use hdf5 to combine the two.
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
		                cout << temp << endl;
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
	fmat mySlices;
	mySlices.zeros(numSlices,numPixels);
	
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
		
	if (world.rank() == 0) {
				
		cout << "Start" << endl;
		cout << mySize << endl;
		
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
			cout << "myDP: " << myDP.n_rows << "x" << myDP.n_cols << endl;
			//cout << "myDP: " << myDP(35,24) << endl;
			
	        // Get rotation matrix
  			u = randu<fvec>(3); // uniform random distribution in the [0,1] interval
			// generate uniform random quaternion on SO(3)
			quaternion << sqrt(1-u(0)) * sin(2*datum::pi*u(1)) << sqrt(1-u(0)) * cos(2*datum::pi*u(1))
					   << sqrt(u(0)) * sin(2*datum::pi*u(2)) << sqrt(u(0)) * cos(2*datum::pi*u(2));
			
			myR = CToolbox::quaternion2rot3D(quaternion);
			active = 1;
			CToolbox::merge3D(&myDP, &pix, &goodpix, &myR, pix_max, &myIntensity, &myWeight, active, interpolate);
			myImages.row(r) = reshape(myDP,1,numPixels); // read along fs
			//cout << "myImageRow: " << myImages(r,0) << " " << myImages(r,1) << " " << myImages(r,2) << endl;
  		}
  		cout << "Done random merge" << endl;
  		// Normalize here
  		CToolbox::normalize(&myIntensity,&myWeight);
  		cout << "Done normalize" << endl;
	}
	
	// Main iteration
	for (int iter = 0; iter < numIterations; iter++) { // number of iterations
		//for (int mode = 1; mode <= 3; mode++) { // loop through 3 steps
			if (world.rank() == 0) {
				// Distribution of quaternions
  				fmat myQuaternions = CToolbox::pointsOn4Sphere(numSlices);
  				//myQuaternions.print("myQuaternions:");
				master_expansion(comm, &myQuaternions, &myIntensity, &pix, pix_max, &goodpix);
		  	} else {
		  		slave_expansion(comm, &myIntensity, &pix, pix_max, &goodpix, &myDP);
		  	}
		  	world.barrier();
		  	if (world.rank() == 0) {
		  		cout << "Barrier passed" << endl; // " << mode << endl;
		  	}
	  	//}
	}

	cout << "Enter the dragon" << endl;

  	return 0;
/*


	if(!USE_CUDA) {

  		// Distribution of quaternions
  		fmat myQuaternions = CToolbox::pointsOn4Sphere(numSlices);
  		//myQuaternions.print("4Sphere: ");
  		cout << "Done 4Sphere" << endl;
  		//myQuaternions.print("Q: ");
  		
  		for (int iter = 0; iter < 1; iter++) {
  		
	  		// Expansion
			for (int s = 0; s < numSlices; s++) {
				//cout << s << endl;
			    // Get rotation matrix
				myR = CToolbox::quaternion2rot3D(trans(myQuaternions.row(s)));
				//cout << "Got myR" << endl;
				active = 1;
				CToolbox::slice3D(&myDP, &pix, &goodpix, &myR, pix_max, &myIntensity, active, interpolate);
				
				mySlices.row(s) = reshape(myDP,1,numPixels);
	  		}

	  		// Maximization
	  		cout << "Calculating similarity metric" << endl;
	  		fmat lse(numImages,numSlices); // least squared error
	  		fmat imgRep;
			imgRep.zeros(numSlices,numPixels);
	  		for (int i = 0; i < numImages; i++) {
	  			cout << i << endl;
	  			
	  			imgRep = repmat(myImages.row(i), numSlices, 1);
	  				  			
	  			lse.row(i) = trans(sum(pow(imgRep-mySlices,2),1));
	  		}
	  		uvec bestFit(numImages);
	  		//lse.print("lse:");
	  		float lowest;
	  		for (int i = 0; i < numImages; i++) {
	  			lowest = lse(i,0);
	  			bestFit(i) = 0;
				for (int j = 0; j < numSlices; j++) {
					if (lowest > lse(i,j)) {
						lowest = lse(i,j);
						bestFit(i) = j; // minimum lse index
					}
	  			}
	  		}
	  		//bestFit.print("bestFit: ");
	  		
	  		// Compression
	  		myWeight.zeros(mySize,mySize,mySize);
			myIntensity.zeros(mySize,mySize,mySize);
	  		for (int r = 0; r < numImages; r++) {
		  		// Get image
		  		std::stringstream sstm;
	  			sstm << imageList << setfill('0') << setw(6) << r << ".dat";
				filename = sstm.str();
				myDP = load_asciiImage(filename);
			    // Get rotation matrix
			    cout << myQuaternions.row(bestFit(r)) << endl;
				myR = CToolbox::quaternion2rot3D(trans(myQuaternions.row(bestFit(r))));
				active = 1;
				CToolbox::merge3D(&myDP, &pix, &goodpix, &myR, pix_max, &myIntensity, &myWeight, active, interpolate);
	  		}
	  		// Normalize here
	  		CToolbox::normalize(&myIntensity,&myWeight);
		
		
			// Save output
	  		fmat mySlice;
	  		for (int i = 0; i < mySize; i++) {
	  			std::stringstream sstm;
	  			sstm << output << setfill('0') << setw(6) << i << ".dat";
				string outputName = sstm.str();
				mySlice = myIntensity.slice(i).save(outputName,raw_ascii);
			}
			
		}
		
    }

	//cout << "Total time: " <<timerMaster.toc()<<" seconds."<<endl;

  	return 0;
*/
}

static void master_expansion(mpi::communicator* comm, fmat* quaternions, fcube* myIntensity, fmat* pix, float pix_max, uvec* goodpix) {
	cout << "Master in expansion mode" << endl;
	sleep(1);
	
	//cout << myIntensity->at(45,45,45) << endl;
	//cout << myIntensity->slice( 45 ) << endl;
	//cout << quaternions->row(3) << endl;
	
  	int ntasks, rank, numProcesses, numSlaves;
  	int numMaster = 1;
  	int work;
  	float result;
  	int workID = 0;
  	fvec quaternion;
  	boost::mpi::status status;//MPI_Status status;

	numProcesses = comm->size(); // 3
	numSlaves = numProcesses-1; // 2
	ntasks = quaternions->n_rows; // 5
	if (ntasks < numSlaves) { // 5 < 2
		numProcesses = ntasks + numMaster; // 
	}
	cout << "numProcesses: " << numProcesses << endl;
	// Find out how many processes there are in the default communicator
	//http://en.wikipedia.org/wiki/ANSI_escape_code#graphics
	// 033: Escape
	// 1: Effect = Bold
	// 30+1: Color = Red
	//cout << "\033[1;31mhow many processes?: \033[0m" << ntasks << endl;
  	//MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

	// Vectorise a cube into a vector (There must be a faster way to do this)
	cout.precision(5);
	cout.setf(ios::fixed);
	myIntensity->raw_print(cout, "intensity =");
	//myIntensity->print("myIntensity: ");
	
	fvec intensityVector(myIntensity->n_elem);
    for(int i = 0; i < myIntensity->n_elem; i++) {
    	intensityVector(i) =  myIntensity->at(i); // column-wise
	}
	cout.precision(5);
	cout.setf(ios::fixed);
	intensityVector.raw_print(cout, "intensityVector =");
	std::vector<float> data = conv_to< std::vector<float> >::from(intensityVector);
	//intensityVector.print("intensityVector: ");
	//std::vector<float> msg1 = conv_to< std::vector<float> >::from(a);

  	// Seed the slaves; send one unit of work to each slave.
	// Also, send merged intensity volume
  	for (rank = 1; rank < numProcesses; ++rank) {

    	// Find the next item of work to do

   		work = get_next_work_item(quaternions, workID);
		//cout << "work: " << work << endl;
		// Check if work is NULL
		if (work != NULL) {
			// Send it to each rank
			//fvec mvec = vectorise(myIntensity->slice(45));
			//std::vector<float> msg = conv_to< std::vector<float> >::from(mvec);
			//cout << rank << ": " << quaternions->row(workID) << endl;
			std::vector<float> quat = conv_to< std::vector<float> >::from(quaternions->row(workID));
			
			//comm->send(rank, WORKTAG, std::string("Hello"));
			comm->send(rank, QUATERNIONTAG, quat);
			comm->send(rank, VOLTAG, data);
			//MPI_Send(&work,             // message buffer 
		    //     1,                 // one data item 
		    //     MPI_INT,           // data item is an integer 
		    //     rank,              // destination process rank
		    //     WORKTAG,           // user chosen message tag 
		    //     MPI_COMM_WORLD);   // default communicator 
		    workID++;
        }
  	}

  	// Loop over getting new work requests until there is no more work to be done

  	work = get_next_work_item(quaternions, workID);
  	//cout << "work: " << work << endl;
  	
  	int msgDone;
  	while (work != NULL) {
		//cout << "Enter the while loop" << endl;
    	// Receive results from a slave
		//std::vector<float> msg; //std::string msg;
    	//status = comm->recv(boost::mpi::any_source, boost::mpi::any_tag, msg);
    	status = comm->recv(boost::mpi::any_source, DONETAG, msgDone);
    	//cout << "Received msg from: " << status.source() << endl;
    	//    std::cout << status.source() << ": The contents of msg> ";
	  	//	for (std::vector<float>::iterator it = msg.begin(); it != msg.end(); ++it)
		//		std::cout << ' ' << *it;
	  	//	std::cout << '\n';    	
		//	std::cout.flush();
    	
    	//MPI_Recv(&result,           // message buffer
        //     1,                 // one data item 
        //     MPI_DOUBLE,        // of type double real 
        //     MPI_ANY_SOURCE,    // receive from any sender 
        //     MPI_ANY_TAG,       // any type of message 
        //     MPI_COMM_WORLD,    // default communicator 
        //     &status);          // info about the received message 
		
		
    	// Send the slave a new work unit 
		std::vector<float> msgSend = conv_to< std::vector<float> >::from(quaternions->row(workID));
		workID++;
		
		comm->send(status.source(), QUATERNIONTAG, msgSend);
		comm->send(status.source(), VOLTAG, data);
    	
    	//MPI_Send(&work,             // message buffer
        //     1,                 // one data item 
        //     MPI_INT,           // data item is an integer 
        //     status.MPI_SOURCE, // to who we just received from 
        //     WORKTAG,           // user chosen message tag 
        //     MPI_COMM_WORLD);   // default communicator 
		
    	// Get the next unit of work to be done 

    	work = get_next_work_item(quaternions, workID);
    	//cout << "work: " << work << endl;
  	}

	cout << "Master finished sending all jobs" << endl;

  	// There's no more work to be done, so receive all the outstanding results from the slaves.

	
	for (rank = 1; rank < numProcesses; ++rank) {
    	status = comm->recv(rank, boost::mpi::any_tag, msgDone);
    	//std::cout << rank << ": The contents of msg> ";
	  	//	for (std::vector<float>::iterator it = msg.begin(); it != msg.end(); ++it)
		//		std::cout << ' ' << *it;
	  	//	std::cout << '\n';    	
		//	std::cout.flush();
		cout << "Received: " << rank << endl;
	}
  	//for (rank = 1; rank < ntasks; ++rank) {
    //	MPI_Recv(&result, 1, MPI_DOUBLE, MPI_ANY_SOURCE,
    //         MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  	//}

	std::vector<float> msg;
  	// Tell all the slaves to exit by sending an empty message with the DIETAG.
	for (rank = 1; rank < numProcesses; ++rank) {
		comm->send(rank, DIETAG, msg);
		//cout << "Finsh working: " << rank << endl;
	}
  	//for (rank = 1; rank < ntasks; ++rank) {
    //	MPI_Send(0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);
  	//}	

}

static void slave_expansion(mpi::communicator* comm, fcube* myIntensity, fmat* pix, float pix_max, uvec* goodpix, fmat* myDP) {
	cout << comm->rank() << ": Slave in expansion mode" << endl;
	sleep(1);
	
	float work;
	float results;
	boost::mpi::status status;//MPI_Status status;
	int master = 0;
	fvec quaternion(4);
	std::vector<float> msg; //std::string msg;
	
	while (1) {

		// Receive a message from the master
    	status = comm->recv(master, boost::mpi::any_tag, msg);
    	
    	if (status.tag() == QUATERNIONTAG) {
    		quaternion = conv_to< fvec >::from(msg);
    		quaternion.print("quaternion: ");
    	}
    	
    	if (status.tag() == VOLTAG) {
    		for(int i = 0; i < msg.size(); i++) {
    			myIntensity->at(i) =  msg.at(i); // column-wise
			}
    		cout.precision(5);
			cout.setf(ios::fixed);
			myIntensity->raw_print(cout, "SLAVE intensity =");
			
			// Do the work
			cout << "Enter the dragon" << endl;
			int result = do_work(pix, goodpix, pix_max, myIntensity, &quaternion, myDP);
			myDP->print("myDP: ");

			// Send the result back

			comm->send(master, DONETAG, 0);
			//MPI_Send(&result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			/*MPI_Send(&work,             // message buffer
             1,                 // one data item 
             MPI_INT,           // data item is an integer 
             status.MPI_SOURCE, // to who we just received from 
             WORKTAG,           // user chosen message tag 
             MPI_COMM_WORLD);   // default communicator 
             */
			//return;
    	}
    	//cout << comm->rank() << ": received msg from master" << endl;
    	//cout << "status tag: " << status.tag() << endl;
    	//cout << "status: " << status.source() << endl;
    	
    	//cout << "\033[3;32mSlave: \033[0m" << msg << endl;
    	/*if (comm->rank() == 1) {
			std::cout << "The contents of msg:";
	  		for (std::vector<float>::iterator it = msg.begin(); it != msg.end(); ++it)
				std::cout << ' ' << *it;
	  		std::cout << '\n';    	
			std::cout.flush();
    	}*/
    	
    	sleep(1);
    	
		//MPI_Recv(&work, 1, MPI_INT, 0, MPI_ANY_TAG,
		//         MPI_COMM_WORLD, &status);

		// Check the tag of the received message.
		
		if (status.tag() == DIETAG) {
			//cout << comm->rank() << ": I'm told to exit from my while loop" << endl;
		  	return;
		}

	}
}

static void master_maximization(mpi::communicator* comm) {
	cout << "Master in maximization mode" << endl;
	sleep(1);
}

static void slave_maximization(mpi::communicator* comm) {
	cout << comm->rank() << ": Slave in maximization mode" << endl;
	sleep(1);
}

static void master_compression(mpi::communicator* comm) {
	cout << "Master in compression mode" << endl;
	sleep(1);
}

static void slave_compression(mpi::communicator* comm) {
	cout << comm->rank() << ": Slave in compression mode" << endl;
	sleep(1);
}

static int get_next_work_item(fmat* workPool, int workID)
{
  /* Fill in with whatever is relevant to obtain a new unit of work
     suitable to be given to a slave. */
    if (workPool->n_rows > workID) {
		return 1;
    }
	return NULL;
}

static void 
process_results(float result)
{
  /* Fill in with whatever is relevant to process the results returned
     by the slave */
}

static int
do_work(fmat* pix, uvec* goodpix, int pix_max, fcube* myIntensity, fvec* quaternion, fmat* myDP) {
  /* Fill in with whatever is necessary to process the work and
     generate a result */
    
    //cout << "Do_work" << endl;
    fmat myR;
    //cout << quaternion->col(0) << endl;
	// Get rotation matrix
	myR = CToolbox::quaternion2rot3D(quaternion->col(0)); // transpose
	//cout << "Got myR" << endl;
	myR.print("myR: ");
	
	int active = 1;
	string interpolate = "linear";
	cout << "Got here" << endl;
	CToolbox::slice3D(myDP, pix, goodpix, &myR, pix_max, myIntensity, active, interpolate);
	
	cout << "Got here1" << endl;
	return 0;
}

