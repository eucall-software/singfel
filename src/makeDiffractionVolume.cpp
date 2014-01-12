/*
 * makeDiffractionVolume.cpp
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

using namespace std;

using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

//#define ARMA_NO_DEBUG

#define USE_CUDA 0
#define USE_CHUNK 0

int main( int argc, char* argv[] ){

	cout << "The name used to start the program: " << argv[ 0 ] << "\nArguments you have entered are:\n";
	if (argc != 2) {
		cout << "Please folder name containg diffraction patterns and .stream file" << endl;
	}
    for (int n = 1; n < argc; n++)
    	cout << setw( 2 ) << n << ": " << argv[ n ] << '\n';

	std::string folderName;
	folderName = argv[1];

	float inc_res = 3300;
	
	wall_clock timer, timer1, timer2, timer3, timerMaster;

	timerMaster.tic();

	/****** Beam ******/
	double lambda = 4.1328e-10; 				// (m) wavelength

	CBeam beam = CBeam();
	beam.set_wavelength(lambda);

	/****** Detector ******/
	double d = 8e-2;					// (m) detector distance
	double pix_width = 110e-6;			// (m)
	double pix_height = pix_width;		// (m)
	const int px = 1456;				// number of pixels in x
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
	det.init_dp(&beam);

	double theta = atan((px/2*pix_height)/d);
	double qmax = 2/lambda*sin(theta/2);
	double dmin = 1/(2*qmax);
	cout << "max q to the edge: " << qmax << " m^-1" << endl;
	cout << "Half period resolution: " << dmin << " m" << endl;

	if(!USE_CUDA) {
		cout << det.q_xyz(0,0,0) << endl; // 1456x1456x3

		
		int counter = 0;
		fmat pix;
		pix.zeros(det.numPix,3);
		for (int i = 0; i < px; i++) {
			for (int j = 0; j < py; j++) {
				pix(counter,0) = det.q_xyz(j,i,0);
				pix(counter,1) = det.q_xyz(j,i,1);
				pix(counter,2) = det.q_xyz(j,i,2);
				counter++;
			}
		}
		pix = pix * 1e-10 * inc_res; // (A^-1)
		
		cout << pix.row(197) << endl;
		fvec pix_mod = sqrt(sum(pix%pix,1));
		float pix_max = max(pix_mod);
		int mySize = 2*ceil(pix_max)+1;
		cout << "mySize: " << mySize <<endl;
		
		int folderNum = 4;
		string line;
		ifstream myfile ("/data/yoon/diffVol3D/crystfel4/1JB0.stream");
		int numCrystals = 0;
		if (myfile.is_open())
		{
			while ( getline (myfile,line) )
			{
				if (line.compare("--- Begin crystal") == 0) {
					numCrystals++;
				}
			}
			myfile.close();
		}
		else cout << "Unable to open file";
		
		cout << numCrystals << endl;
		
		fcube Astar;
		Astar.zeros(3,3,numCrystals);
		field<std::string> filenames(numCrystals);
		string subs;
		myfile.open ("/data/yoon/diffVol3D/crystfel4/1JB0.stream", std::ifstream::in);
		counter = 0;
		if (myfile.is_open()) {
			while ( getline (myfile,line) ) {
				if (line.compare("----- Begin chunk -----") == 0) {
		 			getline (myfile,line); // Image filename
		 			subs = line.substr(16);
		 		} else if (line.compare("--- Begin crystal") == 0) {
		 			filenames(counter) = subs;
		 			getline (myfile,line); // Cell parameters
		 			for (int i = 0; i < 3; i++) { // Loop over a*,b*,c*
			 			getline (myfile,line);
			 			subs = line.substr(8,18);
			 			Astar(i,0,counter) = atof(subs.c_str());
			 			subs = line.substr(19,29);
			 			Astar(i,1,counter) = atof(subs.c_str());
			 			subs = line.substr(30,40);
			 			Astar(i,2,counter) = atof(subs.c_str());
		 			}
		 			counter++;
		 		}
  			}
  			myfile.close();
  		}
  		cout << filenames(numCrystals-1) << endl;
		cout << Astar(2,2,numCrystals-1) << endl;
		
		// Compress
		fcube weight;
		weight.zeros(mySize,mySize,mySize);
		fcube intensity;
		intensity.zeros(mySize,mySize,mySize); 
		fmat origin;
		origin << 0.0355708 << 0.0205367 << 0.0000019 << endr
		       << 0.0000001 << 0.0410735 << 0.0000004 << endr
  		       << -0.0000029 << -0.0000005 << 0.0605860 << endr;
  		
  		int r = 0;
  		r = hdf_read(line,line);
  		//filenames(r)
  		//for (int r = 0; r < 1; r++) {

  		//}
		/*
		fmat F_hkl_sq;
		string outputName;

		CParticle rotatedParticle = CParticle();
		rotatedParticle = particle;
		fmat rot3D(3,3);
		//vec u(3);
		vec quaternion(4);	
		int i = 0;
		//for (int i = 0; i < numPatterns; i++) {
			// Rotate single particle
			//rot3D.zeros(3,3);			
			//u = randu<vec>(3); // uniform random distribution in the [0,1] interval
			//u << 0.501 << 0.31 << 0.82;
			// generate uniform random quaternion on SO(3)
			quaternion << q0 << q1 << q2 << q3;
			
			//cout << quaternion << endl;
			
			rot3D = CToolbox::quaternion2rot3D(quaternion, 1);
			
			//cout << rot3D << endl;

			fmat myPos = particle.get_atomPos();
			//cout << myPos << endl;
			
			myPos = myPos * trans(rot3D);	// rotate atom positions
			
			rotatedParticle.set_atomPos(&myPos);

			//cout << rotatedParticle.get_atomPos() << endl;

			F_hkl_sq = CDiffraction::calculate_intensity(&rotatedParticle,&det);
			//cout<<"Calculate F_hkl: Elapsed time is "<<timer.toc()<<" seconds."<<endl; // 14.25s
			//F_hkl_sq.save("../F_hkl_sq.dat",raw_ascii);
			//timer.tic();
			fmat detector_intensity = F_hkl_sq % det.solidAngle % det.thomson * beam.phi_in; //2.105e30
			umat detector_counts = CToolbox::convert_to_poisson(detector_intensity);		
			//cout<<"Calculate dp: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
			//det.solidAngle.save("../solidAngle.dat",raw_ascii);
			//timer.tic();
			stringstream sstm;
			sstm << "/data/yoon/singfel/dataMonster/diffraction_" << setfill('0') << setw(6) << i << ".dat";
			outputName = sstm.str();
			detector_counts.save(outputName,raw_ascii);
			stringstream sstm1;
			sstm1 << "/data/yoon/singfel/dataMonster/quaternion_" << setfill('0') << setw(6) << i << ".dat";
			outputName = sstm1.str();
			quaternion.save(outputName,raw_ascii);			
			//cout<<"Save image: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
		//}
		*/
	}

cout << "Total time: " <<timerMaster.toc()<<" seconds."<<endl;
  	return 0;
}

