/*
 * makeDiffractionVolume.cpp
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

	std::string myStream = "/data/yoon/diffVol3D/test_x/1JB0.stream";

	std::string folderName;
	folderName = argv[1];

	float inc_res = 3268;
	
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

		int counter = 0;
		fmat pix;
		pix.zeros(det.numPix,3);
		//imat pixInd;
		//pixInd.zeros(det.numPix,2);
		for (int i = 0; i < px; i++) {
			for (int j = 0; j < py; j++) { // column-wise
				pix(counter,0) = det.q_xyz(j,i,0);
				pix(counter,1) = det.q_xyz(j,i,1);
				pix(counter,2) = det.q_xyz(j,i,2);
				//pixInd(counter,0) = j;
				//pixInd(counter,1) = i;
				counter++;
		
			}
		}

		pix = pix * 1e-10 * inc_res; // (A^-1)
	
		fvec pix_mod = sqrt(sum(pix%pix,1));
		float pix_max = max(pix_mod);
		int mySize = 2*ceil(pix_max)+1;
		
		// Determine numCrystals in stream file
		int folderNum = 4;
		string line;
		ifstream myfile (myStream.c_str());
		//ifstream myfile ("/data/yoon/diffVol3D/crystfel4/1JB0.stream");
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
		
		// Read in stream file
		fcube Astar;
		Astar.zeros(3,3,numCrystals);
		field<std::string> filenames(numCrystals);
		string subs;
		myfile.open (myStream.c_str(), std::ifstream::in);
		//myfile.open ("/data/yoon/diffVol3D/crystfel4/1JB0.stream", std::ifstream::in);
		counter = 0;
		if (myfile.is_open()) {
			while ( getline (myfile,line) ) {
				if (line.compare("----- Begin chunk -----") == 0) {
		 			getline (myfile,line); // Image filename
		 			subs = line.substr(16);
		 		} else if (line.compare("--- Begin crystal") == 0) {
		 			filenames(counter) = subs;
		 			
		 			cout << subs << endl;
		 			
		 			getline (myfile,line); // Cell parameters
		 			for (int i = 0; i < 3; i++) { // Loop over a*,b*,c*
			 			getline (myfile,line);
			 			subs = line.substr(8,18);
			 			Astar(i,0,counter) = atof(subs.c_str());
			 			//cout << subs << endl;
			 			subs = line.substr(19,29);
			 			Astar(i,1,counter) = atof(subs.c_str());
			 			//cout << subs << endl;
			 			subs = line.substr(30,40);
			 			Astar(i,2,counter) = atof(subs.c_str());
			 			//cout << subs << endl;
		 			}
		 			counter++;
		 		}
  			}
  			myfile.close();
  		}
		
		// Fill diffraction volume using Trilinear interpolation
		fmat origin;
		origin << 0.0355708 << 0.0205367 << 0.0000019 << endr
		       << 0.0000001 << 0.0410735 << 0.0000004 << endr
  		       << -0.0000029 << -0.0000005 << 0.0605860 << endr;
  		
  		string filename;
  		string datasetname = "/data/data";
  		fmat myDP;
  		  		
  		mat B;
  		fmat temp;
  		fmat originRot;
  		originRot.zeros(3,3);
  		mat U;
		vec s;
		mat V;
		mat M;
		mat rot;
		fmat pixRot;
		pixRot.zeros(det.numPix,3);
		imat xyz;
		xyz.zeros(det.numPix,3);
		fmat fxyz;
		fxyz.zeros(det.numPix,3);
		fmat cxyz;
		cxyz.zeros(det.numPix,3);
		float weight;
		float photons;
		float x,y,z,fx,fy,fz,cx,cy,cz;
		fcube myWeight;
		myWeight.zeros(mySize,mySize,mySize);
		fcube myIntensity;
		myIntensity.zeros(mySize,mySize,mySize);
		
		cout << "Start" << endl;
		
  		for (int r = 0; r < 1; r++) {//(int r = 0; r < numCrystals; r++) {
  			cout << "filename: "<< filenames(r) << endl;
  			filename = "/data/yoon/diffVol3D/test_x/" + filenames(r);
			//filename = "/data/yoon/diffVol3D/crystfel"+ folderName +"/" + filenames(r);
			myDP = hdf5read(filename,datasetname);
			
        	// Form rotation matrix by solving Wahba's problem
        	originRot = Astar.slice(r);
        	cout <<"originRot: "<< Astar.slice(r) << endl;
        	temp.zeros(3,3);
        	for (int i = 0; i < 3; i++) {
        		temp += trans(origin.row(i)) * originRot.row(i);
        	}
        	B = conv_to<mat>::from(temp);
        	svd(U,s,V,B);
        	M.eye(3,3);
        	M(2,2) = arma::det(U)*arma::det(trans(V));
        	rot = U*M*trans(V);
        	cout << "rot: " << rot << endl;
 
        	fvec myPhotons = vectorise(myDP); // column-wise
        	cout << "DP: " << myPhotons(92) << endl;
        	
        	// rotate and centre Ewald slice
        	pixRot = conv_to<fmat>::from(rot)*trans(pix) + pix_max; 
        	xyz = conv_to<imat>::from(floor(pixRot));
        	fxyz = pixRot - xyz;
        	cxyz = 1. - fxyz;
        	for (int p = 0; p < det.numPix; p++) {

        		photons = myPhotons(p);
        		
        		x = xyz(0,p);
        		y = xyz(1,p);
        		z = xyz(2,p);

        		if (x >= mySize-1 || y >= mySize-1 || z >= mySize-1)
        			continue;
        			
        		if (x < 0 || y < 0 || z < 0)
        			continue;	
        		
        		fx = fxyz(0,p);
        		fy = fxyz(1,p);
        		fz = fxyz(2,p);
        		cx = cxyz(0,p);
        		cy = cxyz(1,p);
        		cz = cxyz(2,p);
        		
        		weight = cx*cy*cz;
        		myWeight(y,x,z) += weight;
        		myIntensity(y,x,z) += weight * photons;

        		weight = cx*cy*fz;
        		myWeight(y,x,z+1) += weight;
        		myIntensity(y,x,z+1) += weight * photons; 

        		weight = cx*fy*cz;
        		myWeight(y+1,x,z) += weight;
        		myIntensity(y+1,x,z) += weight * photons; 

        		weight = cx*fy*fz;
        		myWeight(y+1,x,z+1) += weight;
        		myIntensity(y+1,x,z+1) += weight * photons;         		        
        		
        		weight = fx*cy*cz;
        		myWeight(y,x+1,z) += weight;
        		myIntensity(y,x+1,z) += weight * photons; 		       		 

        		weight = fx*cy*fz;
        		myWeight(y,x+1,z+1) += weight;
        		myIntensity(y,x+1,z+1) += weight * photons; 

        		weight = fx*fy*cz;
        		myWeight(y+1,x+1,z) += weight;
        		myIntensity(y+1,x+1,z) += weight * photons;

        		weight = fx*fy*fz;
        		myWeight(y+1,x+1,z+1) += weight;
        		myIntensity(y+1,x+1,z+1) += weight * photons;

        	}
  		}
  		
  		int i,j;
  		for (j = 814; j < 817; j++)
   	  	{
      	for (i = 365; i < 368; i++)
      		{
	    		cout << myIntensity(j,i,724) << " ";
      		}
      		cout << endl;
      	}
      
      	fcube intens;
      	intens.zeros(mySize,mySize,mySize);
      	int numSym = 6; // 1JB0 in-plane symmetry
      	fcube mySlice;
      	mySlice.zeros(mySize,mySize,numSym);
      	fcube sliceWeight;
      	sliceWeight.zeros(mySize,mySize,numSym);
      	for (i = 0; i < mySize; i++) {
      		for (j = 0; j < numSym; j++) {
      			
      		}
      	}
  		//uvec ind = find(myWeight > 0); // Arma 4.0
  		//myIntensity(ind) /= myWeight(ind);
  		/*
  		for (int i = 0; i < det.numPix; i++) {
  			if (myWeight(i) != 0.) {
  				myIntensity(i) /= myWeight(i);
  			}
  		}
  		*/
		/*
  		cube myIntensity1 = conv_to<cube>::from(myIntensity);
  		cube myWeight1 = conv_to<cube>::from(myWeight);
  		myIntensity1.save("myIntensity.mat",raw_ascii);
		myWeight1.save("myWeight.mat",raw_ascii);
		*/
	}

cout << "Total time: " <<timerMaster.toc()<<" seconds."<<endl;

  	return 0;
}

