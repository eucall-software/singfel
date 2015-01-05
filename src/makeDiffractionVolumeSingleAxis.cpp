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

#ifdef COMPILE_WITH_CXX11
	#define ARMA_DONT_USE_CXX11
#endif
#include <armadillo>

using namespace std;

using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

#define USE_CUDA 0

int main( int argc, char* argv[] ){

	cout << "The name used to start the program: " << argv[ 0 ] << "\nArguments you have entered are:\n";
	if (argc != 2) {
		cout << "Please folder name containg diffraction patterns and .stream file" << endl;
	}
    
    	for (int n = 1; n < argc; n++)
    		cout << setw( 2 ) << n << ": " << argv[ n ] << '\n';

	float inc_res = 30000;
	
	//std::string imageName = "/data/yoon/singfel/dataShrine/img1.dat";
	//fmat img;
	//img = load_asciiImage(imageName);
	//img.print("img:");
	//cout << img(81,64) << endl;
	//cout << img(64,81) << endl; // Matlab coord (82,65)

	/****** Beam ******/
	double lambda = 1.5497e-10; 				// (m) wavelength

	CBeam beam = CBeam();
	beam.set_wavelength(lambda);

	/****** Detector ******/
	double d = 4;					// (m) detector distance
	double pix_width = 150e-6;			// (m)
	double pix_height = pix_width;		// (m)
	const int px = 128;				// number of pixels in x
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
		for (int i = 0; i < px; i++) {
			for (int j = 0; j < py; j++) { // column-wise
				pix(counter,0) = det.q_xyz(j,i,0);
				pix(counter,1) = det.q_xyz(j,i,1);
				pix(counter,2) = det.q_xyz(j,i,2);
				counter++;
			}
		}
cout << "1/lambda: " << 1/lambda << endl;
cout << det.q_xyz(0,0,0) << endl;
cout << "***" << pix.row(0) << endl;
cout << "***" << pix.row(1) << endl;
cout << "***" << pix.row(2) << endl;
		pix = pix * 1e-10 * inc_res; // (A^-1)
	
		fvec pix_mod = sqrt(sum(pix%pix,1));
		float pix_max = max(pix_mod);
cout << "pix_max: " << pix_max << endl;
		int mySize = 2*ceil(pix_max)+1;
		
		// Determine number of image
		int numImages = 10000;
		
		cout << numImages << endl;
		
		// Read in stream file
		fcube Astar;
		Astar.zeros(3,3,numImages);
		
		for (int i = 0; i < numImages; i++) {
			std::stringstream sstm1;
			sstm1 << "/data/yoon/singfel/dataShrine/euler" << i+1 << ".dat";
			string eulerName = sstm1.str();
			//cout << eulerName << endl;

			fvec euler;
			euler = load_asciiEuler(eulerName);
			fmat R;
			R = zeros<fmat>(3,3);
			float phi = euler(0);
			float theta = euler(1);
			float psi = euler(2);
			cout << psi << endl;
			R(0,0) = cos(psi)*cos(phi) - cos(theta)*sin(phi)*sin(psi);
			R(0,1) = cos(psi)*sin(phi) + cos(theta)*cos(phi)*sin(psi);
			R(0,2) = sin(psi)*sin(theta);
			R(1,0) = -sin(psi)*cos(phi) - cos(theta)*sin(phi)*cos(psi);
			R(1,1) = -sin(psi)*sin(phi) + cos(theta)*cos(phi)*cos(psi);
			R(1,2) = cos(psi)*sin(theta);
			R(2,0) = sin(theta)*sin(phi);
			R(2,1) = -sin(theta)*cos(phi);
			R(2,2) = cos(theta);
			//cout << R << endl; 
			Astar.slice(i) = R;
			//R.print();
		}
		//cout << "R: " << Astar.slice(0) << endl;
		
		// Fill diffraction volume using Trilinear interpolation
		//fmat origin;
		//origin << 0.0355708 << 0.0205367 << 0.0000019 << endr
		//       << 0.0000001 << 0.0410735 << 0.0000004 << endr
  		//       << -0.0000029 << -0.0000005 << 0.0605860 << endr;
  		
  		string filename;
  		//string datasetname = "/data/data";
  		fmat myDP;
  		  		
  		mat B;
  		fmat temp;
  		fmat rot;
  		rot.zeros(3,3);
  		mat U;
		vec s;
		mat V;
		mat M;

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
		//cout << mySize << endl;
		
  		for (int r = 0; r < numImages; r++) {//(int r = 0; r < numImages; r++) {
	  		std::stringstream sstm;
  			sstm << "/data/yoon/singfel/dataShrine/img" << r+1 << ".dat";
			filename = sstm.str();
			//cout << filename << endl;
			myDP = load_asciiImage(filename);
	
  			//cout << "filename: "<< filenames(r) << endl;
  			//filename = "/data/yoon/diffVol3D/test_x/" + filenames(r);
			//filename = "/data/yoon/diffVol3D/crystfel"+ folderName +"/" + filenames(r);
			//myDP = hdf5read(filename,datasetname);
			
        		// Get rotation matrix
        		rot = Astar.slice(r);
        		//cout << "rot: " << rot << endl;
 
        		fvec myPhotons = vectorise(myDP); // column-wise
        		//cout << "DP: " << myPhotons(5313) << endl;
        	
        		// rotate and centre Ewald slice
        	//cout << pix << endl;
        	//fmat tempRotPix = conv_to<fmat>::from(rot)*trans(pix);
        	//cout << "RotPix: " << tempRotPix.col(5313) << endl;
        	
        		pixRot = conv_to<fmat>::from(rot)*trans(pix) + pix_max;
        	//cout << "pixRot: " << pixRot.col(5313) << endl;	 
        		cout << pixRot.n_rows << endl;
        		cout << pixRot.n_cols << endl;

        		xyz = conv_to<imat>::from(floor(pixRot));
        	//cout << "xyz: " << xyz.col(5313) << endl;	
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
				
				if (p == 5313) {
					cout << "photons: " << photons << endl;
					cout << rot << endl;
					cout << pix_max << endl;
					cout << "pix0: " << pix.row(0) << endl;
					cout << "pix1: " << pix.row(1) << endl;
					cout << "pix: " << pix.row(p) << endl;
					cout << "tx,ty,tz:" << pixRot.col(p) << endl;
					cout << "x: " << x <<","<< y <<","<< z << endl;
					cout << "fx: " << fx <<","<< fy <<","<< fz << endl;
				}
				
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
  		myIntensity.save("myIntensity.dat",raw_ascii);
      	}
      	// Normalize here
      	

	//cout << "Total time: " <<timerMaster.toc()<<" seconds."<<endl;

  	return 0;

}

