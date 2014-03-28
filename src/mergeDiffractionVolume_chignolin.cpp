/*
 * THIS SHOULD BE RENAMED AS PROCESS_HKL.CPP
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

using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

#define USE_CUDA 0

int main( int argc, char* argv[] ){

    string imageList;
    string quaternionList;
    string beamFile;
    string geomFile;
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
	cout << "max q to the edge: " << qmax*1e-10 << " A^-1" << endl;
	cout << "Half period resolution: " << dmin*1e10 << " A" << endl;

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
        fvec pix_mod;
		float pix_max;
		pix = pix * 1e-10; // (nm)
		pix_mod = sqrt(sum(pix%pix,1));		
		pix_max = max(pix_mod);
        //cout << "pix_max: " << pix_max << endl;
        float inc_res = (mySize-1)/(2*pix_max);
        pix = pix * inc_res;
        pix_mod = sqrt(sum(pix%pix,1));		
		pix_max = max(pix_mod);
		
  		string filename;
  		fmat myDP;
  		fmat myR;
  		myR.zeros(3,3);
  		//float psi,theta,phi;
		fcube myWeight;
		myWeight.zeros(mySize,mySize,mySize);
		fcube myIntensity;
		myIntensity.zeros(mySize,mySize,mySize);
		
		cout << "Start" << endl;
		//cout << mySize << endl;
		
		int active = 1;
		string interpolate = "linear";
		//string interpolate = "nearest";
		fmat origin;
		origin << 1 << 0 << 0 << endr
			<< 0 << 1 << 0 << endr
			<< 0 << 0 << 1 << endr;
  		for (int r = 0; r < numImages; r++) {//(int r = 0; r < numImages; r++) {
	  		// Get image
	  		std::stringstream sstm;
  			sstm << imageList << setfill('0') << setw(6) << r << ".dat";
			filename = sstm.str();
			myDP = load_asciiImage(filename);
			//myDet.apply_badPixels();
			cout << filename << endl;
			cout << "myDP(0): " << myDP(0) << endl; // 0
			cout << "myDP(71): " << myDP(71) << endl; // 2.94
			//cout << "myDP(130): " << myDP(130) << endl; // 5.08 <-- mask out
	        // Get rotation matrix
  			std::stringstream sstm1;
			sstm1 << quaternionList << setfill('0') << setw(6) << r << ".dat";
			string quaternionName = sstm1.str();
			fvec quaternion;
			quaternion = load_asciiQuaternion(quaternionName);
			
			/*
			if (r==0){
				quaternion = CToolbox::euler2quaternion(0,0,0*datum::pi/180);//
				//quaternion << 1 << 0 << 0 << 0;
				//myR = CToolbox::euler2rot3D(0,0,0*datum::pi/180);
			}else if (r==1){
				quaternion = CToolbox::euler2quaternion(0,0,15*datum::pi/180);//
				//quaternion << 0.7071 << 0 << 0 << -0.7071;
				//myR = CToolbox::euler2rot3D(0,0,15*datum::pi/180);
			}else if (r==2){
				quaternion = CToolbox::euler2quaternion(0,0,30*datum::pi/180);//
				//quaternion << 0.7071 << 0 << 0 << -0.7071;
				//myR = CToolbox::euler2rot3D(0,0,30*datum::pi/180);
			}else if (r==3){
				quaternion = CToolbox::euler2quaternion(0,0,45*datum::pi/180);//quaternion << 0. << 0 << 0 << 1;	
				//myR = CToolbox::euler2rot3D(0,0,45*datum::pi/180);
			}else if (r==4){
				quaternion = CToolbox::euler2quaternion(0,0,90*datum::pi/180);
				//myR = CToolbox::euler2rot3D(0,0,60*datum::pi/180);
			}else if (r==5){
				quaternion = CToolbox::euler2quaternion(0,0,120*datum::pi/180);
				//myR = CToolbox::euler2rot3D(0,0,75*datum::pi/180);
			}else if (r==6){
				quaternion = CToolbox::euler2quaternion(0,0,180*datum::pi/180);
				//myR = CToolbox::euler2rot3D(0,0,180*datum::pi/180);
			}
			*/
			
			myR = CToolbox::quaternion2rot3D(quaternion);
			
			// Form rotation matrix by solving Wahba's problem
		  	//myR = CToolbox::get_wahba(myR,origin);
			if (r == 0) {
				cout << "quaternion: " << quaternion << endl;
				cout << "rot3D: " << myR << endl;
			}
			CToolbox::merge3D(&myDP, &pix, &goodpix, &myR, pix_max, &myIntensity, &myWeight, active, interpolate);
  		}
  		// Normalize here
  		CToolbox::normalize(&myIntensity,&myWeight);
  		//myIntensity.save(output,raw_ascii);
  		fmat mySlice;
  		for (int i = 0; i < numSlices; i++) {
  			std::stringstream sstm;
  			sstm << output << setfill('0') << setw(6) << i << ".dat";
			string outputName = sstm.str();
			mySlice = myIntensity.slice(i).save(outputName,raw_ascii);
		}
    }

	//cout << "Total time: " <<timerMaster.toc()<<" seconds."<<endl;

  	return 0;

}

