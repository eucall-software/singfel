/*
 * Merge diffraction patterns in a diffraction volume given known angles
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
    string eulerList;
    string quaternionList;
    string rotationList;
    string beamFile;
    string geomFile;
    int numImages = 0;
    int mySize = 0;
    string output;
    string format;
    // Let's parse input
    for (int n = 1; n < argc; n++) {
    cout << argv [ n ] << endl; 
        if(boost::algorithm::iequals(argv[ n ], "-i")) {
            imageList = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-e")) {
            eulerList = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-q")) {
		    quaternionList = argv[ n+1 ];
		} else if (boost::algorithm::iequals(argv[ n ], "-r")) {
		    rotationList = argv[ n+1 ];
		} else if (boost::algorithm::iequals(argv[ n ], "-b")) {
            beamFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-g")) {
            geomFile = argv[ n+1 ];   
        } else if (boost::algorithm::iequals(argv[ n ], "--num_images")) {
            numImages = atoi(argv[ n+2 ]);
        } else if (boost::algorithm::iequals(argv[ n ], "--vol_dim")) {
            mySize = atof(argv[ n+2 ]);
        } else if (boost::algorithm::iequals(argv[ n ], "--output_name")) {
            output = argv[ n+2 ];
        } else if (boost::algorithm::iequals(argv[ n ], "--format")) {
            format = argv[ n+2 ];
        }
    }
    //cout << numImages << endl;
    //cout << inc_res << endl;
    // image.lst and euler.lst are not neccessarily in the same order! So can not use like this. Perhaps use hdf5 to combine the two.

	/****** Beam ******/
	// Let's read in our beam file
	double photon_energy = 0;
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
            cout << line << endl; 
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
	fmat pix = det.pixSpace;
	float pix_max = det.pixSpaceMax;

    uvec goodpix;
    goodpix = det.get_goodPixelMap();

	double theta = atan((px/2*pix_height)/d);
	double qmax = 2/beam.get_wavelength()*sin(theta/2);
	double dmin = 1/(2*qmax);
	cout << "max q to the edge: " << qmax << " m^-1" << endl;
	cout << "Half period resolution: " << dmin << " m" << endl;

	if(!USE_CUDA) {

  		string filename;
  		
  		fmat myDP;
  		fmat myR;
  		myR.zeros(3,3);
  		float psi,theta,phi;

		fcube myWeight;
		myWeight.zeros(px,px,px);
		fcube myIntensity;
		myIntensity.zeros(px,px,px);
		
		cout << "Start" << endl;
		
		int active = 1;
		string interpolate = "linear";// "nearest";
		
		cout << format << endl;
  		for (int r = 0; r < numImages; r++) {
			if (format == "S2E") {
				// Get image from hdf
				stringstream sstm;
				sstm << imageList << "/diffr_out_" << setfill('0') << setw(7) << r+1 << ".h5";
				string inputName;
				inputName = sstm.str();
				// Read in diffraction				
				myDP = hdf5readT<fmat>(inputName,"/data/data");
				// Read angle
				fvec quat = hdf5readT<fvec>(inputName,"/data/angle");
				myR = CToolbox::quaternion2rot3D(quat);
				myR = trans(myR);
			} else {
				// Get image from dat file
		  		std::stringstream sstm;
	  			sstm << imageList << setfill('0') << setw(7) << r << ".dat";
				filename = sstm.str();
				myDP = load_asciiImage(filename);
				// Get badpixelmap from dat file
		  		std::stringstream sstm2;
	  			sstm2 << imageList << "BadPixels_" << setfill('0') << setw(7) << r << ".dat";
				badpixmap = sstm2.str();
				det.set_pixelMap(badpixmap);
				goodpix = det.get_goodPixelMap();
			    // Get rotation matrix from dat file
	  			std::stringstream sstm1;
				sstm1 << rotationList << setfill('0') << setw(7) << r << ".dat";
				string rotationName = sstm1.str();
				fvec euler = load_asciiEuler(rotationName);
				psi = euler(0);
				theta = euler(1);
				phi = euler(2);
				myR = CToolbox::euler2rot3D(psi,theta,phi); // WARNING: euler2rot3D changed sign 24/7/14
			}
        	CToolbox::merge3D(&myDP, &pix, &goodpix, &myR, pix_max, &myIntensity, &myWeight, active, interpolate);
  		}
  		// Normalize here
  		CToolbox::normalize(&myIntensity,&myWeight);
  		
  		// ########### Save diffraction volume ##############
  		cout << "Saving diffraction volume..." << endl;
		for (int i = 0; i < px; i++) {
			std::stringstream sstm;
			sstm << output << "vol_" << setfill('0') << setw(7) << i << ".dat";
			string outputName = sstm.str();
			myIntensity.slice(i).save(outputName,raw_ascii);
			// empty voxels
			std::stringstream sstm1;
			sstm1 << output << "volGoodVoxels_" << setfill('0') << setw(7) << i << ".dat";
			string outputName1 = sstm1.str();
			fmat goodVoxels = sign(myWeight.slice(i));
			goodVoxels.save(outputName1,raw_ascii);
		}
		
    }
  	return 0;
}

