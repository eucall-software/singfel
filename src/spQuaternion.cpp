#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include "detector.h"
#include "beam.h"
#include "particle.h"
#include "diffraction.h"
#include "toolbox.h"
#include "diffraction.cuh"

#include <armadillo>

//#include <cuda.h>
#include <algorithm>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

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

    wall_clock timer;
    
    string atomTypeFile;
    string posFile;
    string xyzIndFile;
    string ffTableFile;
    string qSampleFile;    
    string beamFile;
    string geomFile;
    int numImages = 0;
    string output;
    // Let's parse input
    for (int n = 1; n < argc; n++) {
    cout << argv [ n ] << endl; 
        if(boost::algorithm::iequals(argv[ n ], "-a")) {
            atomTypeFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-p")) {
            posFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-x")) {
            xyzIndFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-f")) {
            ffTableFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-q")) {
            qSampleFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-b")) {
            beamFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-g")) {
            geomFile = argv[ n+1 ];   
        } else if (boost::algorithm::iequals(argv[ n ], "--num_images")) {
            numImages = atoi(argv[ n+2 ]);
        } else if (boost::algorithm::iequals(argv[ n ], "--output_dir")) {
            output = argv[ n+2 ];
        }
    }
	
	// Particle
	CParticle particle = CParticle();
	particle.load_atomType(atomTypeFile.c_str()); 	// rowvec atomType 
	particle.load_atomPos(posFile.c_str());		// mat pos
	particle.load_xyzInd(xyzIndFile.c_str());		// rowvec xyzInd (temporary)
	particle.load_ffTable(ffTableFile.c_str());	// mat ffTable (atomType x qSample)
	particle.load_qSample(qSampleFile.c_str());	// rowvec q vector sin(theta)/lambda
	

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
	beam.set_focus(focus_radius*2); // radius to diameter
	beam.set_photonsPerPulse(fluence);
	beam.set_photonsPerPulsePerArea();

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
	det.set_pixelMap(badpixmap);
	det.set_center_x(cx);
	det.set_center_y(cy);

	det.init_dp(&beam);

	CDiffraction::calculate_atomicFactor(&particle,&det); // get f_hkl

	double theta = atan((px/2*pix_height)/d);
	double qmax = 2/beam.get_wavelength()*sin(theta/2);
	double dmin = 1/(2*qmax);
	cout << "max q to the edge: " << qmax << " m^-1" << endl;
	cout << "Half period resolution: " << dmin << " m" << endl;

	if(!USE_CUDA) {
		
		CDiffraction::get_atomicFormFactorList(&particle,&det);
		
		fmat F_hkl_sq;
		string outputName;

		fmat myPos = particle.get_atomPos();
		
		CParticle rotatedParticle = CParticle();	
		rotatedParticle.load_atomType(atomTypeFile.c_str()); 	// rowvec atomType 
		rotatedParticle.load_atomPos(posFile.c_str());		// mat pos
		rotatedParticle.load_xyzInd(xyzIndFile.c_str());		// rowvec xyzInd (temporary)
		rotatedParticle.load_ffTable(ffTableFile.c_str());	// mat ffTable (atomType x qSample)
		rotatedParticle.load_qSample(qSampleFile.c_str());	// rowvec q vector sin(theta)/lambda
		
		fmat rot3D(3,3);
		fvec u(3);
		fvec quaternion(4);	
		for (int i = 0; i < numImages; i++) {
			timer.tic();
			// Rotate single particle			
			u = randu<fvec>(3); // uniform random distribution in the [0,1] interval
			quaternion << sqrt(1-u(0)) * sin(2*datum::pi*u(1)) << sqrt(1-u(0)) * cos(2*datum::pi*u(1))
					   << sqrt(u(0)) * sin(2*datum::pi*u(2)) << sqrt(u(0)) * cos(2*datum::pi*u(2));

			rot3D = CToolbox::quaternion2rot3D(quaternion);

			fmat myRotPos = myPos * rot3D;
	
			rotatedParticle.set_atomPos(&myRotPos);
		
			F_hkl_sq = CDiffraction::calculate_intensity(&rotatedParticle,&det);
			
			fmat detector_intensity = F_hkl_sq % det.solidAngle % det.thomson * beam.get_photonsPerPulsePerArea();
			umat detector_counts = CToolbox::convert_to_poisson(detector_intensity);	

			stringstream sstm2;
			sstm2 << output << "detector_intensity_" << setfill('0') << setw(6) << i << ".dat";
			outputName = sstm2.str();
			detector_intensity.save(outputName,raw_ascii);
			stringstream sstm;
			sstm << output << "diffraction_" << setfill('0') << setw(6) << i << ".dat";
			outputName = sstm.str();
			detector_counts.save(outputName,raw_ascii);
			stringstream sstm1;
			sstm1 << output << "quaternion_" << setfill('0') << setw(6) << i << ".dat";
			outputName = sstm1.str();
			quaternion.save(outputName,raw_ascii);			
			
			cout<<"Save image: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
		}
	}

  	return 0;
}

