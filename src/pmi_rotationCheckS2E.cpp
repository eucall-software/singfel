#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <armadillo>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>       /* time */
#include <math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include "detector.h"
#include "beam.h"
#include "particle.h"
#include "diffraction.h"
#include "toolbox.h"
#include "diffraction.cuh"

//#include <cuda.h>
#include <algorithm>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

int main( int argc, char* argv[] ){

    string beamFile;
    string geomFile;
    int gpu = 0; // not used
    string output;
    
	string inputDir;
	string outputDir;
	int USE_CUDA;
	
	int sliceInterval;
	int numSlices;
	int pmiStartID;
	int pmiEndID;
	int numDP;

    // Let's parse input
    for (int n = 1; n < argc; n++) {
    cout << argv [ n ] << endl;
        if(boost::algorithm::iequals(argv[ n ], "--sliceInterval")) {
            sliceInterval = atoi(argv[ n+2 ]);
        } else if (boost::algorithm::iequals(argv[ n ], "--input_dir")) {
            inputDir = argv[ n+2 ];
        } else if (boost::algorithm::iequals(argv[ n ], "--output_dir")) {
            outputDir = argv[ n+2 ];
        }else if (boost::algorithm::iequals(argv[ n ], "-b")) {
            beamFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-g")) {
            geomFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "--gpu")) {
            gpu = 1;
        } else if (boost::algorithm::iequals(argv[ n ], "--numSlices")) {
            numSlices = atoi(argv[ n+2 ]);
        } else if (boost::algorithm::iequals(argv[ n ], "--pmiStartID")) {
            pmiStartID = atoi(argv[ n+2 ]);
        } else if (boost::algorithm::iequals(argv[ n ], "--pmiEndID")) {
            pmiEndID = atoi(argv[ n+2 ]);
        } else if (boost::algorithm::iequals(argv[ n ], "--numDP")) {
            numDP = atoi(argv[ n+2 ]);
        }
    }

	string filename;
	string datasetname, datasetname1;

	int pmiID = pmiStartID;

	stringstream sstm;
	sstm << inputDir << "/pmi_out_" << setfill('0') << setw(7) << pmiID << ".h5";
	filename = sstm.str();
	
	int timeSlice = 1;
	stringstream sstm0;
	sstm0 << "/data/snp_" << setfill('0') << setw(7) << timeSlice;
	datasetname = sstm0.str();
			
	stringstream sstm1;
	sstm1 << "/data/angle";
	datasetname1 = sstm1.str();
		
	// Particle //
	CParticle particle = CParticle();
	//particle.load_atomType(filename,datasetname+"/T"); 	// rowvec atomType
	particle.load_atomPos(filename,datasetname+"/r");		// mat pos
	//particle.load_ionList(filename,datasetname+"/xyz");		// rowvec ion list
	particle.load_ffTable(filename,datasetname+"/ff");	// mat ffTable (atomType x qSample)
	particle.load_qSample(filename,datasetname+"/Q");	// rowvec q vector sin(theta)/lambda
	particle.load_particleOrientation(filename,datasetname1);		// frowvec quaternion

cout << "orientation: " << particle.get_particleOrientation() << endl;	
cout << "Done particle" << endl;
//cout << "Atom pos: " << particle.get_atomPos() << endl;

	// Rotate atom positions
	fmat myPos(3,3);
	myPos  << 1 << 0 << 0 << endr
  << 0 << 1 << 0 << endr
  << 0 << 0 << 1 << endr
  << 0.3 << 0.4 << 0.5 << endr; 
	//fmat myPos = particle.get_atomPos();
	fvec quat(4);
	quat << sqrt(0.5) << sqrt(0.5) << 0 << 0;
	fmat rot3D = CToolbox::quaternion2rot3D(quat);
	//fmat rot3D = CToolbox::quaternion2rot3D(particle.get_particleOrientation());
cout << "myPos: " << myPos << endl;
cout << "rot3D: " << rot3D << endl;	

cout << "Atom pos Trans: " << myPos * trans(rot3D) << endl; // right handed rotation
cout << "Atom pos: " << myPos * rot3D << endl; // left handed rotation
cout << "Atom pos: " << rot3D * trans(myPos) << endl; //
	return 0;
}			

