/*
 * Program for phasing 3D diffraction volume
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

using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

int main( int argc, char* argv[] ){

	wall_clock timerMaster;

	string diffVol;
	int mySize = 0;
	int startIter = 0;
	int numIterations = 0;
	
	// Perhaps use hdf5 to combine the two.
	for (int n = 1; n < argc; n++) {
		cout << argv [ n ] << endl; 
	    if (boost::algorithm::iequals(argv[ n ], "--vol_dim")) {
		    mySize = atof(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--num_iterations")) {
		    numIterations = atoi(argv[ n+2 ]);
		} else if (boost::algorithm::iequals(argv[ n ], "--diffraction_volume")) {
	        diffVol = argv[ n+2 ];
	    }
	}
	
	fcube myIntensity;
	myIntensity.zeros(mySize,mySize,mySize);
	
	for (int i = 0; i < mySize; i++) {
		// Get a layer of volume
		std::stringstream sstm;
		sstm << diffVol << setfill('0') << setw(7) << i << ".dat";
		string outputName = sstm.str();
		myIntensity.slice(i).load(outputName,raw_ascii);
	}
			
	return 0;			
}
