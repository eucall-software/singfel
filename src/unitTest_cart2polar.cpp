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
#include "io.h"

#include <armadillo>

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

#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
#include <string>

#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

//#define ARMA_NO_DEBUG

#define USE_CUDA 0
#define USE_CHUNK 0

int main( int argc, char* argv[] ){

	// Get image
	std::stringstream sstm;
  	sstm << "/data/yoon/singfel/cartEven.dat";
	string filename = sstm.str();
	
	fmat myDP = load_asciiImage(filename);
	myDP.print("myDP: ");
	
	float rhoMin = 0;
	float rhoMax = (myDP.n_rows - 1)/2.;
	cout << "rhoMax: " << rhoMax << endl;
	cout << endl;
	float circumference = rhoMax * (2 * datum::pi); // pixels
	int numRotSamples = ceil(circumference);
	int numRadSamples = floor(rhoMax - rhoMin)+1;
	fcube samplePoints = zeros<fcube>(numRotSamples,numRadSamples,2);
	CToolbox::cart2polar(&samplePoints, myDP.n_cols, rhoMin, rhoMax);
	samplePoints.print("sample points: ");
	fmat myPolarDP = zeros<fmat>(numRotSamples,numRadSamples);
	CToolbox::interp_linear2D(&myPolarDP, &samplePoints, &myDP);
	
	myPolarDP.print("polar dp: ");
	
	fmat shiftDP = zeros<fmat>(numRotSamples,numRadSamples);
	shiftDP.rows(0,3) = myPolarDP.rows(numRotSamples-4,numRotSamples-1);
	cout << "done" << endl;
	shiftDP.rows(4,numRotSamples-1) = myPolarDP.rows(0,numRotSamples-5);
	
	shiftDP.print("shift dp: ");
	
	cx_fmat A = conj(fft2(myPolarDP));
	cx_fmat B = fft2(shiftDP);
	fmat CC = real(fft2(A % B));
	CC.print("CC: ");
	int bestRot = 0;
	int maxVal = 0;
	for (int i = 0; i < numRotSamples; i++) {
		if (CC(i,0) > maxVal) {
			bestRot = i;
			maxVal = CC(i,0);
		}
	}
	cout << "bestRot: " << bestRot << endl;
	cout << "Successful!" << endl;

  	return 0;
}

