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

#ifdef COMPILE_WITH_CXX11
	#define ARMA_DONT_USE_CXX11
#endif
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
	
	float tol = 1e-6;

	fvec quaternion(4);
	fmat myR;
	fvec euler;
	fvec newQuat;
	
	// Test no rotation
	cout << "*** Test 1 ***" << endl;
	quaternion << 1 << 0 << 0 << 0;
	myR = CToolbox::quaternion2rot3D(quaternion);
	euler = CToolbox::quaternion2euler(quaternion);
  	newQuat = CToolbox::euler2quaternion(euler(0),euler(1),euler(2));
  	
	quaternion.print("quaternion: ");
	myR.print("myR: ");
	euler.print("euler: ");
	newQuat.print("newQuaternion: ");
	
	if (abs(sum(quaternion - newQuat)) > tol ) {
		cout << "ERROR!!!" << endl;
		//exit(0);
	}

	// Test 180 degree rotation about x
	cout << "*** Test 2 ***" << endl;
	quaternion << 0 << 1 << 0 << 0;
	myR = CToolbox::quaternion2rot3D(quaternion);
	euler = CToolbox::quaternion2euler(quaternion);
  	newQuat = CToolbox::euler2quaternion(euler(0),euler(1),euler(2));
  	
	quaternion.print("quaternion: ");
	myR.print("myR: ");
	euler.print("euler: ");
	newQuat.print("newQuaternion: ");
	
	if (abs(sum(quaternion - newQuat)) > tol ) {
		cout << "ERROR!!!" << endl;
		//exit(0);
	}

	// Test 180 degree rotation about y
	cout << "*** Test 3 ***" << endl;
	quaternion << 0 << 0 << 1 << 0;
	myR = CToolbox::quaternion2rot3D(quaternion);
	euler = CToolbox::quaternion2euler(quaternion);
  	newQuat = CToolbox::euler2quaternion(euler(0),euler(1),euler(2));
  	
	quaternion.print("quaternion: ");
	myR.print("myR: ");
	euler.print("euler: ");
	newQuat.print("newQuaternion: ");
	
	if (abs(sum(quaternion - newQuat)) > tol ) {
		cout << "ERROR!!!" << endl;
		//exit(0);
	}

	// Test 180 degree rotation about z
	cout << "*** Test 4 ***" << endl;
	quaternion << 0 << 0 << 0 << 1;
	myR = CToolbox::quaternion2rot3D(quaternion);
	euler = CToolbox::quaternion2euler(quaternion);
  	newQuat = CToolbox::euler2quaternion(euler(0),euler(1),euler(2));
  	
	quaternion.print("quaternion: ");
	myR.print("myR: ");
	euler.print("euler: ");
	newQuat.print("newQuaternion: ");
	
	if (abs(sum(quaternion - newQuat)) > tol ) {
		cout << "ERROR!!!" << endl;
		//exit(0);
	}

	// Test 90 degree rotation about x
	cout << "*** Test 5 ***" << endl;
	quaternion << sqrt(0.5) << sqrt(0.5) << 0 << 0;
	myR = CToolbox::quaternion2rot3D(quaternion);
	euler = CToolbox::quaternion2euler(quaternion);
  	newQuat = CToolbox::euler2quaternion(euler(0),euler(1),euler(2));
  	
	quaternion.print("quaternion: ");
	myR.print("myR: ");
	euler.print("euler: ");
	newQuat.print("newQuaternion: ");
	
	if (abs(sum(quaternion - newQuat)) > tol ) {
		cout << "ERROR!!!" << endl;
		//exit(0);
	}

	// Test 90 degree rotation about y
	cout << "*** Test 6 ***" << endl;
	quaternion << sqrt(0.5) << 0 << sqrt(0.5) << 0;
	myR = CToolbox::quaternion2rot3D(quaternion);
	euler = CToolbox::quaternion2euler(quaternion);
  	newQuat = CToolbox::euler2quaternion(euler(0),euler(1),euler(2));
  	
	quaternion.print("quaternion: ");
	myR.print("myR: ");
	euler.print("euler: ");
	newQuat.print("newQuaternion: ");
	
	if (abs(sum(quaternion - newQuat)) > tol ) {
		cout << "ERROR!!!" << endl;
		//exit(0);
	}

	// Test 90 degree rotation about z
	cout << "*** Test 7 ***" << endl;
	quaternion << sqrt(0.5) << 0 << 0 << sqrt(0.5);
	myR = CToolbox::quaternion2rot3D(quaternion);
	euler = CToolbox::quaternion2euler(quaternion);
  	newQuat = CToolbox::euler2quaternion(euler(0),euler(1),euler(2));
  	
	quaternion.print("quaternion: ");
	myR.print("myR: ");
	euler.print("euler: ");
	newQuat.print("newQuaternion: ");
	
	if (abs(sum(quaternion - newQuat)) > tol ) {
		cout << "ERROR!!!" << endl;
		//exit(0);
	}

	// Test random quaternion
	cout << "*** Test 8 ***" << endl;
	srand (time(NULL));
	// Get rotation matrix
  	fvec u = randu<fvec>(3); // uniform random distribution in the [0,1] interval
	// generate uniform random quaternion on SO(3)
	quaternion << sqrt(1-u(0)) * sin(2*datum::pi*u(1)) << sqrt(1-u(0)) * cos(2*datum::pi*u(1))
				<< sqrt(u(0)) * sin(2*datum::pi*u(2)) << sqrt(u(0)) * cos(2*datum::pi*u(2));
	if (quaternion(0) < 0) {
		quaternion *= -1;
	}
	myR = CToolbox::quaternion2rot3D(quaternion);
	euler = CToolbox::quaternion2euler(quaternion);
  	newQuat = CToolbox::euler2quaternion(euler(0),euler(1),euler(2));
  	
	quaternion.print("quaternion: ");
	myR.print("myR: ");
	euler.print("euler: ");
	newQuat.print("newQuaternion: ");
	
	if (abs(sum(quaternion - newQuat)) > tol ) {
		cout << "ERROR!!!" << endl;
		//exit(0);
	}
	
	cout << "Successful!" << endl;

  	return 0;
}

