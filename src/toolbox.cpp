#include <iostream>
#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cfloat>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "toolbox.h"

#include <string>
#include <sstream>
#include <armadillo>

using namespace std;
using namespace arma;
using namespace toolbox;

// Calculate magnitude of x,y,z
mat CToolbox::mag(cube x){
	x = pow(x,2);
	mat y = sqrt(x.slice(0)+x.slice(1)+x.slice(2));
	return y;
}

fmat CToolbox::mag(fcube x){
	x = pow(x,2);
	fmat y = sqrt(x.slice(0)+x.slice(1)+x.slice(2));
	return y;
}

umat CToolbox::convert_to_poisson(fmat x){
	unsigned long randSeed;
	gsl_rng * gBaseRand;
	const gsl_rng_type * T;
	T = gsl_rng_default;
	gBaseRand = gsl_rng_alloc (T); 
  	randSeed = rand();                    /* returns a non-negative integer */
//  	cout << "time: " << time(NULL) << endl;
//cout << "randSeed: " << randSeed << endl;
  	gsl_rng_set (gBaseRand, randSeed);    /* seed the PRNG */
//cout << "set done" << endl;	            
	umat y;
//cout << "init y" << endl;		
	y.copy_size(x);
//cout << "copy done" << endl;		
	for (unsigned i = 0; i < y.n_elem; i++) {
		//cout << "i: " << i << endl;
		//cout << "x(i): " << x(i) << endl;
		if (x(i) <= FLT_MAX) {
			y(i) = gsl_ran_poisson (gBaseRand, x(i));
		} else {
			cerr << "detector intensity too large for poisson" << endl;
			y(i) = 0;
		}
		//cout << "y: " << y(i) << endl;	
	}	
//cout << "x: " << x(0) << endl;	
	gsl_rng_free(gBaseRand);
	
	return y;
}

fmat CToolbox::quaternion2rot3D(vec quaternion, int convention){
	fmat rot3D;
	float theta = 0.0;
	vec axis(3);
	quaternion2AngleAxis(quaternion,theta,axis);
	//cout << "theta: " << theta << endl;
	//cout << "axis: " << axis << endl;
	return rot3D = angleAxis2rot3D(axis,theta);
}

void CToolbox::quaternion2AngleAxis(vec quaternion,float& theta,vec& axis){
	float HA = acos(quaternion(0));
	theta = 2 * HA;
	if (theta < datum::eps){ // eps ~= 2.22045e-16
		theta = 0.0;
		axis << 1 << 0 << 0;
	} else {	
		axis = quaternion.subvec(1,3)/sin(HA);
	}
}

fmat CToolbox::angleAxis2rot3D(vec axis, float theta){
	if (axis.n_elem != 3) {
		cout << "Number of axis element must be 3" << endl;
		exit(EXIT_FAILURE);
	}
	axis = axis/norm(axis,2); // L2 norm
	double a = axis(0);
	double b = axis(1);
	double c = axis(2);
	double cosTheta = cos(theta);
	double bracket = 1-cosTheta;
	double aBracket = a * bracket;
	double bBracket = b * bracket;
	double cBracket = c * bracket;
	double sinTheta = sin(theta);
	double aSinTheta = a * sinTheta;
	double bSinTheta = b * sinTheta;
	double cSinTheta = c * sinTheta;
	fmat rot3D(3,3);
	rot3D << a*aBracket+cosTheta << a*bBracket-cSinTheta << a*cBracket+bSinTheta << endr
		  << b*aBracket+cSinTheta << b*bBracket+cosTheta << b*cBracket-aSinTheta << endr
		  << c*aBracket-bSinTheta << c*bBracket+aSinTheta << c*cBracket+cosTheta << endr;
	return rot3D;
}

#ifdef COMPILE_WITH_BOOST
	#include <boost/python.hpp>
	using namespace boost::python;
	using namespace toolbox;
	using namespace arma;

	//BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(CToolbox_overloads, mag, 1, 1)

	BOOST_PYTHON_MODULE(toolbox)
	{
		//static mat (CToolbox::*mag1)(cube) = &CToolbox::mag;
		//static fmat (CToolbox::*mag2)(fcube) = &CToolbox::mag;

		class_<CToolbox>("CToolbox", init<std::string>())	// constructor with string input
			//.def(init<>())	// Overloaded constructor
			.def_readwrite("name", &CToolbox::name)
			//.add_property("number", &SomeClass::getNumber, &SomeClass::setNumber)
			//.def("test", &SomeClass::test)
			//.def("mag", mag1)
			//.def("mag", mag2)
			.def("convert_to_poisson", &CToolbox::convert_to_poisson)
		;
	}
#endif
