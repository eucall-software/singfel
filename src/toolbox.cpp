#include <iostream>
#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cfloat>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "toolbox.h"
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
cout << "randSeed: " << randSeed << endl;
  	gsl_rng_set (gBaseRand, randSeed);    /* seed the PRNG */
cout << "set done" << endl;	            
	umat y;
cout << "init y" << endl;		
	y.copy_size(x);
cout << "copy done" << endl;		
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
cout << "x: " << x(0) << endl;	
	gsl_rng_free(gBaseRand);
	
	return y;
}

/*
float CToolbox::Uniform(float mean)
{
	//Generate a random number between 0 and 1.
	//REMEMBER: Always cast the oparands of a division to float, or truncation will be performed.
	float R;
	R = (float)rand()/(float)(RAND_MAX+1);
           
	return  2*mean*R;
}

int CToolbox::Poisson(float mean) //Special technique required: Box-Muller method...
{
	float R;
	float sum = 0;
	int i = -1;
	float z;
 
	while(sum <=mean) {
		R = (float)rand()/(float)(RAND_MAX+1);
		z = -log(R);
		sum += z;
		i++;
	}
	
	return i;
}
*/

