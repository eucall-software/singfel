#ifndef SINGFEL_TOOLBOX_H
#define SINGFEL_TOOLBOX_H

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
#include <math.h>

//#include <boost/math/special_functions/factorials.hpp>

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
  	gsl_rng_set (gBaseRand, randSeed);    /* seed the PRNG */            
	umat y;	
	y.copy_size(x);		
	for (unsigned i = 0; i < y.n_elem; i++) {
		if (x(i) <= FLT_MAX) {
			y(i) = gsl_ran_poisson (gBaseRand, x(i));
		} else {
			cerr << "detector intensity too large for poisson" << endl;
			y(i) = 0; // This should be Gaussian for large intensity
		}	
	}		
	gsl_rng_free(gBaseRand);
	
	return y;
}

// Right hand rotation theta about an axis
void CToolbox::quaternion2AngleAxis(fvec quaternion, float& theta, fvec& axis){
	if (quaternion.n_rows == 4 && quaternion.n_cols == 1) {
		// do nothing
	} else if (quaternion.n_rows == 1 && quaternion.n_cols == 4){
		// convert to row vector
		quaternion = quaternion.t();
	} else {
		cout << "Not a quaternion. Exiting..." << endl;
		exit(0);
	}
	float HA = acos(quaternion(0));
	theta = 2 * HA;
	if (theta < datum::eps){ // eps ~= 2.22045e-16
		theta = 0.0;
		axis << 1 << 0 << 0;
	} else {	
		axis = quaternion.rows(1,3)/sin(HA);
	}
}

// Let's use zyz convention after Heymann (2005)
fmat CToolbox::quaternion2rot3D(fvec q){ // it doesn't accept frowvec
	fmat rot3D;     
    float theta = 0.0;
	fvec axis(3);
	quaternion2AngleAxis(q,theta,axis);
	//cout << "theta: " << theta << endl;
	//cout << "axis: " << axis << endl;
	//cout << "angleAxisRot: " << angleAxis2rot3D(axis,theta) << endl;
	rot3D = angleAxis2rot3D(axis,theta);
	return rot3D;
}

fmat CToolbox::angleAxis2rot3D(fvec axis, float theta){
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

/* NOT USED */
// Let's use zyz convention after Heymann (2005)
fvec CToolbox::quaternion2euler(fvec q) {
	// input: quaternion
	// output: euler(psi,theta,phi)
	fvec euler(3);
	// first rotation about phi, then theta, then psi
	float psi, theta, phi;
	fmat myR = quaternion2rot3D(q);
	cout << "quaternion2euler::myR: " << myR << endl;
	theta = acos(myR(2,2));
	cout << "quaternion2euler::theta: " << theta << endl;
	if (theta == 0 || theta == datum::pi) {
		cout << "quaternion2euler::here" << endl;
		phi = 0;
		psi = atan2(-myR(1,0),myR(0,0));
	} else {
		phi = atan2(myR(2,1),myR(2,0));
		psi = atan2(myR(1,2),-myR(0,2));
	}

	euler(0) = psi;
	euler(1) = theta;
	euler(2) = phi;
	return euler;
}

// zyz, euler in radians
fvec CToolbox::euler2quaternion(float psi, float theta, float phi) {
	
	fvec quaternion(4);
	
	if (abs(psi) < datum::eps && abs(theta) < datum::eps && abs(phi) < datum::eps ) {
		quaternion << 1 << 0 << 0 << 0;
	} else { 
		fmat R(3,3);
		//cout << "theta:" << theta << endl;
		//quaternion << sin(theta/2)*sin((phi-psi)/2) << sin(theta/2)*cos((phi-psi)/2) << cos(theta/2)*sin((phi+psi)/2) << cos(theta/2)*cos((phi+psi)/2);
		//cout << "quaternion: " << quaternion << endl;
	
		R = euler2rot3D(psi, theta, phi);
	
		fvec VV(3);
		VV << R(1,2)-R(2,1) << R(2,0)-R(0,2) << R(0,1)-R(1,0);
		if (VV(0) == 0) {
		    VV = VV/norm(VV,2); // Added by Chuck
		} else if (VV(0) > 0) {
		    VV = VV/norm(VV,2);
		} else if(VV(0) < 0) {
			VV = VV/norm(VV,2)*-1;
		}    
		theta = acos(0.5*(trace(R)-1.0));

		float CCisTheta = corrCoeff(R,angleAxis2rot3D(VV,theta));
		float CCisNegTheta = corrCoeff(R,angleAxis2rot3D(VV,-theta));
		
		//cout << "CCisTheta: " << CCisTheta << endl;
		
		if (CCisNegTheta > CCisTheta) {
		    theta = -theta;
		}
		quaternion << cos(theta/2) << sin(theta/2)*VV(0)/norm(VV) << sin(theta/2)*VV(1)/norm(VV) << sin(theta/2)*VV(2)/norm(VV);
	}
	
	if (quaternion(0) < 0) {
		quaternion *= -1;
	}
	
	return quaternion;
}

float CToolbox::corrCoeff(fmat X, fmat Y) {
	float cc = 0;
	fvec x = vectorise(X);
	fvec y = vectorise(Y);
	
	float meanX = mean(x);
	float meanY = mean(y);
	
	x = x - meanX;
	y = y - meanY;
	
	cc = (dot(x,y))/sqrt(dot(x,x)*dot(y,y));
	
	return cc;
}

// Let's use zyz convention after Heymann (2005)
fmat CToolbox::euler2rot3D(float psi, float theta, float phi) {
    fmat Rphi(3,3);
    fmat Rtheta(3,3);
    fmat Rpsi(3,3);
    Rphi << cos(phi) << sin(phi) << 0 << endr
         << -sin(phi) << cos(phi) << 0 << endr
         << 0 << 0 << 1 << endr;
    Rtheta << cos(theta) << 0 << -sin(theta) << endr
           << 0 << 1 << 0 << endr
           << sin(theta) << 0 << cos(theta) << endr;
    Rpsi << cos(psi) << sin(psi) << 0 << endr
         << -sin(psi) << cos(psi) << 0 << endr
         << 0 << 0 << 1 << endr;
    fmat rot3D(3,3);
    return rot3D = Rpsi * Rtheta * Rphi;
}

/* NOT USED */
// Let's use zyz convention after Heymann (2005)
fvec CToolbox::rot3D2euler(fmat rot3D) {
    fvec euler(3);
    float theta, phi, psi;
    theta = acos(rot3D(2,2)); // theta
    if (theta != 0 || theta != 3.14159) {
        phi = atan2(rot3D(2,1),rot3D(2,0));
        psi = atan2(rot3D(1,2),rot3D(0,2));
    } else {
        phi = 0;
        psi = atan2(rot3D(1,0),rot3D(0,0));
    }
    euler(0) = psi;
    euler(1) = theta;
    euler(2) = phi;
    //cout << "euler:"<< euler << endl;
    return euler;
}

fmat CToolbox::get_wahba(fmat currentRot,fmat originRot) {
	fmat myR;
	mat B;
  	mat U;
	vec s;
	mat V;
	mat M;
	fmat temp;
	temp.zeros(3,3);
	// Form rotation matrix by solving Wahba's problem
	for (int i = 0; i < 3; i++) {
		temp += trans(originRot.row(i)) * currentRot.row(i); // <-- what is this black magic? 
	}
	B = conv_to<mat>::from(temp);				
	svd(U,s,V,B);
	M.eye(3,3);
	M(2,2) = arma::det(U)*arma::det(trans(V));
			
	myR = conv_to<fmat>::from(U*M*trans(V));
	return myR;
}

// Given number of points, distribute evenly on hyper surface of a 3-sphere
fmat CToolbox::pointsOn3Sphere(int numPts) {
	fmat points;
	points.zeros(2*numPts,3);
	const int N = 3;	
	float surfaceArea = N * pow(2,N) * pow(datum::pi,(N-1)/2) / (3*2); // for odd N
	float delta = exp(log(surfaceArea/numPts)/2);
	int iter = 0;
	int ind = 0;
	int maxIter = 1000;
	float deltaW1,deltaW2;
	float q0, q1, q2;
	float w1, w2;
	frowvec q(3); 
	while (ind != numPts && iter < maxIter) {
		ind = 0;
		deltaW1 = delta;
		for (w1 = 0.5*deltaW1; w1 < datum::pi; w1+=deltaW1) {
			q0 = cos(w1);
			deltaW2 = deltaW1/sin(w1);
			for (w2 = 0.5*deltaW2; w2 < 2*datum::pi; w2+=deltaW2) {
				q1 = sin(w1) * cos(w2);
				q2 = sin(w1) * sin(w2);
				q << q0 << q1 << q2 << endr;
				points.row(ind)= q;
				ind += 1;
			}
		}
		delta *= exp(log((float)ind/numPts)/2);
		iter += 1;
	}
	return points.rows(0, numPts-1);
}

// Given number of points, distribute evenly on hyper surface of a 4-sphere
fmat CToolbox::pointsOn4Sphere(int numPts) {
	fmat quaternion;
	quaternion.zeros(2*numPts,4);
	const int N = 4;	
	float surfaceArea = N * pow(datum::pi,N/2) /  (N/2); // for even N
	float delta = exp(log(surfaceArea/numPts)/3);
	int iter = 0;
	int ind = 0;
	int maxIter = 1000;
	float deltaW1,deltaW2,deltaW3;
	float q0, q1, q2, q3;
	float w1, w2, w3;
	frowvec q(4); 
	while (ind != numPts && iter < maxIter) {
		ind = 0;
		deltaW1 = delta;
		for (w1 = 0.5*deltaW1; w1 < datum::pi; w1+=deltaW1) {
			q0 = cos(w1);
			deltaW2 = deltaW1/sin(w1);
			for (w2 = 0.5*deltaW2; w2 < datum::pi; w2+=deltaW2) {
				q1 = sin(w1) * cos(w2);
				deltaW3 = deltaW2/sin(w2);
				for (w3 = 0.5*deltaW3; w3 < 2*datum::pi; w3+=deltaW3) {
					q2 = sin(w1)*sin(w2)*cos(w3);
					q3 = sin(w1)*sin(w2)*sin(w3);
					q << q0 << q1 << q2 << q3 << endr;
					quaternion.row(ind)= q;
					ind += 1;
				}
			}
		}
		delta *= exp(log((float)ind/numPts)/3);
		iter += 1;
	}
	return quaternion.rows(0, numPts-1);
}

// Calculate a set of sampling points on a polar grid given a cartesian grid
void CToolbox::cart2polar(fcube* samplePoints, int detectorWidth, float rhoMin, float rhoMax){
	// samplePoints: polar grid positions (number of rotational samples x number of radial samples x 2)
	// rhoMin: starting radial value in pixels
	// rhoMax: last radial value in pixels
	int numRotSamples = samplePoints->n_rows;
	//cout << "numRotSamples: " << numRotSamples << endl;
	float deltaTheta = (2 * datum::pi) / numRotSamples; // radians
	fvec rotPositions(numRotSamples);
	for (int i = 0; i < numRotSamples; i++) {
		rotPositions(i) = i*deltaTheta;
	}

	int numRadSamples = samplePoints->n_cols;
	//cout << "numRadSamples: " << numRadSamples << endl;
	fvec radPositions(numRadSamples);
	float deltaRad = floor(rhoMax - rhoMin + 1) / numRadSamples;
	
	for (int j = 0; j < numRadSamples; j++) {
		radPositions(j) = j*deltaRad + rhoMin; 
	}
	
	// Origin at centre of matrix
	for (int i = 0; i < numRotSamples; i++) {
		for (int j = 0; j < numRadSamples; j++) {
			samplePoints->at(i,j,0) = radPositions(j) * cos(rotPositions(i)); // x position
			samplePoints->at(i,j,1) = radPositions(j) * sin(rotPositions(i)); // y position
		}
	}
	// Shift origin to top left corner of matrix
	for (int i = 0; i < numRotSamples; i++) {
		for (int j = 0; j < numRadSamples; j++) {
			samplePoints->at(i,j,0) += (detectorWidth-1)/2.; // x position
			samplePoints->at(i,j,1) += (detectorWidth-1)/2.; // y position
		}
	}
	
}

// Interpolate detector intensities onto a set of sampling points
void CToolbox::interp_linear2D(fmat* newDP, fcube* samplePoints, fmat* cartDP){
	// newDP: new interpolated diffraction pattern
	// samplePoints: new sampling positions
	// cartDP: diffraction pattern on a cartesian grid
	int numRotSamples = samplePoints->n_rows;
	int numRadSamples = samplePoints->n_cols;
	
	int xl, xu, yl, yu;
	float fx, fy, cx, cy;

	for (int i = 0; i < numRotSamples; i++) {
		for (int j = 0; j < numRadSamples; j++) {
		//	cout << "i,j: " << i << "," << j << endl;
			xl = floor(samplePoints->at(i,j,0));
			xu = xl + 1;
			yl = floor(samplePoints->at(i,j,1));
			yu = yl + 1;
			fx = samplePoints->at(i,j,0) - xl;
			fy = samplePoints->at(i,j,1) - yl;
			cx = xu - samplePoints->at(i,j,0);
			cy = yu - samplePoints->at(i,j,1);
		/*
		//if (i == 4 && j == 3) {
				cout << samplePoints->at(i,j,0) << "," << samplePoints->at(i,j,1) << endl;
		
				cout << xl << endl;
				cout << xu << endl;
				cout << yl << endl;
				cout << yu << endl;
				
				cout << fx << endl;
				cout << fy << endl;
				cout << cx << endl;
				cout << cy << endl;
				
				cout << cartDP->at(yl,xl) << endl;
				cout << cartDP->at(yl,xu) << endl;
				cout << cartDP->at(yu,xl) << endl;
				cout << cartDP->at(yu,xu) << endl;
		//}
		*/
			newDP->at(i,j) = cartDP->at(yl,xl)*cx*cy + cartDP->at(yl,xu)*fx*cy + cartDP->at(yu,xl)*cx*fy + cartDP->at(yu,xu)*fx*fy;
	
		}
	}
	
	
	
}

// Take an Ewald's slice from a diffraction volume
// Rename: extract_slice
void CToolbox::extract_interp_linear3D(fmat *myValue, fmat *myPoints, uvec *pixmap, fcube *myIntensity1) {
    int mySize = myIntensity1->n_rows;
    
    fmat& mySlice = myValue[0];
    
    //mySlice.print("mySlice: ");
    
    fmat& pixRot = myPoints[0];
    
    imat xyz;
	xyz = conv_to<imat>::from(floor(pixRot));
    
    fmat fxyz = pixRot - xyz;
    fmat cxyz = 1. - fxyz;
    float x,y,z,fx,fy,fz,cx,cy,cz;
	fcube& myIntensity = myIntensity1[0];
	
	uvec& goodpixmap = pixmap[0];
	uvec::iterator a = goodpixmap.begin();
    uvec::iterator b = goodpixmap.end();
    for(uvec::iterator p=a; p!=b; ++p) {

		x = xyz(1,*p);
		y = xyz(0,*p);
		z = xyz(2,*p);

		if (x >= mySize-1 || y >= mySize-1 || z >= mySize-1)
		    continue;
					
		if (x < 0 || y < 0 || z < 0)
			continue;	
				
		fx = fxyz(1,*p);
		fy = fxyz(0,*p);
		fz = fxyz(2,*p);
		cx = cxyz(1,*p);
		cy = cxyz(0,*p);
		cz = cxyz(2,*p);

		mySlice(*p) = cx*(cy*(cz*myIntensity(y,x,z) + fz*myIntensity(y,x,z+1)) + fy*(cz*myIntensity(y+1,x,z) + fz*myIntensity(y+1,x,z+1))) + fx*(cy*(cz*myIntensity(y,x+1,z) + fz*myIntensity(y,x+1,z+1)) + fy*(cz*myIntensity(y+1,x+1,z) + fz*myIntensity(y+1,x+1,z+1)));
  
	}
}

// Insert a Ewald's slice into a diffraction volume
//void CToolbox::interp_linear3D(fmat *myValue, fmat *myPoints, imat *myGridPoints, fcube *myIntensity1, fcube *myWeight1) {
// Rename: insert_slice
void CToolbox::interp_linear3D(fmat *myValue, fmat *myPoints, uvec *pixmap, fcube *myIntensity1, fcube *myWeight1) {
    int mySize = myIntensity1->n_rows;
    
    //cout << "mySize: " << endl;
    //cout << xyz[0] << endl;
    fmat& pixRot = myPoints[0];
    
    //imat& xyz = myGridPoints[0];
    imat xyz;
	xyz = conv_to<imat>::from(floor(pixRot));
    
    fmat fxyz = pixRot - xyz;
    fmat cxyz = 1. - fxyz;
    float x,y,z,fx,fy,fz,cx,cy,cz;
    float weight;
	float photons;
	fmat& myPhotons = myValue[0];
	//fmat& myImg = myValue[0];
	//fmat myPhotons = myImg.dp;
	fcube& myIntensity = myIntensity1[0];
	fcube& myWeight = myWeight1[0];
	
	uvec& goodpixmap = pixmap[0];
	uvec::iterator a = goodpixmap.begin();
    uvec::iterator b = goodpixmap.end();
    for(uvec::iterator p=a; p!=b; ++p) {
        //cout << p << endl;
        photons = myPhotons(*p);

		x = xyz(1,*p);
		y = xyz(0,*p);
		z = xyz(2,*p);

		if (x >= mySize-1 || y >= mySize-1 || z >= mySize-1)
		    continue;
					
		if (x < 0 || y < 0 || z < 0)
			continue;	
				
		fx = fxyz(1,*p);
		fy = fxyz(0,*p);
		fz = fxyz(2,*p);
		cx = cxyz(1,*p);
		cy = cxyz(0,*p);
		cz = cxyz(2,*p);
				
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

// Insert a Ewald's slice into a diffraction volume
void CToolbox::interp_nearestNeighbor(fmat *myValue, fmat *myPoints, uvec *pixmap, fcube *myIntensity1, fcube *myWeight1) {
    int mySize = myIntensity1->n_rows;
    //cout << xyz[0] << endl;
    fmat& pixRot = myPoints[0];
    
    //imat& xyz = myGridPoints[0];
    imat xyz;
	xyz = conv_to<imat>::from(round(pixRot));
    
    //fmat fxyz = pixRot - xyz;
    //fmat cxyz = 1. - fxyz;
    float x,y,z;
    float weight;
	float photons;
	fmat& myPhotons = myValue[0];
	//fmat& myImg = myValue[0];
	//fmat myPhotons = myImg.dp;
	fcube& myIntensity = myIntensity1[0];
	fcube& myWeight = myWeight1[0];
	
	uvec& goodpixmap = pixmap[0];
	uvec::iterator a = goodpixmap.begin();
    uvec::iterator b = goodpixmap.end();
    for(uvec::iterator p=a; p!=b; ++p) {
        //cout << *p << endl;
        photons = myPhotons(*p);
        
        x = xyz(1,*p);
		y = xyz(0,*p);
		z = xyz(2,*p);

		if (x >= mySize-1 || y >= mySize-1 || z >= mySize-1)
		    continue;
					
		if (x < 0 || y < 0 || z < 0)
			continue;	
				
		weight = 1;
		myWeight(y,x,z) += weight;
		myIntensity(y,x,z) += photons;
    }   
}

// Normalize the diffraction volume
void CToolbox::normalize(fcube *myIntensity1, fcube *myWeight1) {
    fcube& myIntensity = myIntensity1[0];
	fcube& myWeight = myWeight1[0];
	uvec ind = find(myWeight > 0);
	//cout << "ind: " << ind << endl;
	myIntensity.elem(ind) = myIntensity.elem(ind) / myWeight.elem(ind); // Use sparse indexing
}

// Insert a Ewald's slice into a diffraction volume
// active = 1: active rotation
// interpolate = 1: trilinear
void CToolbox::merge3D(fmat *myValue, fmat *myPoints, uvec *goodpix, fmat *myRot, float pix_max, fcube *myIntensity, fcube *myWeight, int active, string interpolate ) {
    fmat& pix = myPoints[0];
    //cout << "pix: " << pix(0,0) << " " << pix(0,1) << " " << pix(0,2) << endl;
    fmat& myR = myRot[0];
    //myR.print("myR: ");
    fmat pixRot;
	pixRot.zeros(pix.n_elem,3);
	if (active == 1) {
        pixRot = pix*conv_to<fmat>::from(trans(myR)) + pix_max; // this is active rotation
        pixRot = trans(pixRot);
        //cout << "pixRot: " << pixRot(0,0)- pix_max << " " << pixRot(1.0)- pix_max << " " << pixRot(2,0)- pix_max << endl;
        //pixRot = conv_to<fmat>::from(myR)*trans(pix) + pix_max;
    } else {
        //pixRot = conv_to<fmat>::from(myR)*trans(pix) + pix_max; // this is passive rotation
        pixRot = pix*conv_to<fmat>::from(myR) + pix_max; // this is passive rotation
        pixRot = trans(pixRot);  
        //cout << "pixRot: " << pixRot(0,0)- pix_max << " " << pixRot(1.0)- pix_max << " " << pixRot(2,0)- pix_max << endl;
    }
    if ( boost::algorithm::iequals(interpolate,"linear") ) {
        interp_linear3D(myValue,&pixRot,goodpix,myIntensity,myWeight);
    } else if ( boost::algorithm::iequals(interpolate,"nearest") ) {
        interp_nearestNeighbor(myValue,&pixRot,goodpix,myIntensity,myWeight);
    }
}

// Extract an Ewald's slice from a diffraction volume
// active = 1: active rotation
// interpolate = 1: trilinear
void CToolbox::slice3D(fmat *myValue, fmat *myPoints, uvec *goodpix, fmat *myRot, float pix_max, fcube *myIntensity, int active, string interpolate ) {
    fmat& pix = myPoints[0];
    fmat& myR = myRot[0];
    fmat pixRot;
	pixRot.zeros(pix.n_elem,3);
	if (active == 1) {
        pixRot = pix*conv_to<fmat>::from(trans(myR)) + pix_max; // this is active rotation
        pixRot = trans(pixRot);
    } else {
        pixRot = pix*conv_to<fmat>::from(myR) + pix_max; // this is passive rotation
        pixRot = trans(pixRot);  
    }
    //pixRot.print("pixRot: ");
    if ( boost::algorithm::iequals(interpolate,"linear") ) {
        extract_interp_linear3D(myValue,&pixRot,goodpix,myIntensity);
    }// else if ( boost::algorithm::iequals(interpolate,"nearest") ) {
    //    interp_nearestNeighbor(myValue,&pixRot,goodpix,myIntensity,myWeight);
    //}
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

#endif /* SINGFEL_TOOLBOX_H */
