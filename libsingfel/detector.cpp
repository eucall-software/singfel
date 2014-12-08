#include <iostream>
#include <math.h>
#include "detector.h"
#include "toolbox.h"
#include "io.h"
using namespace std;
using namespace arma;
using namespace detector;
using namespace toolbox;

double CDetector::d; 				// (m) detector distance
double CDetector::pix_width;		// (m)
double CDetector::pix_height;		// (m)
int CDetector::px;					// number of pixels in x
int CDetector::py;					// number of pixels in y
int CDetector::numPix;				// total number of pixels (px*py)
double CDetector::cx;				// center of detector in x
double CDetector::cy;				// center of detector in y
fmat CDetector::dp;					// diffraction pattern
fmat CDetector::q_x;				// pixel reciprocal space in x
fmat CDetector::q_y;				// pixel reciprocal space in y
fmat CDetector::q_z;				// pixel reciprocal space in z
fcube CDetector::q_xyz; // slow? perhaps waste of memory
fmat CDetector::q_mod;				// pixel reciprocal space
fmat CDetector::pixSpace;			// pixel reciprocal space in Angstrom
float CDetector::pixSpaceMax;		// max pixel reciprocal space in Angstrom
fmat CDetector::solidAngle;			// solid angle
fmat CDetector::thomson;			// Thomson scattering
uvec CDetector::badpixmap;			// bad pixel map (bad = 1)
uvec CDetector::goodpixmap;			// good pixel map (good = 1)

CDetector::CDetector (){
    d = 0;
    pix_width = 0;
    pix_height = 0;
    px = 0;
    py = 0;
    numPix = 0;
    cx = 0;
    cy = 0;
}

void CDetector::set_detector_dist(double dist){
	d = dist;
}

double CDetector::get_detector_dist(){
	cout << "dist: " << d << endl;
	return d;
}

void CDetector::set_pix_width(double width){
	pix_width = width;
}

double CDetector::get_pix_width(){
	return pix_width;
}

void CDetector::set_pix_height(double height){
	pix_height = height;
}

double CDetector::get_pix_height(){
	return pix_height;
}

void CDetector::set_numPix_x(int x){
	px = x;
}

int CDetector::get_numPix_x(){
	return px;
}

void CDetector::set_numPix_y(int y){
	py = y;
}

void CDetector::set_numPix(int y,int x){
	py = y;
	px = x;
	numPix = py*px;
}

int CDetector::get_numPix_y(){
	return py;
}

void CDetector::set_center_x(double x){
	cx = x;
}

double CDetector::get_center_x(){
	return cx;
}

void CDetector::set_center_y(double y){
	cy = y;
}

double CDetector::get_center_y(){
	return cy;
}

uvec CDetector::get_goodPixelMap(){
    return goodpixmap;
}

uvec CDetector::get_badPixelMap(){
    return badpixmap;
}

void CDetector::set_pixelMap(string x){
	fmat temp;
	if (x.empty()) { // badpixelmap is not specified		
		if (py > 0 && px > 0) {
			temp.zeros(py,px);
		} else {
			cout << "Please set detector dimensions before calling set_pixelMap" << endl;
			exit(0);
		}
	} else { // load badpixelmap
		temp = load_asciiImage(x);
	}
	badpixmap = find(temp == 1);
	goodpixmap = find(temp == 0);
}

void CDetector::apply_badPixels() {
    uvec::iterator a = badpixmap.begin();
    uvec::iterator b = badpixmap.end();
    for(uvec::iterator i=a; i!=b; ++i) {
        dp(*i) = 0;
    }
}

void CDetector::apply_badPixels(fmat *x) {
	fmat& myDP = x[0];
    uvec::iterator a = badpixmap.begin();
    uvec::iterator b = badpixmap.end();
    for(uvec::iterator i=a; i!=b; ++i) {
        myDP(*i) = 0;
    }
}

// set beam object variables and assign q to each pixel
void CDetector::init_dp( beam::CBeam *beam ){
    // this bit should be set already, get rid of this?
	set_detector_dist(d);	
	set_pix_width(pix_width);	
	set_pix_height(pix_height);
	set_numPix(py,px);
	set_center_x(cx);
	set_center_y(cy);	

	// Used for single particle. Set 1/pix_width=9090 for LCLS
	q_xyz.zeros(py,px,3);
	float rx, ry, r, twotheta, az;
	float pixDist, alpha;
	solidAngle.zeros(py,px);
	fmat twoTheta(py,px);
	for (int ind_x = 0; ind_x < px; ind_x++) {
		for (int ind_y = 0; ind_y < py; ind_y++) {
			rx = (ind_x - cx) * pix_width; // equivalent to dividing by pixel resolution
			ry = (ind_y - cy) * pix_width;
			r = sqrt(pow(rx,2)+pow(ry,2));
			twotheta = atan2(r,d);
			twoTheta(ind_y,ind_x) = twotheta;
			az = atan2(ry,rx);
			q_xyz(ind_y,ind_x,0) = beam->get_wavenumber() * sin(twotheta)*cos(az);
			q_xyz(ind_y,ind_x,1) = beam->get_wavenumber() * sin(twotheta)*sin(az);
			q_xyz(ind_y,ind_x,2) = beam->get_wavenumber() * (cos(twotheta) - 1.0);
			
			// Analytical formula of solid angle (Zaluzec2014)
			pixDist = sqrt(pow(rx,2) + pow(ry,2) + pow(d,2)); // distance from interaction to pixel center in real space
			alpha = atan(pix_width/(2*pixDist));
			solidAngle(ind_y,ind_x) = 4 * asin( pow(sin(alpha),2) );
		}
	}
	q_mod = CToolbox::mag(q_xyz);

	double const re = 2.81793870e-15;			// classical electron radius (m)
	// Polarization factor, cos^2 mu + sin^2 mu * cos^2(2theta)
	double mu; // polarization angle
	string electricFieldPlane = "horizontal";
	if (electricFieldPlane == "horizontal") {
		mu = 0;
	} else if (electricFieldPlane == "vertical") {
		mu = datum::pi/2;
	} else {
		// unpolarized
		mu = datum::pi/4;
	}
	fmat polarizationFactor = ( pow(cos(mu),2) + pow(sin(mu),2)*pow(cos(twoTheta),2) );
	thomson = pow(re,2) * polarizationFactor;
	
	// Setup pixel space
	int counter = 0;
	pixSpace.zeros(numPix,3);
	for (int i = 0; i < px; i++) {
		for (int j = 0; j < py; j++) { // column-wise
			pixSpace(counter,0) = q_xyz(j,i,0);
			pixSpace(counter,1) = q_xyz(j,i,1);
			pixSpace(counter,2) = q_xyz(j,i,2);
			counter++;
		}
	}
    fvec pix_mod;
	pixSpace = pixSpace * 1e-10; // (A)
	pix_mod = sqrt(sum(pixSpace%pixSpace,1));		
	pixSpaceMax = max(pix_mod);
    float inc_res = (px-1)/(2*pixSpaceMax/sqrt(2));
    pixSpace = pixSpace * inc_res;
    pix_mod = sqrt(sum(pixSpace%pixSpace,1));		
	pixSpaceMax = cx;
}

void CDetector::set_param(Packet *x){
	umat temp(x->dp, x->py, x->px, false, true);
	dp = conv_to<fmat>::from(trans(temp));
	d = x->d;
	pix_width = x->pix_width;
	pix_height = x->pix_height;
	py = x->py;
	px = x->px;
	cx = ((double) px-1)/2;			// this can be user defined
	cy = ((double) py-1)/2;			// this can be user defined
}

