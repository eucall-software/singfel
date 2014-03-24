#include <iostream>
#include <math.h>
#include "detector.h"
#include "toolbox.h"
using namespace std;
using namespace arma;
using namespace detector;
using namespace toolbox;

double CDetector::d; 				// (m) detector distance
double CDetector::pix_width;		// (m)
double CDetector::pix_height;		// (m)
int CDetector::px;					// number of pixels in x
int CDetector::py;					// number of pixels in y
int CDetector::numPix;
double CDetector::cx;				// center of detector in x
double CDetector::cy;				// center of detector in y
umat CDetector::dp;					// diffraction pattern
fmat CDetector::q_x;
fmat CDetector::q_y;
fmat CDetector::q_z;
fcube CDetector::q_xyz; // slow? perhaps waste of memory
fmat CDetector::q_mod;
fmat CDetector::solidAngle;
fmat CDetector::thomson;
uvec CDetector::badpixmap;//sp_imat CDetector::badpixmap;
uvec CDetector::goodpixmap;

CDetector::CDetector (){
    d = 0;
    pix_width = 0;
    pix_height = 0;
    px = 0;
    py = 0;
    numPix = 0;
    cx = 0;
    cy = 0;
	//cout << "init detector" << endl;
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
	fmat temp = load_asciiImage(x);
	badpixmap = find(temp == 1);
	badpixmap.print("badpix: ");
	
	goodpixmap = find(temp == 0);
	/*
	badpixmap.copy_size(temp);
	for (int i = 0; i < temp.n_rows; i++) {
		for (int j = 0; j < temp.n_cols; j++) {
		    if (temp(i,j) == 1) { 
	            badpixmap(i,j) = 1;
	        }
	    }
	}
	badpixmap.print("badpixmap: ");
	// Wrong way to access sparse elements
	cout << badpixmap(0) << endl;
	cout << badpixmap(1) << endl;
	// Correct way to access sparse elements
	sp_imat::iterator a = badpixmap.begin();
    sp_imat::iterator b = badpixmap.end();
    for(sp_imat::iterator i=a; i!=b; ++i) {
        cout << *i << endl;
    }
    */
}

void CDetector::apply_badPixels() {
    //fmat& myDP = in[0];
    uvec::iterator a = badpixmap.begin();
    uvec::iterator b = badpixmap.end();
    for(uvec::iterator i=a; i!=b; ++i) {
        cout << *i << endl;
        dp(*i) = 0;
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
//cout << "Got here" << endl;	

	// Used for LCLS making diffraction volume
	q_xyz.zeros(py,px,3);
	float rx, ry, r, twotheta, az;
	for (int ind_x = 0; ind_x < px; ind_x++) {
		for (int ind_y = 0; ind_y < py; ind_y++) {
			rx = (ind_x - cx) * pix_width;
			ry = (ind_y - cy) * pix_width;
			r = sqrt(pow(rx,2)+pow(ry,2));
			twotheta = atan2(r,d);
			az = atan2(ry,rx);
			q_xyz(ind_y,ind_x,0) = beam->get_wavenumber() * sin(twotheta)*cos(az);
			q_xyz(ind_y,ind_x,1) = beam->get_wavenumber() * sin(twotheta)*sin(az);
			q_xyz(ind_y,ind_x,2) = beam->get_wavenumber() * (cos(twotheta) - 1.0);
			
			//if (ind_x==0 && ind_y==0) {
			//cout << "cnx,cny,res: " << cx<<","<<cy<<","<<1./pix_width<<endl;
			//cout << "fs,ss: " << ind_x<<","<<ind_y<<endl;
			//cout << "cx,lambda: " << cx <<"," << beam->get_wavelength() << endl;
			//cout << "rx,ry,r,twotheta: " <<rx<<","<<ry<<","<<r<<","<<twotheta<<","<<az<<","<<q_xyz(ind_y,ind_x,0)<<","<<q_xyz(ind_y,ind_x,1)<<","<<q_xyz(ind_y,ind_x,2)<<endl;
			//}
		}
	}

/* // Hard coded 9090
	// Used for LCLS making diffraction volume
	q_xyz.zeros(py,px,3);
	float rx, ry, r, twotheta, az;
	for (int ind_x = 0; ind_x < px; ind_x++) {
		for (int ind_y = 0; ind_y < py; ind_y++) {
			rx = (ind_x - cx) / 9090.0;
			ry = (ind_y - cy) / 9090.0;
			r = sqrt(pow(rx,2)+pow(ry,2));
			twotheta = atan2(r,d);
			az = atan2(ry,rx);
			q_xyz(ind_y,ind_x,0) = beam->k * sin(twotheta)*cos(az);
			q_xyz(ind_y,ind_x,1) = beam->k * sin(twotheta)*sin(az);
			q_xyz(ind_y,ind_x,2) = beam->k * (cos(twotheta) - 1.0);
			//if (ind_x==331 && ind_y==513) {
			//cout << "cnx,cny,res: " << cx<<","<<cy<<","<<1./pix_width<<endl;
			//cout << "fs,ss: " << ind_x<<","<<ind_y<<endl;
			//cout << "rx,ry,r,twotheta: " <<rx<<","<<ry<<","<<r<<","<<twotheta<<","<<az<<","<<q_xyz(ind_y,ind_x,0)<<","<<q_xyz(ind_y,ind_x,1)<<","<<q_xyz(ind_y,ind_x,2)<<endl;
			//}
		}
	}
*/
/*		
	q_xyz.zeros(py,px,3);
	fmat r_x, r_y, k;
	frowvec coord_x = (linspace<frowvec>(0.0, px-1, px) - cx) * pix_width;
	fcolvec coord_y = (cy - linspace<fcolvec>(0.0, py-1, py)) * pix_height;
	//cout << coord_x << endl;
	double c2r = 1/(beam->lambda*d); // convert to reciprocal space 
	fmat coord_x_mat = repmat(coord_x, py, 1);
	fmat coord_y_mat = repmat(coord_y, 1, px);
	r_x = coord_x_mat*c2r; // reciprocal coord
	r_y = coord_y_mat*c2r;
	//cout << "cx,cy: " << cx << "," << cy << endl;
cout << "r_x r_y: " << r_x(0,0) << "," << r_y(0,0) << endl;
	k = beam->k * ones<fmat>(py,px);
	fmat c2e = k/(sqrt( pow(r_x,2) + pow(r_y,2) + pow(k,2) ));
	q_xyz.slice(0) = r_x % c2e;
	q_xyz.slice(1) = r_y % c2e;
	q_xyz.slice(2) = k % (c2e - 1);
	q_mod = CToolbox::mag(q_xyz);
	// twotheta
	fmat radial = sqrt(pow(coord_x_mat,2)+pow(coord_y_mat,2));
	fmat twotheta;
	twotheta.zeros(py,px);
	for (int ind_x = 0; ind_x < px; ind_x++) {
	for (int ind_y = 0; ind_y < py; ind_y++) {
		twotheta(ind_y,ind_x) = atan2(radial(ind_y,ind_x),d);
	}
	}
	fmat r_sq =  pow(coord_x_mat,2) + pow(coord_y_mat,2) + pow(d,2); // real space
	solidAngle = pix_width * pix_height * cos(twotheta) / r_sq; // real space (Unitless)
//cout << "solidAngle:" << solidAngle << endl;
*/
/******************THOMSON SCATTERING FIX HERE********************/
	double re = 2.81793870e-15;			// classical electron radius (m)
	// Vertical polarization, mu = pi/2, cos^2(mu) = 0
	fmat tmp(py,px);
	thomson = pow(re,2) * tmp.ones();
}

void CDetector::set_param(Packet *x){
	umat temp(x->dp, x->py, x->px, false, true);
	dp = trans(temp);
	d = x->d;
	pix_width = x->pix_width;
	pix_height = x->pix_height;
	py = x->py;
	px = x->px;
	cx = ((double) px-1)/2;			// this can be user defined
	cy = ((double) py-1)/2;			// this can be user defined
}

