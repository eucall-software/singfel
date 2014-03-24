#include <iostream>
#include "beam.h"
using namespace std;
using namespace beam;
using namespace arma;

double CBeam::lambda; 							// (m) wavelength
double CBeam::k;								// (m^-1)
double CBeam::focus;							// (m)
double CBeam::focus_area; 						// (m^2)
double CBeam::n_phot;							// number of photons per pulse
double CBeam::phi_in;  							// number of photon per pulse per area (m^-2)

CBeam::CBeam (){
	//cout << "init beam" << endl;
}

void CBeam::set_wavelength(double x){
	lambda = x;
	k = 1/lambda;
}

double CBeam::get_wavelength(){
	return lambda;
}

void CBeam::set_focus(double x){
	focus = x;
	focus_area = datum::pi * pow(focus/2,2);	// pi*r^2
}

double CBeam::get_focus(){
	return focus;
}

void CBeam::set_photonsPerPulse(double x){
	n_phot = x;
}

double CBeam::get_photonsPerPulse(){
	return n_phot;
}

double CBeam::get_focus_area(){
	return focus_area;
}

void CBeam::set_photonsPerPulsePerArea(){
	// need to check the variable are not empty
	phi_in = n_phot/focus_area;
}

double CBeam::get_photonsPerPulsePerArea(){
	return phi_in;
}

void CBeam::set_param(Packet *x){
	lambda = x->lambda;
	k = 1/lambda;
	focus = x->focus;
	focus_area = datum::pi * pow(focus/2,2);	// pi*r^2
	n_phot = x->n_phot;	
	phi_in = n_phot/focus_area;
//cout << "phi_in=n_phot/focus_area: " << phi_in << endl;	
}