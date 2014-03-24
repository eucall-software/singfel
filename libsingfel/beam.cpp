#include <iostream>
#include "beam.h"
using namespace std;
using namespace beam;
using namespace arma;

double CBeam::lambda; 							// (m) wavelength
double CBeam::photon_energy; 					// (eV) photon energy
double CBeam::k;								// (m^-1)
double CBeam::focus;							// (m)
double CBeam::focus_area; 						// (m^2)
double CBeam::n_phot;							// number of photons per pulse
double CBeam::phi_in;  							// number of photon per pulse per area (m^-2)

CBeam::CBeam (){
	//cout << "init beam" << endl;
    lambda = 0;
	photon_energy = 0;
	k = 0;
	focus = 0;
	focus_area = 0;
	n_phot = 0;
	phi_in = 0;
}

void CBeam::update(){
    if (photon_energy != 0) {
        lambda = photonEnergy2wavlength(photon_energy);
        k = wavelength2wavenumber(lambda);
    }
    if (focus != 0) {
        set_focusArea();
        if (n_phot != 0) {
            set_photonsPerPulsePerArea();
        }
    }
}

void CBeam::set_wavelength(double x){
	lambda = x;
	update();
}

double CBeam::wavelength2wavenumber(double lambda){
	return 1/lambda;
}

double CBeam::photonEnergy2wavlength(double photonEnergy){
	return 1.2398e-6/photonEnergy;
}

void CBeam::set_photon_energy(double ev){
	photon_energy = ev;
	update();
}

double CBeam::get_wavelength(){
	return lambda;
}

double CBeam::get_wavenumber(){
	return k;
}

void CBeam::set_focus(double x){
	focus = x;
	update();
}

double CBeam::get_focus(){
	return focus;
}

void CBeam::set_focusArea(){
	focus_area = datum::pi * pow(focus/2,2);	// pi*r^2
}

void CBeam::set_photonsPerPulse(double x){
	n_phot = x;
	update();
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

void CBeam::set_param(Packet *pack){
	lambda = pack->lambda;
	//k = wavelength2wavenumber(lambda);
	focus = pack->focus;
	//set_focusArea();	// pi*r^2
	n_phot = pack->n_phot;	
	//set_photonsPerPulsePerArea();
	update();
//cout << "phi_in=n_phot/focus_area: " << phi_in << endl;	
}
