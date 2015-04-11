#include <iostream>
#include "beam.h"
using namespace std;
using namespace beam;
using namespace arma;

double CBeam::lambda; 							// (m) wavelength
double CBeam::photon_energy; 					// (eV) photon energy
double CBeam::k;								// (m^-1)
double CBeam::focus_xFWHM;							// (m) beam focus diameter
double CBeam::focus_yFWHM;							// (m) beam focus diameter
string CBeam::focus_shape;                      // focus shape: {square, default:circle}
double CBeam::focus_area; 						// (m^2)
double CBeam::n_phot;							// number of photons per pulse
double CBeam::phi_in;  							// number of photon per pulse per area (m^-2)

CBeam::CBeam (){
    lambda = 0;
	photon_energy = 0;
	k = 0;
	focus_xFWHM = 0;
	focus_yFWHM = 0;
	focus_shape = "circle";
	focus_area = 0;
	n_phot = 0;
	phi_in = 0;
}

void CBeam::update(){
    if (photon_energy != 0) {
        lambda = photonEnergy2wavlength(photon_energy);
        k = wavelength2wavenumber(lambda);
    } else if (lambda != 0) {
        k = wavelength2wavenumber(lambda);
        photon_energy = wavelength2photonEnergy(lambda);
    } else if (k != 0) {
        lambda = wavenumber2wavelength(k);
        photon_energy = wavelength2photonEnergy(lambda);
    }
    if (focus_xFWHM != 0) {
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

double CBeam::wavenumber2wavelength(double k){
	return 1/k;
}

double CBeam::photonEnergy2wavlength(double photonEnergy){
	return 1.2398e-6/photonEnergy;
}

double CBeam::wavelength2photonEnergy(double wavelength){
	return 1.2398e-6/wavelength;
}

void CBeam::set_photon_energy(double ev){
	photon_energy = ev;
	update();
}

double CBeam::get_photon_energy(){
	return photon_energy;
}

double CBeam::get_wavelength(){
	return lambda;
}

double CBeam::get_wavenumber(){
	return k;
}

void CBeam::set_focus(double x){
	focus_xFWHM = x;
	update();
}

void CBeam::set_focus(double x, string shape){
	focus_xFWHM = x;
	focus_shape = shape;
	update();
}

void CBeam::set_focus(double x, double y, string shape){
	focus_xFWHM = x;
	focus_yFWHM = y;
	focus_shape = shape;
	update();
}

double CBeam::get_focus(){
	return focus_xFWHM;
}

void CBeam::set_focusArea() {
	if (focus_shape == "square") {
		focus_area = focus_xFWHM * focus_yFWHM;	// d^2
	} else if (focus_shape == "ellipse") {
		focus_area = datum::pi * focus_xFWHM/2 * focus_yFWHM/2;	// pi*rx*ry
	} else {
		focus_area = datum::pi * pow(focus_xFWHM/2,2); // pi*r^2
	}
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
	focus_xFWHM = pack->focus;
	n_phot = pack->n_phot;	
	update();
}

void CBeam::readBeamFile(string beamFile) {
	/****** Beam ******/
	// Parse the beam file
	string line;
	ifstream myFile(beamFile.c_str());
	while (getline(myFile, line)) {
		if (line.compare(0,1,"#") && line.compare(0,1,";") && line.length() > 0) {
			// line now contains a valid input
			cout << line << endl;
			typedef boost::tokenizer<boost::char_separator<char> > Tok;
			boost::char_separator<char> sep(" ="); // default constructed
			Tok tok(line, sep);
			for(Tok::iterator tok_iter = tok.begin(); tok_iter != tok.end(); ++tok_iter){
			    if ( boost::algorithm::iequals(*tok_iter,"beam/photon_energy") ) {            
			        string temp = *++tok_iter;
			        photon_energy = atof(temp.c_str()); // photon energy to wavelength
			        break;
			    } else if ( boost::algorithm::iequals(*tok_iter,"beam/fluence") ) {            
			        string temp = *++tok_iter;
			        n_phot = atof(temp.c_str()); // number of photons per pulse
			        break;
			    } else if ( boost::algorithm::iequals(*tok_iter,"beam/radius") ) {            
			        string temp = *++tok_iter;
			        double focus_xFWHM = atof(temp.c_str()); // focus radius
			        break;
			    }
			}
		}
	}
	update();
}

#ifdef COMPILE_WITH_BOOST
	#include <boost/python.hpp>
	using namespace boost::python;
	using namespace beam;
	using namespace arma;

	BOOST_PYTHON_MODULE(libbeam)
	{
		class_<CBeam>("CBeam", init<>())	// constructor
			//.def_readwrite("name", &CToolbox::name)
			//.add_property("photon_energy", &CBeam::set_photon_energy, &CBeam::get_photon_energy)
			.def("set_photon_energy", &CBeam::set_photon_energy)
			.def("get_photon_energy", &CBeam::get_photon_energy)
		;
	}
#endif
