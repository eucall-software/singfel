#ifndef BEAM_H
#define BEAM_H
#include <armadillo>
#include "io.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace beam{

class CBeam{
private:
	static double lambda; 							// (m) wavelength
	static double photon_energy;
	static double k;								// (m^-1)
	static double focus_xFWHM;							// (m)
	static double focus_yFWHM;							// (m)
	static string focus_shape;                      // focus shape: {square,default:circle}
	static double focus_area; 						// (m^2)
	static double n_phot;							// number of photons per pulse
	static double phi_in;  							// number of photon per pulse per area (m^-2)
public:
	CBeam ();
	static void set_wavelength(double);
	static double wavelength2wavenumber(double);
	static double wavenumber2wavelength(double);
	static double photonEnergy2wavlength(double);
	static double wavelength2photonEnergy(double);
	static void set_photon_energy(double);
	static double get_photon_energy();
	static double get_wavelength();
	static double get_wavenumber();
	static void set_focus(double);
	static void set_focus(double,string);
	static void set_focus(double x, double y, string);	
	static double get_focus();
	static void set_focusArea();
	static double get_focus_area();
	static void set_photonsPerPulse(double);
	static double get_photonsPerPulse();
	static void set_photonsPerPulsePerArea();
	static double get_photonsPerPulsePerArea();
	static void set_param(Packet*);
	static void update();
};

}

#ifdef __cplusplus
}
#endif

#endif

