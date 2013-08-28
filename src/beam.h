#ifndef BEAM_H
#define BEAM_H
#include <armadillo>
#include "io.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace beam{

class CBeam{
public:
	static double lambda; 							// (m) wavelength
	static double k;								// (m^-1)
	static double focus;							// (m)
	static double focus_area; 						// (m^2)
	static double n_phot;							// number of photons per pulse
	static double phi_in;  							// number of photon per pulse per area (m^-2)
public:
	CBeam ();
	static void set_wavelength(double);
	static double get_wavelength();
	static void set_focus(double);
	static double get_focus();
	static double get_focus_area();
	static void set_photonsPerPulse(double);
	static double get_photonsPerPulse();
	static void set_photonsPerPulsePerArea();
	static double get_photonsPerPulsePerArea();
	static void set_param(Packet*);
};

}
#ifdef __cplusplus
}
#endif

#endif

