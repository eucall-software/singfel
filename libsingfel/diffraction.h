#ifndef DIFFRACTION_H
#define DIFFRACTION_H
#include <math.h>
#include <armadillo>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include "particle.h"
#include "detector.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

namespace diffraction{

class CDiffraction{

public:
	static arma::fcube f_hkl;
	static arma::fcube f_hkl_list;
	static arma::cx_fmat F_hkl;
	CDiffraction ();
	static double calculate_Thomson(double);
	static void calculate_atomicFactor(particle::CParticle *particle, detector::CDetector *detector);
	static void get_formFactorList(particle::CParticle *particle);
	static void get_atomicFormFactorList(particle::CParticle *particle, detector::CDetector *detector);
	static arma::cube get_FourierMap(particle::CParticle *particle, detector::CDetector *detector);
	static arma::fmat calculate_intensity(particle::CParticle *particle, detector::CDetector *detector);
	
double cuda_func(double);

protected:

private:
	

};

}

double cuda_func(double);

#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */

#endif

