#include <iostream>
#include <armadillo>
#include "diffraction.h"
using namespace std;
using namespace diffraction;
using namespace arma;
//#define ARMA_NO_DEBUG

fcube CDiffraction::f_hkl;
fcube CDiffraction::f_hkl_list;

CDiffraction::CDiffraction() {
}

void CDiffraction::displayResolution(detector::CDetector* det, beam::CBeam* beam) {
	double d = det->get_detector_dist();
	double pix_height = det->get_pix_height();
	int py = det->get_numPix_y();
	
	double theta = atan((py/2*pix_height)/d);
	double qmax = 2/beam->get_wavelength()*sin(theta/2);
	double dmin = 1/(2*qmax);
	cout << "max q to the edge: " << qmax*1e-10 << " A^-1" << endl;
	cout << "Half period resolution: " << dmin*1e10 << " A" << endl;
}

double CDiffraction::calculate_Thomson(double ang) {
// Should fix this to accept angles mu and theta
	const double re = 2.81793870e-15;			// classical electron radius (m)
	double P = (1 + cos(ang))/2;
	return pow(re,2) * P;	// Thomson scattering (m^2)
}

void CDiffraction::calculate_atomicFactor(particle::CParticle *particle, detector::CDetector *detector) {
	// Atomic form factor
	f_hkl.zeros(detector->py,detector->px,particle->numAtomTypes);
	fmat q_mod_Bragg = detector->q_mod * (1e-10/2);
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
		
	gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, particle->numQSamples); // 200 should be function of numQSamples
	double qs[particle->numQSamples], ft[particle->numQSamples];
	// silly copying of arma::rowvec to vector
	for(int i=0; i<particle->numQSamples; ++i) {
		qs[i] = particle->qSample(i);
	}
	for(int j=0; j<particle->numAtomTypes; ++j) {
		// silly copying of arma::rowvec to vector
		for(int i=0; i<particle->numQSamples; ++i) {
			ft[i] = particle->ffTable(j,i); // 0 is first row
  		}	
		gsl_spline_init(spline, qs, ft, particle->numQSamples);	
		for(int a=0; a<detector->py; a++) {
		for(int b=0; b<detector->px; b++) {
			f_hkl(a,b,j) = gsl_spline_eval(spline, q_mod_Bragg(a,b), acc); // interpolate
		}
		}
	}
	gsl_spline_free (spline);
	gsl_interp_accel_free (acc);
}

void CDiffraction::get_formFactorList(particle::CParticle *particle) {
	for(int j = 0; j < particle->numAtoms; ++j) {
		particle->formFactorList(j) = (unsigned int) particle->xyzInd(j);
	}
}

void CDiffraction::get_atomicFormFactorList(particle::CParticle *particle, detector::CDetector *detector) {
	f_hkl_list.zeros(detector->py,detector->px,particle->numAtoms);
	for(int j = 0; j < particle->numAtoms; ++j) {
		f_hkl_list.slice(j) = f_hkl.slice((unsigned int) particle->xyzInd(j));
	}
}

cube CDiffraction::get_FourierMap(particle::CParticle *particle, detector::CDetector *detector) {
	cube map = zeros(detector->py,detector->px,particle->numAtoms);
	frowvec r(3);
	fcolvec c(3);
	for (int ind_x = 0; ind_x < detector->px; ind_x++) {
	for (int ind_y = 0; ind_y < detector->py; ind_y++) {
	for (int j = 0; j < particle->numAtoms; ++j) {
		r = detector->q_xyz.subcube( span(ind_y),  span(ind_x),  span() );
		c = strans(particle->atomPos.row(j));
		map(ind_y,ind_x,j) = as_scalar(r * c);
	}
	}
	}
	map = 2*datum::pi*map;
	return map;
}

// Calculate molecular form factor |F_hkl(q)|^2.
// Rename to calculate_molecularFormFactorSq
fmat CDiffraction::calculate_intensity(particle::CParticle *particle, detector::CDetector *detector) {
	fmat F_hkl_sq;
	F_hkl_sq.zeros(detector->py,detector->px);
	
	fcolvec map;
	frowvec f;
	fcolvec q;
	for (int ind_x = 0; ind_x < detector->px; ind_x++) {
		for (int ind_y = 0; ind_y < detector->py; ind_y++) {
			f = CDiffraction::f_hkl_list(span(ind_y),span(ind_x),span()); // 1xN
			q = detector->q_xyz(span(ind_y),span(ind_x),span()); // py x px x 3
			map = 2*datum::pi*particle->atomPos * q; // Nx3 3x1
			F_hkl_sq(ind_y,ind_x) = as_scalar( pow(f * cos(map),2) + pow(f * sin(map),2) ); // 1xN Nx1
		}
	}
	return F_hkl_sq;
}

// Calculate static structure factor S(q) and number of free electrons N_free(q)
void CDiffraction::calculate_compton(particle::CParticle *particle, detector::CDetector *detector, fmat* Compton) {
	fmat& _Compton = Compton[0];
	fmat S_bound;
	float N_free;
	_Compton.zeros(detector->py,detector->px);
	S_bound.zeros(detector->py,detector->px);
	N_free = particle->nFree(0);

	fmat half_q = detector->q_mod * (1e-10/2);
	gsl_interp_accel *acc = gsl_interp_accel_alloc();

	gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, particle->numComptonQSamples);
	double qs[particle->numComptonQSamples], ft[particle->numComptonQSamples];
	// silly copying of arma::rowvec to vector
	for(int i=0; i<particle->numComptonQSamples; ++i) {
		qs[i] = particle->comptonQSample(i);
	}

	// silly copying of arma::rowvec to vector
	for(int i=0; i<particle->numComptonQSamples; ++i) {
		ft[i] = particle->sBound(i); // 0 is first row
  	}
	
	gsl_spline_init(spline, qs, ft, particle->numComptonQSamples);
		
	for(int a=0; a<detector->py; a++) {
	for(int b=0; b<detector->px; b++) {
		S_bound(a,b) = gsl_spline_eval(spline, half_q(a,b), acc); // interpolate
	}
	}

	gsl_spline_free (spline);
	gsl_interp_accel_free (acc);
	
	_Compton = S_bound + N_free;
}
