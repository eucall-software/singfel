// Copyright © 2012-2014 Deutsches Elektronen-Synchrotron DESY,
//                      a research centre of the Helmholtz Association /
//                      European XFEL GmbH
// This program is distributed under the GNU General Public License, GPLv3.

/**
 * @file detector.h
 * @class detector
 * @author  Chun Hong Yoon <chun.hong.yoon@desy.de> 
 *
 * @brief  The class which handles various detectors.
 *
 * This class can be used to perform phase diverse or ptychographic
 * reconstruction of either Fresnel or plane-wave CDI data. Any number
 * of frames (also called local/single frames or probes in this
 * documentation) may be added to the reconstruction. In order to
 * perform a reconstruction, you will need to create either a new
 * FresnelCDI or PlanarCDI object for each of these 'local' datasets.
 * The FresnelCDI or PlanarCDI objects must then be passed to a
 * PhaseDiverseCDI object. Because the each sub-iteration (ie. each
 * iteration of a local frame) is performed using the code in
 * FresnelCDI/PlanarCDI, all the functionality available in these
 * classes is also available here. For example complex constraint can
 * be set, shrink-wrap can be used, the estimate can be initialised
 * using the results from a previous reconstruction etc. An example
 * can be found in the /examples directory, demonstrating how to use
 * PhaseDiverseCDI.
 *
 * The displacement in position between 'local' frames may be either
 * transverse to the beam direction, or longitudinal, along the beam
 * direction. The longitudinal position (for FresnelCDI) is set when
 * the FresnelCDI objects are initially constructed. Transverse
 * positions are set when the FresnelCDI/PlanarCDI objects are added to
 * the PhaseDiverseCDI object. This class allows the transverse
 * positions to be automatically adjusted during reconstruction using
 * the "adjust_positions" function.
 *
 * The code allows for several options in the type of reconstruction
 * which is done:
 * <ul>
 * <li> The reconstruction can be performed in series or parallel.
 *   <ul>
 *   <li> In the case of series (see for example the paper....), a 'local'
 *   frame will undergo one or more iterations, the result will be
 *   updated to a 'global' estimate of the sample, and this estimate
 *   will form the starting point for the next frame. For each call to
 *   the method "iterate()" this process is repeated until each local
 *   frame is used once. The algorithm can be described by: <br> \f$
 *   T_{k+1} = (1-\beta w^n)T_k + \beta w^n T^n_k\f$ <br> where
 *   \f$T_{k+1}\f$ is the updated global function, \f$T^n\f$ is the
 *   updated local function for the nth local frame. \f$\beta\f$ is
 *   the relaxation parameter and the weight is \f$ w^n(\rho) =
 *   \alpha^n (\frac{|T^n(\rho)|}{max|T^n(\rho)|} )^\gamma \f$ for
 *   Fresnel CDI or \f$w^n = \alpha^n\f$ for Plane-wave CDI. The
 *   weight is zero outside of the support.
 *
 *   <li> For a parallel reconstruction (see .....), each frame with
 *   independently undergo one of more iteration, the result from all
 *   frames will be merged to form a new estimate of the sample, this
 *   estimate then becomes the starting point for the next iteration
 *   of all frames. The algorithm can be described by:
 *   <br> \f$ T_{k+1} = (1-\beta)T_k + \beta \sum_n(w^n T^n_k)\f$
 *   <br> where \f$T_{k+1}\f$, \f$T^n\f$, and \f$\beta\f$ were defined
 *   earlier. The weight, w, is similar to that used for series
 *   reconstruction, but the weight it normalised such that 
 *   \f$ \sum_n w^n = 1 \f$.  i.e. 
 *   \f$ w^n_{parallel}= w^n_{series} / \sum_n w^n_{series} \f$
 *  </ul> 
 *
 * <li> The number of local iterations to perform before updating the
 *   result to the 'global' function can be set.
 *
 * <li> The feedback parameter, beta, may be set. This quantity is used
 *   to set how much of the previous 'global' sample function will be
 *   left after the next 'global' iteration.
 *
 * <li> The amplification factor, gamma, and the probe scaling, alpha,
 *   may also be see. These parameters control the weighting of one
 *   frame (and pixels within a frame) with respect to each
 *   other.
 * </ul>
 */

#ifndef DETECTOR_H
#define DETECTOR_H
#include <armadillo>
#include "beam.h"
#include "io.h"
#include "detector.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace detector{

class CDetector{
public:
	static double d; 				// (m) detector distance
	static double pix_width;		// (m)
	static double pix_height;		// (m)
	static int px;					// number of pixels in x
	static int py;					// number of pixels in y
	static int numPix;				// px * py
	static double cx;				// center of detector in x
	static double cy;				// center of detector in y
	static arma::fmat dp;					// diffraction pattern
	static arma::fmat q_x;
	static arma::fmat q_y;
	static arma::fmat q_z;
	static arma::fcube q_xyz;
	static arma::fmat q_mod;
	static arma::fmat pixSpace;			// pixel reciprocal space in Angstrom
	static float pixSpaceMax;		// max pixel reciprocal space in Angstrom
	static arma::fmat solidAngle;
	static arma::fmat thomson;
	static arma::uvec badpixmap; //static arma::sp_imat badpixmap;
	static arma::uvec goodpixmap;
public:
	CDetector ();
	//CDetector (int);
	static void set_detector_dist(double);
	static double get_detector_dist();
	static void set_pix_width(double);
	static double get_pix_width();
	static void set_pix_height(double);
	static double get_pix_height();
	static void set_numPix_x(int);
	static int get_numPix_x();
	static void set_numPix_y(int);
	static int get_numPix_y();
	static void set_numPix(int,int);
	static void set_center_x(double);
	static double get_center_x();
	static void set_center_y(double);
	static double get_center_y();
	void set_pixelMap(std::string);
	arma::uvec get_goodPixelMap();
	arma::uvec get_badPixelMap();
	void apply_badPixels();
	static void apply_badPixels(arma::fmat*);	
	static void init_dp(beam::CBeam *beam);
	static void set_param(Packet*);
protected:

private:

};

}

#ifdef __cplusplus
}
#endif

#endif

