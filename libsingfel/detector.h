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
	static arma::umat dp;					// diffraction pattern
	static arma::fmat q_x;
	static arma::fmat q_y;
	static arma::fmat q_z;
	static arma::fcube q_xyz;
	static arma::fmat q_mod;
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

