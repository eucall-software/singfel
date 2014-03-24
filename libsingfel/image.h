#ifndef IMAGE_H
#define IMAGE_H
#include <armadillo>
#include "beam.h"
#include "io.h"
#include "detector.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace image{

class CImage{
public:
	static arma::fmat dp;			// diffraction pattern
	static arma::uvec badpixmap;    //static arma::sp_imat badpixmap;
	static arma::uvec goodpixmap;
public:
	CImage ();
protected:

private:

};

}

#ifdef __cplusplus
}
#endif

#endif

