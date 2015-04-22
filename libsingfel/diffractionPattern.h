// Copyright © 2015

/**
 * @file diffractionPattern.h
 * @class diffractionPattern
 * @author  Chun Hong Yoon <chun.hong.yoon@desy.de> 
 *
 * @brief  blah blah blah
 */

#ifndef DIFFRACTIONPATTERN_H
#define DIFFRACTIONPATTERN_H
#include <iostream>
#include <iomanip>
#include <armadillo>
#include <boost/program_options.hpp>
#include "io.h"

namespace opt = boost::program_options;

#ifdef __cplusplus
extern "C" {
#endif

namespace diffractionPattern{

class CDiffrPat{
public:
	int px;					// number of pixels in x
	int py;					// number of pixels in y
	int numPix;				// total number of pixels (px*py)
	arma::fmat photonCount;	// diffraction pattern
	arma::uvec photonpixmap;	// index where photon counts are present
	CDiffrPat();
	void init(int volDim);
	void loadPhotonCount(opt::variables_map vm, int ind);
	void updateExpansionSlice(opt::variables_map vm, fvec* normCondProb, uvec* candidatesInd);
	void loadUpdatedExpansion(opt::variables_map vm, int iter, int sliceInd);
	void loadExpansionSlice(opt::variables_map vm, int iter, int sliceInd);
	void calculateWeightedImage(const float weight, CDiffrPat* myDP);
	void saveExpansionUpdate(opt::variables_map vm, int iter, int expansionInd);
	void saveExpansionSlice(opt::variables_map vm, int iter, int ind);
protected:
private:
};

}

#ifdef __cplusplus
}
#endif

#endif
