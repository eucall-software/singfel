// Copyright © 2015

/**
 * @file diffractionVolume.h
 * @class diffractionVolume
 * @author  Chun Hong Yoon <chun.hong.yoon@desy.de> 
 *
 * @brief  blah blah blah
 */

#ifndef DIFFRACTIONVOLUME_H
#define DIFFRACTIONVOLUME_H
#include <armadillo>
#include <boost/program_options.hpp>
#include <iostream>
#include <iomanip>
#include "io.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace opt = boost::program_options;

namespace diffractionVolume{
	class CDiffrVol{
	public:
		CDiffrVol();
		CDiffrVol(int px);
		void randVol();
		void initVol();		// initialize diffraction volume
		void initVol(int px);			// initialize diffraction volume
		void loadInitVol(opt::variables_map vm);
		void normalize();
		int saveDiffractionVolume(opt::variables_map vm, int iter);
		void saveDiffractionVolume(opt::variables_map vm);
		static arma::fcube intensity;	// 3D diffraction volume
		static arma::fcube weight;		// 3D diffraction volume weight
		static int volDim;				// number of pixels in x,y,z
	protected:
	private:
	

		// TODO: move reconMPI::saveDiffractionVolume here;

	};
}

#ifdef __cplusplus
}
#endif

#endif
