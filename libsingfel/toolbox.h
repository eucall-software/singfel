#ifndef TOOLBOX_H
#define TOOLBOX_H
#include <armadillo>
#include <string>
#include <sstream>
#include <iostream>
#include "detector.h"
#include <boost/algorithm/string.hpp>

#ifdef __cplusplus
extern "C" {
#endif

namespace toolbox{

class CToolbox{
public:
    CToolbox(std::string n) : name(n) {}
    std::string name;
	static arma::mat mag(arma::cube);
	static arma::fmat mag(arma::fcube);
	static arma::umat convert_to_poisson(arma::fmat);
	static arma::fmat quaternion2rot3D(arma::vec,int);
	static void quaternion2AngleAxis(arma::vec,float&,arma::vec&);
	static arma::fmat angleAxis2rot3D(arma::vec,float);
	static arma::fmat euler2rot3D(float, float, float);
	static arma::fvec rot3D2euler(arma::fmat);
	
	//static void interp_linear3D(arma::fmat*, arma::fmat*, arma::fcube*, arma::fcube*);
	//static void interp_nearestNeighbor(arma::fmat*, arma::fmat*, arma::fcube*, arma::fcube*);
	//static void merge3D(arma::fmat*, arma::fmat*, arma::fmat*, float, arma::fcube*, arma::fcube*, int);
	
	static void interp_linear3D(arma::fmat*, arma::fmat*, arma::uvec*, arma::fcube*, arma::fcube*);
	static void interp_nearestNeighbor(arma::fmat*, arma::fmat*, arma::uvec*, arma::fcube*, arma::fcube*);
	static void merge3D(arma::fmat*, arma::fmat*, arma::uvec*, arma::fmat*, float, arma::fcube*, arma::fcube*, int active = 0, std::string interpolate="nearest");
		
	static void normalize(arma::fcube*, arma::fcube*);
};

}

#ifdef __cplusplus
}
#endif

#endif
