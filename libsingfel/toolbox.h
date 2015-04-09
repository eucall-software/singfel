#ifndef TOOLBOX_H
#define TOOLBOX_H
#include <armadillo>
#include <string>
#include <sstream>
#include <iostream>
#include "detector.h"
#include <boost/algorithm/string.hpp>
#include <boost/mpi.hpp>
#include <boost/program_options.hpp>

namespace opt = boost::program_options;
namespace mpi = boost::mpi;

#ifdef __cplusplus
extern "C" {
#endif

namespace toolbox{

class CToolbox{
	
public:
	CToolbox() {}
	CToolbox(std::string n) : name(n), mNumber(0.0) {}
	std::string name;
    
	double getNumber() const { return mNumber; }
	void setNumber(double n) {
		if (n>3.141592654)
			n = -1;
		mNumber = n;
	}
    
	static arma::mat mag(arma::cube);
	static arma::fmat mag(arma::fcube);
	//static arma::umat convert_to_poisson(arma::fmat);
	static arma::umat convert_to_poisson(arma::fmat* z);
	// Let's use zyz convention after Heymann (2005)
	static void quaternion2AngleAxis(arma::fvec,float&,arma::fvec&); // mutual convention
	static arma::fmat angleAxis2rot3D(arma::fvec,float); // mutual convention
	static arma::fmat quaternion2rot3D(arma::fvec); // mutual convention
	static arma::fvec euler2quaternion(float, float, float);
	static arma::fvec quaternion2euler(arma::fvec);
	static arma::fmat euler2rot3D(float, float, float);
	static arma::fvec rot3D2euler(arma::fmat);
	static float corrCoeff(arma::fmat, arma::fmat);
	static arma::fvec getRandomRotation(string rotationAxis);
	
	static arma::fmat get_wahba(arma::fmat,arma::fmat);
	static arma::fmat pointsOn1Sphere(int numPts, string rotationAxis);
	static arma::fmat pointsOn3Sphere(int numPts);
	static arma::fmat pointsOn4Sphere(int numPts);
	
	static void extract_interp_linear3D(arma::fmat*, arma::fmat*, arma::uvec*, arma::fcube*);
	static void extract_interp_linear3D(arma::fcube*, arma::fmat*, arma::uvec*, arma::fcube*);
	static void slice3D(arma::fmat*, arma::fmat*, arma::uvec*, arma::fmat*, float, arma::fcube*, int active = 0, std::string interpolate="nearest");
	static void slice3D(arma::fcube*, arma::fmat*, arma::uvec*, arma::fmat*, float, arma::fcube*, int active = 0, std::string interpolate="nearest");
		
	static void interp_linear3D(arma::fmat*, arma::fmat*, arma::uvec*, arma::fcube*, arma::fcube*);
	static void insert_slice(arma::fcube*, arma::fmat*, arma::fcube*, arma::fcube*);
	static void interp_nearestNeighbor(arma::fmat*, arma::fmat*, arma::uvec*, arma::fcube*, arma::fcube*);
	static void merge3D(arma::fmat*, arma::fmat*, arma::uvec*, arma::fmat*, float, arma::fcube*, arma::fcube*, int active = 0, std::string interpolate="nearest");
	static void merge3D(arma::fcube*, arma::fmat*, arma::fmat*, float, arma::fcube*, arma::fcube*, int active = 0, std::string interpolate="nearest");
		
	static void normalize(arma::fcube*, arma::fcube*);
	
	static void cart2polar(fcube *samplePoints, int detectorWidth, float rhoMin, float rhoMax);
	static void interp_linear2D(fmat* newDP, fcube* samplePoints, fmat* cartDP);

	static arma::fmat badpixmap2goodpixmap(arma::fmat badpixmap);
	
	static double calculateSimilarity(fmat* modelSlice, fmat* dataSlice, fmat* pixmap, string type);

private:
	double mNumber;
};

}

#ifdef __cplusplus
}
#endif

#endif
