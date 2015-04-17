#ifndef TOOLBOX_H
#define TOOLBOX_H
#include <armadillo>
#include <string>
#include <sstream>
#include <iostream>
#include "detector.h"
#include "diffractionVolume.h"
#include "diffractionPattern.h"
#include <boost/algorithm/string.hpp>
#include <boost/mpi.hpp>
#include <boost/program_options.hpp>

namespace opt = boost::program_options;
namespace mpi = boost::mpi;

using namespace diffractionVolume;
using namespace diffractionPattern;

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

	static void displayStatusBar(int numDone, int totalJobs, float* lastPercentDone);
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

	static void extract_interp_linear3D(CDiffrPat* mySlice, fmat *myPoints, \
                                        uvec *pixmap, CDiffrVol* diffrVol);
	static bool isNullWeight(CDiffrVol* diffrVol, int x, int y, int z);
	static void slice3D(CDiffrPat* mySlice, arma::fmat* Rot, \
	                    CDiffrVol* diffrVol, detector::CDetector* det, \
	                    int active = 0, std::string interpolate="nearest");               
	static void interp_linear3D(arma::fmat*, arma::fmat*, arma::uvec*, arma::fcube*, arma::fcube*);
	static void insert_slice(CDiffrPat* mySlice, arma::fmat*, CDiffrVol* diffrVol);
	static void interp_nearestNeighbor(arma::fmat*, arma::fmat*, arma::uvec*, arma::fcube*, arma::fcube*);
	static void merge3D(CDiffrPat* myUpdatedSlice, arma::fmat* R, \
	                    CDiffrVol* diffrVol, \
	                    detector::CDetector* det, int active = 0, \
	                    std::string interpolate="nearest");
	                    	                    		
	//static void normalize(arma::fcube*, arma::fcube*);
	
	static void cart2polar(fcube *samplePoints, int detectorWidth, float rhoMin, float rhoMax);
	static void interp_linear2D(fmat* newDP, fcube* samplePoints, fmat* cartDP);

	static arma::fmat badpixmap2goodpixmap(arma::fmat badpixmap);
	
	static double calculateEuclideanSimilarity(fmat* modelSlice, fmat* dataSlice, fmat* pixmap, string type); // FIXME
	static double calculatePoissonianSimilarity(CDiffrPat* mySlice, CDiffrPat* myDP);
	static double calculateGaussianSimilarity(CDiffrPat* mySlice, CDiffrPat* myDP, float stdDev);

private:
	double mNumber;
};

}

#ifdef __cplusplus
}
#endif

#endif
