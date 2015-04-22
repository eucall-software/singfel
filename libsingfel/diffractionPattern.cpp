#include <iostream>
#include <armadillo>
#include "diffractionPattern.h"

using namespace std;
using namespace arma;
using namespace diffractionPattern;

CDiffrPat::CDiffrPat() {
	px = 1;
	py = 1;
	numPix = 1;
	photonCount.zeros(py,px);
	photonpixmap.zeros(numPix);
}

void CDiffrPat::init(int volDim) {
	px = volDim;
	py = volDim;
	numPix = px * py;
	photonCount.zeros(py,px);
	photonpixmap.zeros(numPix);
}

void CDiffrPat::loadPhotonCount(opt::variables_map vm, int ind) {
	string input = vm["input"].as<string>();
	string output = vm["output"].as<string>();
	string format = vm["format"].as<string>();
	string hdfField = vm["hdfField"].as<string>();
	int volDim = vm["volDim"].as<int>();
	
	init(volDim);
	string filename;
	std::stringstream ss;
	if (format == "S2E") {
		ss << input << "/diffr/diffr_out_" << setfill('0') << setw(7) << ind << ".h5";
		filename = ss.str();
		// Read in diffraction			
		photonCount = hdf5readT<fmat>(filename,hdfField);
	}
	photonpixmap = find( photonCount > 0 );
}

void CDiffrPat::updateExpansionSlice(opt::variables_map vm, fvec* normCondProb, uvec* candidatesInd) {
	string input = vm["input"].as<string>();
	string format = vm["format"].as<string>();
	string hdfField = vm["hdfField"].as<string>();
	int volDim = vm["volDim"].as<int>();
	
	uvec& _candidatesInd = candidatesInd[0];

	int numCandidates = _candidatesInd.n_elem;
	init(volDim);
	CDiffrPat myDP;
	// Load measured diffraction pattern from file
	if (format == "S2E") {
		for (int i = 0; i < numCandidates; i++) {
			// load measured diffraction pattern
			loadPhotonCount(vm, _candidatesInd(i)+1); //loadDPnPixmap(vm, _candidatesInd(i)+1, &myDPnPixmap);
			// calculate weighted image and add to updatedSlice
			calculateWeightedImage(normCondProb->at(i), &myDP);
		}
	} else if (format == "list") { //FIXME
	}
}

void CDiffrPat::loadUpdatedExpansion(opt::variables_map vm, int iter, int sliceInd) {
	string output = vm["output"].as<string>();
	int volDim = vm["volDim"].as<int>();
	init(volDim);
		
	// Get image
	std::stringstream ss;
	string filename;
	ss << output << "/expansion/iter" << iter << "/expansionUpdate_" << setfill('0') << setw(7) << sliceInd << ".dat";
	filename = ss.str();
	photonCount = load_asciiImage(filename);
	photonpixmap = find( photonCount > 0);
}

void CDiffrPat::loadExpansionSlice(opt::variables_map vm, int iter, int sliceInd) {		
	string output = vm["output"].as<string>();
	int volDim = vm["volDim"].as<int>();
	
	init(volDim);
	std::stringstream ss;
	ss << output << "/expansion/iter" << iter << "/expansion_" << setfill('0') << setw(7) << sliceInd << ".dat";
	string filename = ss.str();
	photonCount = load_asciiImage(filename); // load expansion slice
	photonpixmap = find( photonCount > 0);
}

// Save updated expansion slices
void CDiffrPat::saveExpansionUpdate(opt::variables_map vm, int iter, int expansionInd) {
	string output = vm["output"].as<string>();

	string filename;
	std::stringstream ss;
	ss << output << "/expansion/iter" << iter << "/expansionUpdate_" << setfill('0') << setw(7) << expansionInd << ".dat";
	filename = ss.str();
	photonCount.save(filename,raw_ascii);
	// Save photonpixmap?
}

// Save expansion slices
void CDiffrPat::saveExpansionSlice(opt::variables_map vm, int iter, int ind) {
	string output = vm["output"].as<string>();

	string filename;
	std::stringstream ss;
	ss << output << "/expansion/iter" << iter << "/expansion_" << setfill('0') << setw(7) << ind << ".dat";
	filename = ss.str();
	photonCount.save(filename,raw_ascii);
	// Save photonpixmap?
}

void CDiffrPat::calculateWeightedImage(const float weight, CDiffrPat* myDP) {
	uvec photonCountPixmap = myDP->photonpixmap;
	
	// Setup goodpixmap
	uvec::iterator goodBegin = photonCountPixmap.begin();
	uvec::iterator goodEnd = photonCountPixmap.end();
	for(uvec::iterator p=goodBegin; p!=goodEnd; ++p) {
		photonCount(*p) += weight * myDP->photonCount(*p);
	}
	photonpixmap = find(photonCount > 0);
}


