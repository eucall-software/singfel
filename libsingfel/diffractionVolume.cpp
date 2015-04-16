#include "diffractionVolume.h"

using namespace std;
using namespace arma;
using namespace diffractionVolume;

fcube CDiffrVol::intensity;
fcube CDiffrVol::weight;
int CDiffrVol::volDim;

CDiffrVol::CDiffrVol() {
	volDim = 1;
	intensity(volDim,volDim,volDim);
	weight(volDim,volDim,volDim);
}

CDiffrVol::CDiffrVol(int px) {
cout << "got here" << endl;
	volDim = px;
cout << "got here1" << endl;
	intensity.zeros(volDim,volDim,volDim);
cout << "got here2" << endl;
	weight.zeros(volDim,volDim,volDim);
}

void CDiffrVol::initVol() {
	intensity.zeros(volDim,volDim,volDim);
	weight.zeros(volDim,volDim,volDim);
}

void CDiffrVol::initVol(int px) {
	volDim = px;
	intensity.zeros(volDim,volDim,volDim);
	weight.zeros(volDim,volDim,volDim);
}

void CDiffrVol::loadInitVol(opt::variables_map vm) {
	string initialVolume = vm["initialVolume"].as<string>();
	int volDim = vm["volDim"].as<int>();
	
	initVol(volDim);
	std::stringstream ss;
	string filename;
	for (int i = 0; i < volDim; i++) {
		ss.str("");
		ss << initialVolume << "/vol_" << setfill('0') << setw(7) << i << ".dat";
		filename = ss.str();
		intensity.slice(i) = load_asciiImage(filename);
		ss.str("");
		ss << initialVolume << "/volWeight_" << setfill('0') << setw(7) << i << ".dat";
		filename = ss.str();
		weight.slice(i) = load_asciiImage(filename);
	}
}

// Normalize the diffraction volume
void CDiffrVol::normalize() {
	uvec ind = find(weight > 0);
	//cout << "ind: " << ind << endl;
	intensity.elem(ind) = intensity.elem(ind) / weight.elem(ind); // Use sparse indexing
}

// Random intensity
void CDiffrVol::randVol() {
	intensity.randu(volDim,volDim,volDim);
	weight.ones(volDim,volDim,volDim);
}


