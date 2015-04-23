/*
 * Merge diffraction patterns in a diffraction volume given known angles
 */
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <algorithm>
#include <fstream>
#include <string>
// Armadillo library
#include <armadillo>
// Boost library
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
// SingFEL library
#include "detector.h"
#include "beam.h"
#include "particle.h"
#include "diffraction.h"
#include "toolbox.h"
#include "io.h"

using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

opt::variables_map parse_input(int argc, char* argv[]);
void loadDPnPixmap(opt::variables_map vm, int ind, fcube* myDPnPixmap);
void loadQuaternion(opt::variables_map vm, int ind, fvec* quat);
void saveDiffractionVolume(opt::variables_map vm, fcube* myIntensity, fcube* myWeight);

int main( int argc, char* argv[] ){

    // image.lst and euler.lst are not neccessarily in the same order! So can not use like this. Perhaps use hdf5 to combine the two.
    
    opt::variables_map vm = parse_input(argc, argv);
	string input = vm["input"].as<string>();
	string output = vm["output"].as<string>();
	string beamFile = vm["beamFile"].as<string>();
	string geomFile = vm["geomFile"].as<string>();
	string format = vm["format"].as<string>();
	string hdfField = vm["hdfField"].as<string>();
	int numImages = vm["numImages"].as<int>();
	int volDim = vm["volDim"].as<int>();
	
	// Set up diffraction geometry
	CDetector det = CDetector();
	CBeam beam = CBeam();
	beam.readBeamFile(beamFile);
	det.readGeomFile(geomFile);
	det.init_dp(&beam);

	// Optionally display resolution
	CDiffraction::displayResolution(&det, &beam);

	std::stringstream ss;
  	string filename;
  		
  	fcube myDPnPixmap;
  	fvec quat(4);
  	fmat myR;
  	myR.zeros(3,3);
  	//float psi,theta,phi;

	CDiffrVol diffrVol = CDiffrVol(volDim);
	//fcube myIntensity, myWeight;
	//myIntensity.zeros(volDim,volDim,volDim);
	//myWeight.zeros(volDim,volDim,volDim);

	int active = 0;
	string interpolate = "linear";
	float lastPercentDone = 0;
  	// ########### Save diffraction volume ##############
  	cout << "Merging diffraction volume..." << endl;
  	for (int i = 0; i < numImages; i++) {
  		loadDPnPixmap(vm, i+1, &myDPnPixmap);
  		loadQuaternion(vm, i+1, &quat);
  		myR = CToolbox::quaternion2rot3D(quat);
       	CToolbox::merge3D(&myDPnPixmap, &myR, &diffrVol, &det, active, interpolate);
       	// Display status
		CToolbox::displayStatusBar(i+1,numImages,&lastPercentDone);
  	}
  	diffrVol.normalize(); //CToolbox::normalize(&myIntensity,&myWeight);
  		
  	// ########### Save diffraction volume ##############
  	cout << "Saving diffraction volume..." << endl;
  	diffrVol.saveDiffractionVolume(vm);

  	return 0;
}

void loadQuaternion(opt::variables_map vm, int ind, fvec* quat) {
	string input = vm["input"].as<string>();
	string format = vm["format"].as<string>();
	fvec& _quat = quat[0];
	
	string filename;
	std::stringstream ss;
	if (format == "S2E") {
		ss << input << "/diffr/diffr_out_" << setfill('0') << setw(7) << ind << ".h5";
		filename = ss.str();
		_quat = hdf5readT<fvec>(filename,"/data/angle");
	} else {
		// TODO: Add default behaviour
	}
}

void loadDPnPixmap(opt::variables_map vm, int ind, fcube* myDPnPixmap) {
	string input = vm["input"].as<string>();
	string output = vm["output"].as<string>();
	string format = vm["format"].as<string>();
	string hdfField = vm["hdfField"].as<string>();
	int volDim = vm["volDim"].as<int>();
	
	myDPnPixmap->zeros(volDim,volDim,2);
	string filename;
	std::stringstream ss;
	if (format == "S2E") {
		ss << input << "/diffr/diffr_out_" << setfill('0') << setw(7) << ind << ".h5";
		filename = ss.str();
		// Read in diffraction			
		myDPnPixmap->slice(0) = hdf5readT<fmat>(filename,hdfField);
	} else {
		ss << input << "/diffr/diffr_out_" << setfill('0') << setw(7) << ind << ".dat";
		filename = ss.str();
		myDPnPixmap->slice(0) = load_asciiImage(filename);
	}
	ss.str("");
	ss << output << "/badpixelmap.dat";
	filename = ss.str();
	fmat pixmap = load_asciiImage(filename); // load badpixmap
	myDPnPixmap->slice(1) = CToolbox::badpixmap2goodpixmap(pixmap); // goodpixmap
}

opt::variables_map parse_input( int argc, char* argv[] ) {

    // Constructing an options describing variable and giving it a
    // textual description "All options"
    opt::options_description desc("All options");

    // When we are adding options, first parameter is a name
    // to be used in command line. Second parameter is a type
    // of that option, wrapped in value<> class. Third parameter
    // must be a short description of that option
    desc.add_options()
        ("input,i", opt::value<std::string>(), "Input directory for finding /diffr")
        ("eulerList", opt::value<std::string>(), "Input directory for finding list of Euler angles")
        ("quaternionList", opt::value<std::string>(), "Input directory for finding list of quaternions")
        ("rotationList", opt::value<std::string>(), "Input directory for finding list of rotation matrices")
        ("beamFile,b", opt::value<string>(), "Beam file defining X-ray beam")
        ("geomFile,g", opt::value<string>(), "Geometry file defining diffraction geometry")
        ("numImages", opt::value<int>(), "Number of measured diffraction patterns")
        ("volDim", opt::value<int>(), "Number of pixel along one dimension")
        ("output,o", opt::value<string>(), "Output directory for saving /vol")
        ("format", opt::value<string>(), "Defines file format to use")
        ("hdfField", opt::value<string>()->default_value("/data/data"), "Data field to use for reconstruction")
        ("help", "produce help message")
    ;

    // Variable to store our command line arguments
    opt::variables_map vm;

    // Parsing and storing arguments
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
	opt::notify(vm);

	// Print input arguments
    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(0);
    }

	if (!vm.count("input")) { //TODO: print all parameters
    	cout << "NOTICE: input field is required" << endl;
    	exit(0);
    }
	if (!vm.count("beamFile")) {
    	cout << "NOTICE: beamFile field is required" << endl;
    	exit(0);
    }
    if (!vm.count("geomFile")) {
    	cout << "NOTICE: geomFile field is required" << endl;
    	exit(0);
    }
	return vm;
} // end of parse_input
