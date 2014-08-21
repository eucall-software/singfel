#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <armadillo>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>       /* time */
#include <math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include "detector.h"
#include "beam.h"
#include "particle.h"
#include "diffraction.h"
#include "toolbox.h"
#include "diffraction.cuh"

//#include <cuda.h>
#include <algorithm>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

//#define ARMA_NO_DEBUG

//#define USE_CUDA 0
#define USE_CHUNK 0

int main( int argc, char* argv[] ){

    string beamFile;
    string geomFile;
    int gpu = 0; // not used
    string output;
    
	string inputDir;
	string outputDir;
	int USE_CUDA;
	
	int sliceInterval;
	int numSlices;
	int pmiStartID;
	int pmiEndID;
	int numDP;

    // Let's parse input
    for (int n = 1; n < argc; n++) {
    cout << argv [ n ] << endl;
        if(boost::algorithm::iequals(argv[ n ], "--sliceInterval")) {
            sliceInterval = atoi(argv[ n+2 ]);
        } else if (boost::algorithm::iequals(argv[ n ], "--input_dir")) {
            inputDir = argv[ n+2 ];
        } else if (boost::algorithm::iequals(argv[ n ], "--output_dir")) {
            outputDir = argv[ n+2 ];
        }else if (boost::algorithm::iequals(argv[ n ], "-b")) {
            beamFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-g")) {
            geomFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "--gpu")) {
            gpu = 1;
        } else if (boost::algorithm::iequals(argv[ n ], "--numSlices")) {
            numSlices = atoi(argv[ n+2 ]);
        } else if (boost::algorithm::iequals(argv[ n ], "--pmiStartID")) {
            pmiStartID = atoi(argv[ n+2 ]);
        } else if (boost::algorithm::iequals(argv[ n ], "--pmiEndID")) {
            pmiEndID = atoi(argv[ n+2 ]);
        } else if (boost::algorithm::iequals(argv[ n ], "--numDP")) {
            numDP = atoi(argv[ n+2 ]);
        }
    }

	string outputName;
	/* initialize random seed: */
	srand (pmiStartID);//srand (time(NULL)+pmiStartID);
  	
	wall_clock timer, timer1, timer2, timer3, timerMaster;

	timerMaster.tic();
	
	fmat rot3D(3,3);
	fvec u(3);
	fvec quaternion(4);	
	// Rotate single particle
	u = randu<fvec>(3); // uniform random distribution in the [0,1] interval
	// generate uniform random quaternion on SO(3)
	quaternion << sqrt(1-u(0)) * sin(2*datum::pi*u(1)) << sqrt(1-u(0)) * cos(2*datum::pi*u(1))
			   << sqrt(u(0)) * sin(2*datum::pi*u(2)) << sqrt(u(0)) * cos(2*datum::pi*u(2));
	// quaternion to rotation matrix			
	rot3D = CToolbox::quaternion2rot3D(quaternion);
	
	/****** Beam ******/
	// Let's read in our beam file
	double photon_energy = 0;
	double focus_radius = 0;
	double fluence = 0;
	string line;
	ifstream myFile(beamFile.c_str());
	while (getline(myFile, line)) {
		if (line.compare(0,1,"#") && line.compare(0,1,";") && line.length() > 0) {
		    // line now contains a valid input
		    typedef boost::tokenizer<boost::char_separator<char> > Tok;
		    boost::char_separator<char> sep(" ="); // default constructed
		    Tok tok(line, sep);
		    for(Tok::iterator tok_iter = tok.begin(); tok_iter != tok.end(); ++tok_iter){
		        if ( boost::algorithm::iequals(*tok_iter,"beam/photon_energy") ) {            
		            string temp = *++tok_iter;
		            photon_energy = atof(temp.c_str()); // photon energy to wavelength
		            break;
		        } else if ( boost::algorithm::iequals(*tok_iter,"beam/fluence") ) {            
		            string temp = *++tok_iter;
		            fluence = atof(temp.c_str()); // number of photons per pulse
		            break;
		        } else if ( boost::algorithm::iequals(*tok_iter,"beam/radius") ) {            
		            string temp = *++tok_iter;
		            focus_radius = atof(temp.c_str()); // focus radius
		            break;
		        }
		    }
		}
	}
	CBeam beam = CBeam();
	string filename;
	stringstream sstm;
	sstm << inputDir << "/pmi_out_" << setfill('0') << setw(7) << pmiStartID << ".h5";
	filename = sstm.str();
//cout << filename << endl;
//cout << "get vector" << endl;
//	fmat myTemp = hdf5readT<fmat>(filename,"/data/snp_001/r");
//cout << "get photon energy" << endl;
//	fvec myPhotonEnergy = hdf5readT<fvec>(filename,"/history/parent/detail/params/photonEnergy");
//cout << myPhotonEnergy << endl;
//	beam.set_photonsPerPulse(myPhotonEnergy[0]);
					
	beam.set_photon_energy(photon_energy);
	beam.set_focus(focus_radius*2); // radius to diameter
	//beam.set_photonsPerPulse(fluence);
	//beam.set_photonsPerPulsePerArea();
cout << "DONE!!!!!" << endl;

	/****** Detector ******/
	double d = 0;					// (m) detector distance
	double pix_width = 0;			// (m)
	int px_in = 0;                  // number of pixel along x
	string badpixmap = ""; // this information should go into the detector class
	// Parse the geom file
	ifstream myGeomFile(geomFile.c_str());
	while (getline(myGeomFile, line)) {
		if (line.compare(0,1,"#") && line.compare(0,1,";") && line.length() > 0) {
			// line now contains a valid input 
            typedef boost::tokenizer<boost::char_separator<char> > Tok;
            boost::char_separator<char> sep(" ="); // default constructed
            Tok tok(line, sep);
            for(Tok::iterator tok_iter = tok.begin(); tok_iter != tok.end(); ++tok_iter){
                if ( boost::algorithm::iequals(*tok_iter,"geom/d") ) {            
                    string temp = *++tok_iter;
                    d = atof(temp.c_str());
                    break;
                } else if ( boost::algorithm::iequals(*tok_iter,"geom/pix_width") ) {            
                    string temp = *++tok_iter;
                    pix_width = atof(temp.c_str());
                    break;
                } else if ( boost::algorithm::iequals(*tok_iter,"geom/px") ) {            
                    string temp = *++tok_iter;
                    px_in = atof(temp.c_str());
                    break;
                } else if ( boost::algorithm::iequals(*tok_iter,"geom/badpixmap") ) {            
                    string temp = *++tok_iter;
                    cout << temp << endl;
                    badpixmap = temp;
                    break;
                }
            }
        }
    }
	double pix_height = pix_width;		// (m)
	const int px = px_in;				// number of pixels in x
	const int py = px;					// number of pixels in y
	double cx = ((double) px-1)/2;		// this can be user defined
	double cy = ((double) py-1)/2;		// this can be user defined

	CDetector det = CDetector();
	det.set_detector_dist(d);	
	det.set_pix_width(pix_width);	
	det.set_pix_height(pix_height);
	det.set_numPix(py,px);
	det.set_center_x(cx);
	det.set_center_y(cy);
	//det.set_pixelMap(badpixmap);
	//det.init_dp(&beam);

	double theta = atan((px/2*pix_height)/d);
	double qmax = 2/beam.get_wavelength()*sin(theta/2);
	double dmin = 1/(2*qmax);
	cout << "max q to the edge: " << qmax << " m^-1" << endl;
	cout << "Half period resolution: " << dmin*1e10 << " Angstroms" << endl;
	
	umat detector_counts;		
	detector_counts.zeros(py,px);

	int numTrajectories = pmiEndID - pmiStartID + 1;
	int patternID;
	for (int pmiID = pmiStartID; pmiID <= pmiEndID; pmiID++) {
		//TO DO: Check pmiID exists in the workflow
		for (int dp = 0; dp < numDP; dp++) {
			patternID = (pmiID-1)*numDP+dp+1;
			cout << "patternID: " << patternID << endl;	
			double total_phot = 0;
			for (int timeSlice = sliceInterval; timeSlice <= numSlices; timeSlice+=sliceInterval) {
				string filename;
				string datasetname;

				stringstream sstm;
				sstm << inputDir << "/pmi_out_" << setfill('0') << setw(7) << pmiID << ".h5";
				filename = sstm.str();
	
				stringstream sstm0;
				sstm0 << "/data/snp_" << setfill('0') << setw(3) << timeSlice;
				datasetname = sstm0.str();
		
				// Particle //
				CParticle particle = CParticle();

cout << "Load particle" << endl; 
	
				particle.load_atomType(filename,datasetname+"/T"); 	// rowvec atomType
				particle.load_atomPos(filename,datasetname+"/r");		// mat pos
				particle.load_ionList(filename,datasetname+"/xyz");		// rowvec ion list
				particle.load_ffTable(filename,datasetname+"/ff");	// mat ffTable (atomType x qSample)
				particle.load_qSample(filename,datasetname+"/Q");	// rowvec q vector sin(theta)/lambda

cout << "Done particle" << endl; 
cout << "Atom pos: " << particle.get_atomPos() << endl;

				// Rotate atom positions
				fmat myPos = particle.get_atomPos();
				myPos = myPos * trans(rot3D);
				particle.set_atomPos(&myPos);
		
				// Beam //
				double n_phot = 0;
				for (int i = 0; i < sliceInterval; i++) {
					string datasetname;
					stringstream sstm0;
					sstm0 << "/data/snp_" << setfill('0') << setw(3) << timeSlice-i;
					datasetname = sstm0.str();
cout << "Read Nph" << endl; 					
					vec myNph = hdf5readT<vec>(filename,datasetname+"/Nph");
cout << "Done Nph" << endl; 
					beam.set_photonsPerPulse(myNph[0]);
					n_phot += beam.get_photonsPerPulse();	// number of photons per pulse
				}
				total_phot += n_phot;
				cout << "n_phot/total_phot: "<< n_phot << "/" << total_phot << endl;
				beam.set_photonsPerPulse(n_phot);
				beam.set_photonsPerPulsePerArea();

				det.init_dp(&beam);
				CDiffraction::calculate_atomicFactor(&particle,&det); // get f_hkl

			#ifdef COMPILE_WITH_CUDA
			if (!USE_CHUNK) {
				//cout<< "USE_CUDA && NO_CHUNK" << endl;

				CDiffraction::get_atomicFormFactorList(&particle,&det);		

				fmat F_hkl_sq(py,px);		
			 	float* F_mem = F_hkl_sq.memptr();
				float* f_mem = CDiffraction::f_hkl_list.memptr();
				float* q_mem = det.q_xyz.memptr();
				float* p_mem = particle.atomPos.memptr();
				cuda_structureFactor(F_mem, f_mem, q_mem, p_mem, det.numPix, particle.numAtoms);
		
				fmat detector_intensity = F_hkl_sq % det.solidAngle % det.thomson * beam.get_photonsPerPulsePerArea();

				// Add to incoherent sum of diffraction patterns
				detector_counts += CToolbox::convert_to_poisson(detector_intensity);
			
				if (timeSlice == numSlices) {
					stringstream sstm3;
					sstm3 << outputDir << "/diffr_out_" << setfill('0') << setw(7) 
					<< patternID << ".h5";
					outputName = sstm3.str();
					
					std::cout << file_size(filename) << endl;
					copy_file("/data/S2E/data/simulation_test/diffr/out.h5", outputName);
/*					
					//int appendDataset = 0;
					int createSubgroup = 0;
					int success = hdf5writeT(outputName,"data","","/data/data", detector_intensity,createSubgroup);
					//success = hdf5writeT("filename1.h5","data","/data/data", detector_counts);
					fmat angle = conv_to< fmat >::from(quaternion);
					angle = angle.t();
					//appendDataset = 1;
					success = hdf5writeT(outputName,"data","","/data/angle", angle,createSubgroup);
					createSubgroup = 1;
					fmat dist(1,1);
					dist(0) = det.get_detector_dist();
					success = hdf5writeT(outputName,"params","params/geom","/params/geom/detectorDist", dist,createSubgroup);
					createSubgroup = 0;
					fmat pixelWidth(1,1);
					pixelWidth(0) = det.get_pix_width();
					success = hdf5writeT(outputName,"params","params/geom","/params/geom/pixelWidth", pixelWidth,createSubgroup);
					fmat pixelHeight(1,1);
					pixelHeight(0) = det.get_pix_height();
					success = hdf5writeT(outputName,"params","params/geom","/params/geom/pixelHeight", pixelHeight,createSubgroup);
					fmat mask = ones<fmat>(px_in,px_in);
					success = hdf5writeT(outputName,"params","params/geom","/params/geom/mask", mask,createSubgroup);			
					createSubgroup = 1;
					fmat photonEnergy(1,1);
					photonEnergy(0) = beam.get_photon_energy();
					success = hdf5writeT(outputName,"params","params/beam","/params/beam/photonEnergy", photonEnergy,createSubgroup);
					createSubgroup = 0;
					fmat photons(1,1);
					photons(0) = beam.get_photonsPerPulse();
					success = hdf5writeT(outputName,"params","params/beam","/params/beam/photons", photons,createSubgroup);			
					createSubgroup = 0;
					fmat focusArea(1,1);
					focusArea(0) = beam.get_focus_area();
					success = hdf5writeT(outputName,"params","params/beam","/params/beam/focusArea", focusArea,createSubgroup);
*/					
				}
			} else if (USE_CHUNK) {
				//cout<< "USE_CUDA && USE_CHUNK" << endl;
				int max_chunkSize = 100;
				int chunkSize = 0;

				fmat F_hkl_sq(py,px); // F_hkl_sq: py x px
		 
				float* f_mem = CDiffraction::f_hkl.memptr(); // f_hkl: py x px x numAtomTypes
				float* q_mem = det.q_xyz.memptr(); // q_xyz: py x px x 3

				fmat pad_real;
				fmat pad_imag;
				fmat sumDr;
				sumDr.zeros(py*px,1);
				fmat sumDi;
				sumDi.zeros(py*px,1);
		
				int first_ind = 0;
				int last_ind = 0;
				while (first_ind < particle.numAtoms) {			 
					last_ind = min((last_ind + max_chunkSize),particle.numAtoms);

					chunkSize = last_ind-first_ind;

					pad_real.zeros(py*px,chunkSize);
				 	float* pad_real_mem = pad_real.memptr();

					pad_imag.zeros(py*px,chunkSize);
				 	float* pad_imag_mem = pad_imag.memptr();
				 	
					// xyzInd & pos are chunked
					// particle.xyzInd // 1 x chunk
					// particle.pos // chunk x 3
					irowvec xyzInd_sub = particle.xyzInd.subvec( first_ind,last_ind-1 );
					int* i_mem = xyzInd_sub.memptr();	
					fmat pos_sub = particle.atomPos( span(first_ind,last_ind-1), span::all );
					float* p_mem = pos_sub.memptr();

					cuda_structureFactorChunkParallel(pad_real_mem, pad_imag_mem, f_mem, q_mem, i_mem, p_mem, particle.numAtomTypes, det.numPix, chunkSize);

					sumDr += sum(pad_real,1);
					sumDi += sum(pad_imag,1);
			
					first_ind += max_chunkSize;
				}
		
				sumDr.reshape(py,px);
				sumDi.reshape(py,px);
				F_hkl_sq = sumDr % sumDr + sumDi % sumDi;

				fmat detector_intensity = F_hkl_sq % det.solidAngle % det.thomson * beam.get_photonsPerPulsePerArea();
				detector_counts += CToolbox::convert_to_poisson(detector_intensity);
		
				if (timeSlice == numSlices) {
					stringstream sstm3;
					sstm3 << outputDir << "/diffr_out_" << setfill('0') << setw(7) 
					<< patternID << ".h5";
					outputName = sstm3.str();
/*					
					//int appendDataset = 0;
					int createSubgroup = 0;
					int success = hdf5writeT(outputName,"data","","/data/data", detector_intensity,createSubgroup);
					//success = hdf5writeT("filename1.h5","data","/data/data", detector_counts);
					fmat angle = conv_to< fmat >::from(quaternion);
					angle = angle.t();
					//appendDataset = 1;
					success = hdf5writeT(outputName,"data","","/data/angle", angle,createSubgroup);
					createSubgroup = 1;
					fmat dist(1,1);
					dist(0) = det.get_detector_dist();
					success = hdf5writeT(outputName,"params","params/geom","/params/geom/detectorDist", dist,createSubgroup);
					createSubgroup = 0;
					fmat pixelWidth(1,1);
					pixelWidth(0) = det.get_pix_width();
					success = hdf5writeT(outputName,"params","params/geom","/params/geom/pixelWidth", pixelWidth,createSubgroup);
					fmat pixelHeight(1,1);
					pixelHeight(0) = det.get_pix_height();
					success = hdf5writeT(outputName,"params","params/geom","/params/geom/pixelHeight", pixelHeight,createSubgroup);
					fmat mask = ones<fmat>(px_in,px_in);
					success = hdf5writeT(outputName,"params","params/geom","/params/geom/mask", mask,createSubgroup);			
					createSubgroup = 1;
					fmat photonEnergy(1,1);
					photonEnergy(0) = beam.get_photon_energy();
					success = hdf5writeT(outputName,"params","params/beam","/params/beam/photonEnergy", photonEnergy,createSubgroup);
					createSubgroup = 0;
					fmat photons(1,1);
					photons(0) = beam.get_photonsPerPulse();
					success = hdf5writeT(outputName,"params","params/beam","/params/beam/photons", photons,createSubgroup);			
					createSubgroup = 0;
					fmat focusArea(1,1);
					focusArea(0) = beam.get_focus_area();
					success = hdf5writeT(outputName,"params","params/beam","/params/beam/focusArea", focusArea,createSubgroup);
*/					
				}
			}
			#else
			//cout<< "USE_CPU" << endl;
				timer.tic();
				CDiffraction::get_atomicFormFactorList(&particle,&det);

				fmat F_hkl_sq;
				F_hkl_sq = CDiffraction::calculate_intensity(&particle,&det);
		
				fmat detector_intensity = F_hkl_sq % det.solidAngle % det.thomson * beam.get_photonsPerPulse();
				detector_counts += CToolbox::convert_to_poisson(detector_intensity);
			
				if (timeSlice == numSlices) {
					stringstream sstm3;
					sstm3 << outputDir << "/diffr_out_" << setfill('0') << setw(7) 
					<< patternID << ".h5";
					outputName = sstm3.str();
					
					int createSubgroup = 0;
					int success = hdf5writeVector(outputName,"data","","/data/data", detector_intensity,createSubgroup);
					//success = hdf5writeT("filename1.h5","data","/data/data", detector_counts);
					
					
					createSubgroup = 0;
					//appendDataset = 1;
					fvec angle = quaternion;
					success = hdf5writeVector(outputName,"data","","/data/angle", angle,createSubgroup);
					
					createSubgroup = 1;
					double dist = det.get_detector_dist();
					success = hdf5writeScalar(outputName,"params","params/geom","/params/geom/detectorDist", dist,createSubgroup);

					createSubgroup = 0;
					fmat tt(3,2);
					int counter = 0;
					for (int i = 0; i < 2; i++)
					for (int j = 0; j < 3; j++)
						tt(j,i) = 0.1*counter++;
					tt.print("tt:");
					success = hdf5writeVector(outputName,"params","params/geom","/params/geom/mat", tt,createSubgroup);
					
					fmat myT = hdf5readT<fmat>(outputName,"/params/geom/mat");
					myT.print("myT:");
					//double myDist = hdf5readT<double>(outputName,"/params/geom/detectorDist");
					//myDist.print("myDist:");
/*			
					createSubgroup = 0;
					cube ttt(3,2,4);
					counter = 0;
					for (int k = 0; k < 4; k++)
					for (int i = 0; i < 2; i++)
					for (int j = 0; j < 3; j++)
						ttt(j,i,k) = counter++;
					ttt.print("ttt:");
					success = hdf5writeCube(outputName,"params","params/geom","/params/geom/cube", ttt,createSubgroup);
*/
					
					createSubgroup = 0;
					double pixelWidth = det.get_pix_width();
					success = hdf5writeScalar(outputName,"params","params/geom","/params/geom/pixelWidth", pixelWidth,createSubgroup);
					double pixelHeight = det.get_pix_height();
					success = hdf5writeScalar(outputName,"params","params/geom","/params/geom/pixelHeight", pixelHeight,createSubgroup);
					fmat mask = ones<fmat>(px_in,px_in);
					success = hdf5writeVector(outputName,"params","params/geom","/params/geom/mask", mask,createSubgroup);
					createSubgroup = 1;
					double photonEnergy = beam.get_photon_energy();
					success = hdf5writeScalar(outputName,"params","params/beam","/params/beam/photonEnergy", photonEnergy,createSubgroup);
					createSubgroup = 0;
					double photons = total_phot;
					success = hdf5writeScalar(outputName,"params","params/beam","/params/beam/photons", photons,createSubgroup);			
					createSubgroup = 0;
					double focusArea = beam.get_focus_area();
					success = hdf5writeScalar(outputName,"params","params/beam","/params/beam/focusArea", focusArea,createSubgroup);

				}
			#endif
			}
		}
	}
	cout << "Total time: " <<timerMaster.toc()<<" seconds."<<endl;
  	return 0;
}

