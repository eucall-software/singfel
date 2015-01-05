#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include "detector.h"
#include "beam.h"
#include "particle.h"
#include "diffraction.h"
#include "toolbox.h"
#include "diffraction.cuh"

#ifdef COMPILE_WITH_CXX11
	#define ARMA_DONT_USE_CXX11
#endif
#include <armadillo>

//#include <cuda.h>
#include <algorithm>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace arma;
using namespace detector;
using namespace beam;
using namespace particle;
using namespace diffraction;
using namespace toolbox;

//#define ARMA_NO_DEBUG

#define USE_CUDA 0
#define USE_CHUNK 0

int main( int argc, char* argv[] ){

    //string imageList;
    //string quaternionList;
    string atomTypeFile;
    string posFile;
    string xyzIndFile;
    string ffTableFile;
    string qSampleFile;
    
    string beamFile;
    string geomFile;
    int numImages = 0;
    string output;
    // Let's parse input
    // image.lst and euler.lst are not neccessarily in the same order! So can not use like this. 		// Perhaps use hdf5 to combine the two.
    for (int n = 1; n < argc; n++) {
    cout << argv [ n ] << endl; 
        if(boost::algorithm::iequals(argv[ n ], "-a")) {
            atomTypeFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-p")) {
            posFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-x")) {
            xyzIndFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-f")) {
            ffTableFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-q")) {
            qSampleFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-b")) {
            beamFile = argv[ n+1 ];
        } else if (boost::algorithm::iequals(argv[ n ], "-g")) {
            geomFile = argv[ n+1 ];   
        } else if (boost::algorithm::iequals(argv[ n ], "--num_images")) {
            numImages = atoi(argv[ n+2 ]);
        } else if (boost::algorithm::iequals(argv[ n ], "--output_dir")) {
            output = argv[ n+2 ];
        }
    }
    /*
	cout << "The name used to start the program: " << argv[ 0 ] << "\nArguments are:\n";
    for (int n = 1; n < argc; n++)
    	cout << setw( 2 ) << n << ": " << argv[ n ] << '\n';

	int numPatterns = atoi(argv[1]);
	//float q0 = atof(argv[1]);
	//float q1 = atof(argv[2]);
	//float q2 = atof(argv[3]);
	//float q3 = atof(argv[4]);
	
	wall_clock timer, timer1, timer2, timer3, timerMaster;

timerMaster.tic();
	*/
	// Particle
	CParticle particle = CParticle();
	particle.load_atomType(atomTypeFile.c_str()); 	// rowvec atomType 
	particle.load_atomPos(posFile.c_str());		// mat pos
	particle.load_xyzInd(xyzIndFile.c_str());		// rowvec xyzInd (temporary)
	particle.load_ffTable(ffTableFile.c_str());	// mat ffTable (atomType x qSample)
	particle.load_qSample(qSampleFile.c_str());	// rowvec q vector sin(theta)/lambda
	

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
		    cout << line << endl;
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
	beam.set_photon_energy(photon_energy);
	//CBeam beam = CBeam();
	//beam.set_wavelength(lambda);
	beam.set_focus(focus_radius*2); // radius to diameter
	beam.set_photonsPerPulse(fluence);
	beam.set_photonsPerPulsePerArea();

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
            //cout << line << endl; 
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
	det.set_pixelMap(badpixmap);
	det.set_center_x(cx);
	det.set_center_y(cy);
cout << "Enter1" << endl;
//timer.tic();
	det.init_dp(&beam);
//cout<<"Init dp: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
//timer.tic();
cout << "Enter2" << endl;
	CDiffraction::calculate_atomicFactor(&particle,&det); // get f_hkl
//cout<<"Calculate factor: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
cout << "Enter3" << endl;
//cout << "q_xyz rows: " << det.q_xyz.n_rows<<endl;
//cout << "q_xyz cols: " << det.q_xyz.n_cols<<endl;
//cout << "q_xyz slices: " << det.q_xyz.n_slices<<endl;

	double theta = atan((px/2*pix_height)/d);
	double qmax = 2/beam.get_wavelength()*sin(theta/2);
	double dmin = 1/(2*qmax);
	cout << "max q to the edge: " << qmax << " m^-1" << endl;
	cout << "Half period resolution: " << dmin << " m" << endl;

	if(!USE_CUDA) {
		CDiffraction::get_atomicFormFactorList(&particle,&det);
		
		//timer.tic();
		fmat F_hkl_sq;
		string outputName;


		fmat myPos = particle.get_atomPos();
		CParticle rotatedParticle = CParticle();
		//rotatedParticle = particle;	
		
		rotatedParticle.load_atomType(atomTypeFile.c_str()); 	// rowvec atomType 
		rotatedParticle.load_atomPos(posFile.c_str());		// mat pos
		rotatedParticle.load_xyzInd(xyzIndFile.c_str());		// rowvec xyzInd (temporary)
		rotatedParticle.load_ffTable(ffTableFile.c_str());	// mat ffTable (atomType x qSample)
		rotatedParticle.load_qSample(qSampleFile.c_str());	// rowvec q vector sin(theta)/lambda
		
		fmat rot3D(3,3);
		fvec u(3);
		fvec quaternion(4);	
		//int i = 0;
		for (int i = 0; i < numImages; i++) {
			// Rotate single particle			
			u = randu<fvec>(3); // uniform random distribution in the [0,1] interval
			// generate uniform random quaternion on SO(3)
			//quaternion << 1 << 0 << 0 << 0;
			
			if (i==0){
				quaternion = CToolbox::euler2quaternion(0,0,0*datum::pi/180);//
				//quaternion << 1 << 0 << 0 << 0;
				//rot3D = CToolbox::euler2rot3D(0,0,0*datum::pi/180);
			}/*else if (i==1){
				quaternion = CToolbox::euler2quaternion(0,0,15*datum::pi/180);//
				//quaternion << 0.7071 << 0 << 0 << -0.7071;
				//rot3D = CToolbox::euler2rot3D(0,0,15*datum::pi/180);
			}else if (i==2){
				quaternion = CToolbox::euler2quaternion(0,0,30*datum::pi/180);//
				//quaternion << 0.7071 << 0 << 0 << -0.7071;
				//rot3D = CToolbox::euler2rot3D(0,0,30*datum::pi/180);
			}else if (i==3){
				quaternion = CToolbox::euler2quaternion(0,0,45*datum::pi/180);//quaternion << 0. << 0 << 0 << 1;	
				//rot3D = CToolbox::euler2rot3D(0,0,45*datum::pi/180);
			}else if (i==4){
				quaternion = CToolbox::euler2quaternion(0,0,90*datum::pi/180);
				//rot3D = CToolbox::euler2rot3D(0,0,60*datum::pi/180);
			}else if (i==5){
				quaternion = CToolbox::euler2quaternion(0,0,120*datum::pi/180);
				//rot3D = CToolbox::euler2rot3D(0,0,75*datum::pi/180);
			}else if (i==6){
				quaternion = CToolbox::euler2quaternion(0,0,180*datum::pi/180);
				//rot3D = CToolbox::euler2rot3D(0,0,180*datum::pi/180);
			} else {
			*/
				quaternion << sqrt(1-u(0)) * sin(2*datum::pi*u(1)) << sqrt(1-u(0)) * cos(2*datum::pi*u(1))
					   << sqrt(u(0)) * sin(2*datum::pi*u(2)) << sqrt(u(0)) * cos(2*datum::pi*u(2));
			//}
			//quaternion << 0.7071 << 0 << 0 << 0.7071;
			//quaternion << q0 << q1 << q2 << q3;
			//quaternion = CToolbox::euler2quaternion(0,0,u(0)*360*datum::pi/180);
			//quaternion = CToolbox::euler2quaternion(0,0,0*datum::pi/180);
			//quaternion << sqrt(1-u(0)) * sin(2*datum::pi*u(1)) << sqrt(1-u(0)) * cos(2*datum::pi*u(1))
			//		   << sqrt(u(0)) * sin(2*datum::pi*u(2)) << sqrt(u(0)) * cos(2*datum::pi*u(2));
			//cout << "quaternion:" << quaternion << endl;
			rot3D = CToolbox::quaternion2rot3D(quaternion);
			
			//cout << rot3D << endl;

			//cout << myPos << endl;
			
			//if (i == 0) {
			cout << "*********************** i: " << i << endl;
				//cout << "quaternion: " << quaternion << endl;
				cout << "rot3D:" << rot3D << endl;
				cout << "myPos(yxz):" << myPos.row(0) << endl;
				//cout << "myPos:" << myPos.row(1) << endl;
			//}
			
			//myPos = myPos * trans(rot3D);	// This seems to be an active rotation
			fmat myRotPos = myPos * rot3D;
			//myPos = myPos * rot3D;	// This seems to be an passive rotation
			frowvec temp(3);
			temp << 1 << 0 << 0;
			cout << "test rot: " << temp * rot3D << endl;
			
			//myPos = particle.get_atomPos();
			cout << "myPos(yxz):" << myPos.row(0) << endl;
			
			//myPos = trans(rot3D) * trans(myPos);// passively rotate atom positions
			//myPos = trans(myPos);

			//if (i == 0) {
				cout << "myRotPos(yxz):" << myRotPos.row(0) << endl;
				//cout << "myRotPos:" << myPos.row(1) << endl;
			//}
					
				
			rotatedParticle.set_atomPos(&myRotPos);
		cout << "myPos(yxz):" << myPos.row(0) << endl;
			//cout << "rotated particle: " << rotatedParticle.get_atomPos() << endl;

			F_hkl_sq = CDiffraction::calculate_intensity(&rotatedParticle,&det);
			//cout<<"Calculate F_hkl: Elapsed time is "<<timer.toc()<<" seconds."<<endl; // 14.25s
			//F_hkl_sq.save("../F_hkl_sq.dat",raw_ascii);
			//timer.tic();
		cout << "myPos(yxz):" << myPos.row(0) << endl;	
			fmat detector_intensity = F_hkl_sq % det.solidAngle % det.thomson * beam.get_photonsPerPulsePerArea(); //2.105e30
			umat detector_counts = CToolbox::convert_to_poisson(detector_intensity);	
		cout << "myPos(yxz):" << myPos.row(0) << endl;	
			cout << "detector_counts(1,2): " << detector_counts(1,2) << endl;
			cout << det.q_xyz(0,0,0) << endl;
			cout << det.q_xyz(0,0,1) << endl;
			cout << det.q_xyz(0,0,2) << endl;
				
			//cout<<"Calculate dp: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
			stringstream sstm2;
			sstm2 << output << "detector_intensity_" << setfill('0') << setw(6) << i << ".dat";
			outputName = sstm2.str();
			detector_intensity.save(outputName,raw_ascii);
			//det.solidAngle.save("../solidAngle.dat",raw_ascii);
			//timer.tic();
			stringstream sstm;
			sstm << output << "diffraction_" << setfill('0') << setw(6) << i << ".dat";
			outputName = sstm.str();
			detector_counts.save(outputName,raw_ascii);
			stringstream sstm1;
			sstm1 << output << "quaternion_" << setfill('0') << setw(6) << i << ".dat";
			outputName = sstm1.str();
			quaternion.save(outputName,raw_ascii);			
			//cout<<"Save image: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
		}
	}

//cout << "Total time: " <<timerMaster.toc()<<" seconds."<<endl;
  	return 0;
}

