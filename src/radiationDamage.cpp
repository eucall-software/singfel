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

	cout << "The name used to start the program: " << argv[ 0 ] << "\nArguments are:\n";
    for (int n = 1; n < argc; n++)
    	cout << setw( 2 ) << n << ": " << argv[ n ] << '\n';

	int numPatterns = atoi(argv[1]);
	string inputDir = argv[2];
	string outputDir = argv[3];
	int patternID = atoi(argv[4]);
	
	string outputName;
	/* initialize random seed: */
  	//srand (time(NULL));
  	
	wall_clock timer, timer1, timer2, timer3, timerMaster;

timerMaster.tic();

	//CParticle rotatedParticle = CParticle();
		
	fmat rot3D(3,3);
	vec u(3);
	vec quaternion(4);	
	// Rotate single particle		
	//u = randu<vec>(3); // uniform random distribution in the [0,1] interval
	u << 0.0 << 0.4 << 0.0; // good angle
	//u << 0.80 << 0.66 << 0.4; // good angle
	// generate uniform random quaternion on SO(3)
	quaternion << sqrt(1-u(0)) * sin(2*datum::pi*u(1)) << sqrt(1-u(0)) * cos(2*datum::pi*u(1))
			   << sqrt(u(0)) * sin(2*datum::pi*u(2)) << sqrt(u(0)) * cos(2*datum::pi*u(2));
			
			//cout << quaternion << endl;
			
	rot3D = CToolbox::quaternion2rot3D(quaternion, 1);
			
			//cout << rot3D << endl;

	// Beam //
	double lambda = 2.5e-10; 							// (m) wavelength
	double focus = 250e-9;						// (m)
	CBeam beam = CBeam();
	beam.set_wavelength(lambda);
	beam.set_focus(focus);
	
	// Detector //
	double d = 21.3e-3;//16.9e-3;//19.1e-3;//21.9e-3;//23.1e-3;//44.9e-3;//51e-3;//33e-3;//44.1e-3;//50.4e-3;				// (m) detector distance
	double pix_width = 200e-6;		// (m)
	double pix_height = pix_width;		// (m)
	const int px = 95;//55;//141;//71;//49;//109;					// number of pixels in x
	const int py = px;					// number of pixels in y
	double cx = ((double) px-1)/2;			// this can be user defined
	double cy = ((double) py-1)/2;			// this can be user defined

	CDetector det = CDetector();
	det.set_detector_dist(d);	
	det.set_pix_width(pix_width);	
	det.set_pix_height(pix_height);
	det.set_numPix(py,px);
	det.set_center_x(cx);
	det.set_center_y(cy);
	
	umat detector_counts;		
	detector_counts.zeros(py,px);

	int sliceInterval = 5;
	for (int timeSlice = 5; timeSlice <= numPatterns; timeSlice+=sliceInterval) {
	cout << timeSlice << endl;
	
	stringstream sstm0;
	sstm0 << "/snp_" << setfill('0') << setw(3) << timeSlice;
	string filename;
	filename = sstm0.str();
			
	// Particle //
	CParticle particle = CParticle();
	particle.load_atomType(inputDir+filename+"_T.dat"); 	// rowvec atomType 
	particle.load_atomPos(inputDir+filename+"_r.dat");		// mat pos
	particle.load_ionList(inputDir+filename+"_xyz.dat");		// rowvec ion list
	particle.set_xyzInd(&particle.ionList);		// rowvec xyzInd (temporary)
	particle.load_ffTable(inputDir+filename+"_ff.dat");	// mat ffTable (atomType x qSample)
	particle.load_qSample(inputDir+filename+"_Q.dat");	// rowvec q vector sin(theta)/lambda

	//rotatedParticle = particle;
	fmat myPos = particle.get_atomPos();
	myPos = myPos * trans(rot3D);	// rotate atom positions
	particle.set_atomPos(&myPos);
			
	// Beam //
	mat phot;
	double n_phot = 0;
	sstm0.str("");
	for (int i = 0; i < sliceInterval; i++) {
		sstm0 << "/snp_" << setfill('0') << setw(3) << timeSlice-i;
		phot.load(inputDir+filename+"_Nph.dat");
		n_phot += conv_to< double >::from(phot);							// number of photons per pulse
	}
	cout << n_phot << endl;

	beam.set_photonsPerPulse(n_phot);
	beam.set_photonsPerPulsePerArea();

//timer.tic();
	det.init_dp(&beam);
//cout<<"Init dp: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
//timer.tic();
	CDiffraction::calculate_atomicFactor(&particle,&det); // get f_hkl
//cout<<"Calculate factor: Elapsed time is "<<timer.toc()<<" seconds."<<endl;

//cout << "q_xyz rows: " << det.q_xyz.n_rows<<endl;
//cout << "q_xyz cols: " << det.q_xyz.n_cols<<endl;
//cout << "q_xyz slices: " << det.q_xyz.n_slices<<endl;

	double theta = atan((px/2*pix_height)/d);
	double qmax = 2/lambda*sin(theta/2);
	double dmin = 1/(2*qmax);
	cout << "max q to the edge: " << qmax << " m^-1" << endl;
	cout << "Half period resolution: " << dmin << " m" << endl;

#ifdef COMPILE_WITH_CUDA
	if (USE_CUDA && !USE_CHUNK) {
cout<< "USE_CUDA" << endl;
timer.tic();
		CDiffraction::get_atomicFormFactorList(&particle,&det);		
cout<<"Make list: Elapsed time is "<<timer.toc()<<" seconds."<<endl; // 1.00s

timer.tic();
		fmat F_hkl_sq(py,px);
cout<<"1: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
timer.tic();		
	 	float* F_mem = F_hkl_sq.memptr();
cout<<"2: Elapsed time is "<<timer.toc()<<" seconds."<<endl;		 	
timer.tic();
		//fcube f_hkl_list = conv_to<fcube>::from(CDiffraction::f_hkl_list); // py x px x N
cout<<"2: Elapsed time is "<<timer.toc()<<" seconds."<<endl;	
timer.tic();		
		//fcube q_xyz = conv_to<fcube>::from(det.q_xyz); // py x px x 3
cout<<"2: Elapsed time is "<<timer.toc()<<" seconds."<<endl;	
timer.tic();		
		//fmat pos = conv_to<fmat>::from(particle.atomPos);// Nx3
cout<<"2: Elapsed time is "<<timer.toc()<<" seconds."<<endl;			
//pos.print("pos: ");
timer.tic();
		float* f_mem = CDiffraction::f_hkl_list.memptr();
		float* q_mem = det.q_xyz.memptr();
		float* p_mem = particle.atomPos.memptr();
		cuda_structureFactor(F_mem, f_mem, q_mem, p_mem, det.numPix, particle.numAtoms);
cout<<"3: Elapsed time is "<<timer.toc()<<" seconds."<<endl;			
//cout<<"Calculate F_hkl: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
//		F_hkl_sq.save("../F_hkl_sq_cuda.dat",raw_ascii);
//timer.tic();
timer.tic();
		fmat detector_intensity = F_hkl_sq % det.solidAngle % det.thomson * beam.phi_in;
cout<<"4: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
timer.tic();	
//cout<<"phi_in: "<<beam.phi_in<<endl;
		detector_counts += CToolbox::convert_to_poisson(detector_intensity);
cout<<"5: Elapsed time is "<<timer.toc()<<" seconds."<<endl;			
		if (timeSlice == numPatterns) {
			stringstream sstm;
			sstm << outputDir << "/diffraction_" << setfill('0') << setw(6) << patternID << ".dat";
			outputName = sstm.str();
			detector_counts.save(outputName,raw_ascii);
			stringstream sstm1;
			sstm1 << outputDir << "/quaternion_" << setfill('0') << setw(6) << patternID << ".dat";
			outputName = sstm1.str();
			quaternion.save(outputName,raw_ascii);		
		}	
		
//cout<<"Calculate dp: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
//		det.solidAngle.save("../solidAngle_cuda.dat",raw_ascii);
//		det.thomson.save("../thomson_cuda.dat",raw_ascii);
//timer.tic();
//		detector_intensity.save("../diffraction_intensity_cuda.dat",raw_ascii);
//		detector_counts.save("../diffraction_cuda.dat",raw_ascii);
//cout<<"Save image: Elapsed time is "<<timer.toc()<<" seconds."<<endl;	
	} else if (USE_CUDA && USE_CHUNK) {
cout<< "USE_CHUNK" << endl;
		int max_chunkSize = 100;
		int chunkSize = 0;

		fmat F_hkl_sq(py,px); // F_hkl_sq: py x px
		//F_hkl_sq.zeros(py,px); 

		//fcube f_hkl = conv_to<fcube>::from(CDiffraction::f_hkl); // f_hkl: py x px x numAtomTypes
		float* f_mem = CDiffraction::f_hkl.memptr();
		//fcube q_xyz = conv_to<fcube>::from(det.q_xyz); // q_xyz: py x px x 3
		float* q_mem = det.q_xyz.memptr();

		//irowvec xyzInd = conv_to<irowvec>::from(particle.xyzInd); // xyzInd: 1 x numAtom
		//fmat pos = conv_to<fmat>::from(particle.atomPos); // pos: numAtom x 3
	
		fmat pad_real;
		fmat pad_imag;
		fmat sumDr;
		sumDr.zeros(py*px,1);
		fmat sumDi;
		sumDi.zeros(py*px,1);
		
		int first_ind = 0;
		int last_ind = 0;
		while (first_ind < particle.numAtoms) {	
//cout << "(last_ind + max_chunkSize): " << (last_ind + max_chunkSize) << endl;
//cout << "particle.numAtoms: " << particle.numAtoms << endl;		 
			last_ind = min((last_ind + max_chunkSize),particle.numAtoms);
//cout << "last_ind: " << last_ind << endl;
			chunkSize = last_ind-first_ind;

			pad_real.zeros(py*px,chunkSize);
		 	float* pad_real_mem = pad_real.memptr();

			pad_imag.zeros(py*px,chunkSize);
		 	float* pad_imag_mem = pad_imag.memptr();
		 	
			// xyzInd & pos are chunked
			//particle.xyzInd // 1 x chunk
//cout << "xyzInd length: " << xyzInd.n_cols << endl;
//cout << "first_ind: " << first_ind << endl;
//cout << "last_ind-1: " << last_ind-1 << endl;
			irowvec xyzInd_sub = particle.xyzInd.subvec( first_ind,last_ind-1 );
//cout << "hello" << endl;
			int* i_mem = xyzInd_sub.memptr();	
			//particle.pos // chunk x 3
//cout << "pos length: " << pos.n_rows << endl;
			fmat pos_sub = particle.atomPos( span(first_ind,last_ind-1), span::all );
			float* p_mem = pos_sub.memptr();
//pos_sub.print("pos_sub: ");
//xyzInd_sub.print("xyzInd_sub: ");
//cout << "first_ind: " << first_ind << endl;
//cout << "chunkSize: " << chunkSize << endl;

//timer.tic();
			cuda_structureFactorChunkParallel(pad_real_mem, pad_imag_mem, f_mem, q_mem, i_mem, p_mem, particle.numAtomTypes, det.numPix, chunkSize);
//cout<<"Chunk: Elapsed time is "<<timer.toc()<<" seconds."<<endl;

			sumDr += sum(pad_real,1);
			sumDi += sum(pad_imag,1);
			
			first_ind += max_chunkSize;
		}
//timer.tic();
		
		sumDr.reshape(py,px);
		sumDi.reshape(py,px);
		F_hkl_sq = sumDr % sumDr + sumDi % sumDi;
//cout<<"Calculate F_hkl: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
//		F_hkl_sq.save("../F_hkl_sq_cudaChunk.dat",raw_ascii);
//timer.tic();
		fmat detector_intensity = F_hkl_sq % det.solidAngle % det.thomson * beam.phi_in;
		detector_counts += CToolbox::convert_to_poisson(detector_intensity);
		
		if (timeSlice == numPatterns) {
			stringstream sstm;
			sstm << outputDir << "/diffraction_" << setfill('0') << setw(6) << patternID << ".dat";
			outputName = sstm.str();
			detector_counts.save(outputName,raw_ascii);
			stringstream sstm1;
			sstm1 << outputDir << "/quaternion_" << setfill('0') << setw(6) << patternID << ".dat";
			outputName = sstm1.str();
			quaternion.save(outputName,raw_ascii);		
		}	
//cout<<"Calculate dp: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
//		det.solidAngle.save("../solidAngle_cudaChunk.dat",raw_ascii);
//		det.thomson.save("../thomson_cudaChunk.dat",raw_ascii);
//timer.tic();
//		detector_counts.save("../diffraction_cudaChunk.dat",raw_ascii);
//cout<<"Save image: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
	} 
#endif
	if(!USE_CUDA) {
		timer.tic();
		CDiffraction::get_atomicFormFactorList(&particle,&det);
		cout<<"No CUDA! Make list: Elapsed time is "<<timer.toc()<<" seconds."<<endl; // 1.00s

		//timer.tic();
		fmat F_hkl_sq;

		//cout << rotatedParticle.get_atomPos() << endl;
timer.tic();
		F_hkl_sq = CDiffraction::calculate_intensity(&particle,&det);
cout<<"1: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
		//cout<<"Calculate F_hkl: Elapsed time is "<<timer.toc()<<" seconds."<<endl; // 14.25s
		//F_hkl_sq.save("../F_hkl_sq.dat",raw_ascii);
timer.tic();
		fmat detector_intensity = F_hkl_sq % det.solidAngle % det.thomson * beam.phi_in; //2.105e30
cout<<"2: Elapsed time is "<<timer.toc()<<" seconds."<<endl;		
		detector_intensity.save("../detector_intensity.dat",raw_ascii);
timer.tic();
		detector_counts += CToolbox::convert_to_poisson(detector_intensity);
cout<<"3: Elapsed time is "<<timer.toc()<<" seconds."<<endl;				
		//cout<<"Calculate dp: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
		//det.solidAngle.save("../solidAngle.dat",raw_ascii);
		//timer.tic();
			
		if (timeSlice == numPatterns) {
			stringstream sstm;
			sstm << outputDir << "/diffraction_" << setfill('0') << setw(6) << patternID << ".dat";
			outputName = sstm.str();
			detector_counts.save(outputName,raw_ascii);
			stringstream sstm1;
			sstm1 << outputDir << "/quaternion_" << setfill('0') << setw(6) << patternID << ".dat";
			outputName = sstm1.str();
			quaternion.save(outputName,raw_ascii);		
		}	
		//cout<<"Save image: Elapsed time is "<<timer.toc()<<" seconds."<<endl;
		
	}

cout << "Total time: " <<timerMaster.toc()<<" seconds."<<endl;

	}
  	return 0;
}

