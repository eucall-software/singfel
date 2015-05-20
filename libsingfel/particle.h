#ifndef PARTICLE_H
#define PARTICLE_H
#include <armadillo>
#include <string>
#include <iostream>
#include "io.h"

#ifdef __cplusplus
extern "C" {
#endif

using namespace std;

namespace particle{

class CParticle{
public:
	static int numAtoms;				// no. of atoms
	static int numQSamples;				// no. of q samples
	static int numAtomTypes;			// no. of atom types
	static arma::irowvec atomType;		// atom type list
	static arma::fmat atomPos;				// atom position
	static arma::fmat ffTable;			// form factor table (atomType x qSample)
	static arma::frowvec qSample;		// q vector sin(theta)/lambda
	static arma::fvec orientation;	// orientation of particle in quaternion
	static arma::irowvec ionList;			// TEMPORARY	
	static arma::irowvec xyzInd;			// TEMPORARY
	static arma::urowvec formFactorList;
	// Compton properties
	static int numComptonQSamples;			// no. of Compton q samples
	static arma::frowvec comptonQSample;	// Compton: q vector sin(theta)/lambda
	static arma::frowvec sBound;			// Compton: static strucutre factor S(q)
	static arma::frowvec nFree;				// Compton: number of free electrons
public:
	CParticle();
	static void load_atomType(string filename, string datasetname); // load hdf5
	static void load_atomType(string); // load ascii
	//static void set_atomType(arma::irowvec, int);
	static void set_atomType(Packet*); // load Packet structure
	static void load_atomPos(string filename, string datasetname); // load hdf5
	static void load_atomPos(string);
	static void set_atomPos(Packet*);
	static void set_atomPos(arma::fmat*);	
	static arma::fmat get_atomPos();
	static int get_numAtoms();

	static void load_ionList(string filename, string datasetname); // load hdf5
	static void load_ionList(string);	
		
	static void load_xyzInd(string);
	static void set_xyzInd(Packet*);
	
	static void set_xyzInd(arma::irowvec*);
		
	static void load_ffTable(string filename, string datasetname); // load hdf5
	static void load_ffTable(string);
	//static void set_ffTable(Packet*);
	static void load_qSample(string filename, string datasetname); // load hdf5
	static void load_qSample(string);
	static void load_particleOrientation(string filename, string datasetname); // load hdf5
	static arma::fvec get_particleOrientation();
	static void set_qSample(Packet*);
	static void set_param(Packet*);

	static int get_numComptonQSamples();
	static void load_compton_qSample(string filename, string datasetname); // load hdf5
	static void load_compton_sBound(string filename, string datasetname); // load hdf5
	static void load_compton_nFree(string filename, string datasetname); // load hdf5			
protected:

private:
};

}

#ifdef __cplusplus
}
#endif

#endif
