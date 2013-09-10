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
	static arma::frowvec xyzInd;			// TEMPORARY
	static arma::urowvec formFactorList;
public:
	CParticle();
	static void load_atomType(string);
	//static void set_atomType(arma::irowvec, int);
	static void set_atomType(Packet*);
	static void load_atomPos(string);
	static void set_atomPos(Packet*);
	static void set_atomPos(arma::fmat*);	
	static arma::fmat get_atomPos();	
	static void load_xyzInd(string);
	static void set_xyzInd(Packet*);
	static void load_ffTable(string);
	static void set_ffTable(Packet*);
	static void load_qSample(string);
	static void set_qSample(Packet*);
	static void set_param(Packet*);
protected:

private:
};

}

#ifdef __cplusplus
}
#endif

#endif
