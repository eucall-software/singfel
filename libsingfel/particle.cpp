#include <iostream>
#include "particle.h"

using namespace std;
using namespace particle;
using namespace arma;

int CParticle::numAtoms;				// no. of atoms
int CParticle::numQSamples;				// no. of q samples
int CParticle::numAtomTypes;			// no. of atom types
irowvec CParticle::atomType;				// atom type list
fmat CParticle::atomPos;						// atom position
fmat CParticle::ffTable;					// form factor table (atomType x qSample)
frowvec CParticle::qSample;				// q vector sin(theta)/lambda

irowvec CParticle::ionList;
irowvec CParticle::xyzInd;				// TEMPORARY

urowvec CParticle::formFactorList;

CParticle::CParticle (){
	//cout << "init particle" << endl;
}

void CParticle::load_atomType(string x){
	imat B;	// handles rowvec and colvec
	B.load(x,raw_ascii);
	if ( B.n_cols == 1 || B.n_rows == 1 ){ // rowvec or colvec
		atomType = vectorise(B,1);
	}else{
		cout << "Error: unexpected atom type dimension" << endl;
		exit(EXIT_FAILURE);
	}
	numAtomTypes = atomType.n_elem;
}
/*
void CParticle::set_atomType(irowvec x, int n){
	cout << "n: " << n << endl;
	atomType = x;
	atomType.print("set_atomType: ");
	numAtomTypes = n;
}
*/
void CParticle::set_atomType(Packet *x){
	//cout << "n: " << x->T << endl;
	irowvec temp(x->atomType, x->T);
	atomType = temp;
	//CParticle::atomType.print("set_atomType: ");
	numAtomTypes = x->T;
}

void CParticle::load_atomPos(string x){
	atomPos.load(x,raw_ascii);
	numAtoms = atomPos.n_rows;
	formFactorList = zeros<urowvec>(1,numAtoms);
//	CParticle::atomPos.print("set_atomPos: ");
}

void CParticle::set_atomPos(Packet *x){
	fmat temp(x->atomPos, 3, x->N);
	atomPos = trans(temp);
	numAtoms = x->N;
//	CParticle::atomPos.print("set_atomPos: ");
}

void CParticle::set_atomPos(fmat *x){
//cout << "x:" << x[0] << endl;
	atomPos = x[0];
}

fmat CParticle::get_atomPos(){
	return atomPos;
}

void CParticle::load_ionList(string x){
	imat B;	// handles rowvec and colvec
	B.load(x,raw_ascii);
	if ( B.n_cols == 1 || B.n_rows == 1 ){ // rowvec or colvec
		ionList = vectorise(B,1);
	}else{
		cout << "Error: unexpected ion list dimension" << endl;
		exit(EXIT_FAILURE);
	}	
}

void CParticle::load_xyzInd(string x){
	xyzInd.load(x,raw_ascii);
	//CParticle::xyzInd.print("set_xyzInd: ");
}

void CParticle::set_xyzInd(Packet *x){
	irowvec temp(x->xyzInd, x->N);
	xyzInd = temp;
//	CParticle::xyzInd.print("set_xyzInd: ");
}

void CParticle::set_xyzInd(irowvec *ionList){
	int numAtoms = ionList->n_elem;
	//cout << numAtoms << endl;
	xyzInd = zeros<irowvec>(numAtoms);
	for (int i = 0; i < numAtoms; i++) {
		//cout << "iList: " << ionList[i] << endl;
		xyzInd(i) = conv_to< int >::from(find(ionList[0](i) == atomType, 0, "first"));
	}	
	//CParticle::xyzInd.print("set_xyzInd: ");
}

void CParticle::load_ffTable(string x){
	ffTable.load(x,raw_ascii);
//	CParticle::ffTable.print("set_ffTable: ");
}

void CParticle::set_ffTable(Packet *x){
	fmat temp(x->ffTable, x->Q, x->T);
	ffTable = trans(temp);
	numQSamples = x->Q;
//	CParticle::ffTable.print("set_ffTable: ");	
}

void CParticle::load_qSample(string x){
	fmat B;	// handles rowvec and colvec
	B.load(x,raw_ascii);
	if ( B.n_cols == 1 || B.n_rows == 1 ){ // rowvec or colvec
		qSample = vectorise(B,1);
	}else{
		cout << "Error: unexpected qSample dimension" << endl;
		exit(EXIT_FAILURE);
	}
	numQSamples = qSample.n_elem;
//	CParticle::qSample.print("set_qSample: ");
}

void CParticle::set_qSample(Packet *x){
	frowvec temp(x->qSample, x->Q);
	qSample = temp;
//	CParticle::qSample.print("set_qSample: ");
}

void CParticle::set_param(Packet *x){
	set_atomType(x);
	set_atomPos(x);
	set_xyzInd(x);
	set_ffTable(x);
	set_qSample(x);
}
