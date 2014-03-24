#ifndef CIO_H
#define CIO_H
#include <armadillo>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

arma::fmat hdf5read(std::string,std::string);
void hdf5write(std::string,std::string,arma::fmat);

arma::fmat load_asciiImage(std::string x);
arma::fvec load_asciiEuler(std::string x);

void load_constant(int);
void load_array(int*,int);
void load_array2D(int*,int,int);

void load_constantf(float);
void load_arrayf(float*,int);
void load_array2Df(float*,int,int);

void load_atomType(int*,int);
void load_atomPos(float*,int,int);
void load_xyzInd(int*,int);
void load_fftable(float*,int,int);
void load_qSample(float*,int);

struct Packet {
// particle
	int* atomType;
	float* atomPos;
	int* xyzInd;
	float* ffTable;
	float* qSample;
	int T;
	int N;
	int Q;
// detector
	unsigned* dp;
	double d;				// (m) detector distance
	double pix_width;		// (m)
	double pix_height;		// (m)
	int px;					// number of pixels in x
	int py;					// number of pixels in y
// beam
	double lambda; 							// (m) wavelength
	double focus;						// (m)
	double n_phot;							// number of photons per pulse
	unsigned freeElectrons;					// number of free electrons in the beam
// extra
	int finish;
	unsigned long seed;
	int useGPU;
};

void calculate_dp(Packet*);
int write_HDF5(char *);

class CIO{
	//int b;
public:
	void get_image();
	double get_size();
};

#ifdef __cplusplus
}
#endif

#endif
