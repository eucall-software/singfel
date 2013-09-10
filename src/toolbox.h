#ifndef TOOLBOX_H
#define TOOLBOX_H
#include <armadillo>
#include <string>
#include <sstream>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

namespace toolbox{

class CToolbox{
public:
    CToolbox(std::string n) : name(n) {}
    std::string name;
	static arma::mat mag(arma::cube);
	static arma::fmat mag(arma::fcube);
	static arma::umat convert_to_poisson(arma::fmat);
	static arma::fmat quaternion2rot3D(arma::vec,int);
	static void quaternion2AngleAxis(arma::vec,float&,arma::vec&);
	static arma::fmat angleAxis2rot3D(arma::vec,float);
};

}

#ifdef __cplusplus
}
#endif

#endif
