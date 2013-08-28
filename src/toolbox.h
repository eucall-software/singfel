#ifndef TOOLBOX_H
#define TOOLBOX_H
#include <armadillo>

#ifdef __cplusplus
extern "C" {
#endif

namespace toolbox{

class CToolbox{
public:
	static arma::mat mag(arma::cube);
	static arma::fmat mag(arma::fcube);
	static arma::umat convert_to_poisson(arma::fmat);
};

}

#ifdef __cplusplus
}
#endif

#endif
