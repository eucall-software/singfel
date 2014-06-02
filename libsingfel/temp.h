#ifndef TEMP_H
#define TEMP_H
#include <armadillo>
#include <string>
#include <sstream>
#include <iostream>
#include "detector.h"
#include <boost/algorithm/string.hpp>

//#ifdef __cplusplus
//extern "C" {
//#endif

namespace temp{

//class CTemp{

template <class type> type add(type a, type b)
{
    return a + b;
}

//};

}

//#ifdef __cplusplus
//}
//#endif

#endif
