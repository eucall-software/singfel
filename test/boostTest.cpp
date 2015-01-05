// http://www.alittlemadness.com/2009/03/31/c-unit-testing-with-boosttest/
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Suites
#include <boost/test/unit_test.hpp>
#include "toolbox.h"

#include <armadillo>

using namespace arma;
using namespace std;
using namespace toolbox;

struct myTestObject
{
    int m;
    fvec quaternion;
 
    myTestObject() : m(2), quaternion(4)
    {
        BOOST_TEST_MESSAGE("setup mass");
        
        quaternion << 1 << 0 << 0 << 0;
        
    }
 
    ~myTestObject()
    {
        BOOST_TEST_MESSAGE("teardown mass");
    }
};

int add(int i, int j) {
	auto a = i;
	//std::vector<int> b = {0,1,2,3};
	//for (int x : b) {
	//	cout << x << endl;
	//}
    return i + j + a;
}

int rotationConversion(fvec quaternion) {
	float tol = 1e-6;

	//static_assert(__cplusplus > 199711L, "Program requires C++11 capable compiler");
	auto b = add(5,6);
	cout << b << endl;

	//fvec quaternion(4);
	fmat myR;
	fvec euler;
	fvec newQuat;
	
	// Test no rotation
	//quaternion << 1 << 0 << 0 << 0;
	myR = CToolbox::quaternion2rot3D(quaternion);
	euler = CToolbox::quaternion2euler(quaternion);
  	newQuat = CToolbox::euler2quaternion(euler(0),euler(1),euler(2));

	if (abs(sum(quaternion - newQuat)) > tol ) {
		cout << "ERROR!!!" << endl;
		return 1;
	}
	return 0;
}

int indexMatrix() {
	fmat A;
	A << 1 << 2 << endr
	<< 3 << 4 << endr;
	uvec ind;
	ind << 0 << 3 << endr;
	fvec B = A(ind);
	cout << A << endl;
	cout << B << endl;
    return sum(B);
}

BOOST_FIXTURE_TEST_SUITE(Maths, myTestObject)

BOOST_AUTO_TEST_CASE(universeInOrder)
{
    BOOST_CHECK(add(m, 2) == 6);
    BOOST_CHECK(rotationConversion(quaternion) == 0);
    BOOST_CHECK(indexMatrix() == 5);
}

BOOST_AUTO_TEST_SUITE_END()
