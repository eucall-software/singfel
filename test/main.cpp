#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE PulseTest
#include <boost/test/unit_test.hpp>
#include <exception>
#include <unistd.h>

int add(int i, int j)
{
    return i + j;
}

BOOST_AUTO_TEST_CASE(checkFailure)
{
    BOOST_CHECK(add(2, 2) == 5);
}

