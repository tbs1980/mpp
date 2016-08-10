#define BOOST_TEST_MODULE rmhmc sampler
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/multi_array.hpp>

BOOST_AUTO_TEST_CASE(rmhmc) {
    typedef boost::multi_array<double, 3> array_t;
    typedef array_t::index index;
    array_t A(boost::extents[3][4][2]);
    int values = 0;
    for(index i=0; i != 3; ++i){
        for(index j=0; j != 4; ++j){
            for(index k=0; k !=2; ++k){
                A[i][j][k] = values++;
            }
        }
    }

}

