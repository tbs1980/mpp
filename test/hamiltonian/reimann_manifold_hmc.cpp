#define BOOST_TEST_MODULE rmhmc sampler
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/multi_array.hpp>
#include <cassert>

BOOST_AUTO_TEST_CASE(rmhmc) {
    typedef boost::multi_array<double, 3> array_t;
    typedef array_t::index index;
    array_t A(boost::extents[2][3][4]);
    int values = 0;
    for(index i=0; i != 2; ++i){
        for(index j=0; j != 3; ++j){
            for(index k=0; k !=4; ++k){
                A[i][j][k] = values++;
            }
        }
    }

    typedef typename array_t::index_range range_t;
    array_t::index_gen indices;
    array_t::array_view<2>::type myview
        = A[ indices[ range_t(0,2)][range_t(0,3)][0] ];

    for(index i=0; i != 2; ++i){
        for(index j=0; j != 3; ++j){
            assert(myview[i][j] == A[i][j][0]);
        }
    }
}
