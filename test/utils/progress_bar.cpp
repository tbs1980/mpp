#define BOOST_TEST_MODULE progress bar
#define BOOST_TEST_DYN_LINK

#include <cstddef>
#include <boost/test/unit_test.hpp>
#include <mpp/utils/progress_bar.hpp>

BOOST_AUTO_TEST_CASE(progress_bar) {
    std::size_t const max_iter = 1000;
    for(std::size_t i = 0; i < max_iter; ++i){
        mpp::utils::loadbar(i,max_iter);
    }
}
