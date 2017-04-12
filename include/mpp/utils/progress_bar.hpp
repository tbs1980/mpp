#ifndef MPP_UTILS_PROGRESS_BAR_HPP
#define MPP_UTILS_PROGRESS_BAR_HPP

#include <cstddef>
#include <iostream>
#include <iomanip>      // std::setw
#include <ostream>      // std::flush

namespace mpp { namespace utils {

inline void load_progress_bar(std::size_t x, std::size_t n, std::size_t w = 50)
{
    // http://www.ross.click/2011/02/creating-a-progress-bar-in-c-or-any-other-console-app/
    if(x % std::size_t(100) == 0) {
        return;
    }

    float ratio  =  x/(float) (n-1);
    std::size_t   c      =  ratio * w;

    std::cout << std::setw(3) << (int)(ratio*100) << "% [";

    for (std::size_t x = 0; x < c; x++){
        std::cout << "=";
    }
    for (std::size_t x = c; x < w; x++){
        std::cout << " ";
    }
    std::cout << "]\r" << std::flush;
}


}}

#endif //MPP_UTILS_PROGRESS_BAR_HPP
