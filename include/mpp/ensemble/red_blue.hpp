#ifndef MPP_ENSEMBLE_RED_BLUE_HPP
#define MPP_ENSEMBLE_RED_BLUE_HPP

#include <exception>
#include <sstream>
#include <string>
#include <random>
#include <type_traits>
#include <cmath>
#include <functional>
#include <cstddef>
#include <iostream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/assert.hpp>
#include "../config.hpp"
#include "../chains/mcmc_chain.hpp"
#include "../utils/progress_bar.hpp"

namespace mpp{ namespace ensemble{

template<class real_scalar_type>
class red_blue_sampler{
public:
    static_assert(
        std::is_floating_point<real_scalar_type>::value,
        "The real scalar is expected to be a floating point type."
    );
    typedef boost::numeric::ublas::vector<real_scalar_type> real_vector_type;
    typedef boost::numeric::ublas::matrix<real_scalar_type> real_matrix_type;
    typedef mpp::chains::mcmc_chain<real_scalar_type> chain_type;
    typedef typename std::function<
        real_scalar_type (real_vector_type const &) > log_post_func_type;
    typedef typename std::function<
        real_vector_type (real_vector_type const &)> grad_log_post_func_type;


    red_blue_sampler(
        log_post_func_type & log_posterior,
        std::size_t const num_dims,
        std::size_t const num_walkers,
        real_scalar_type const scale_a
    ) throw()
    : m_log_posterior(log_posterior)
    , m_num_dims(num_dims)
    , m_num_walkers(num_walkers)
    , m_scale_a(scale_a)
    , m_beta(1)
    , m_acc_rate(0) {
        BOOST_ASSERT_MSG(
            num_dims <= std::size_t(MPP_MAXIMUM_NUMBER_OF_DIMENSIONS),
            "num_dims too big. Please modify the config.hpp and recompile."
        );

        BOOST_ASSERT_MSG(
            num_walkers <= std::size_t(MPP_MAXIMUM_NUMBER_OF_WALKERS),
            "num_dims too big. Please modify the config.hpp and recompile."
        );

        if( m_num_dims == size_t(0) ) {
            std::stringstream msg;
            msg << "The number of dimensions = "
                << m_num_dims
                << " should be greater than zero.";
            throw std::length_error(msg.str());
        }

        if( m_num_walkers % 2 != 0 ) {
            std::stringstream msg;
            msg << "Number of walkers = "
                << m_num_walkers
                << " should be an even number ";
            throw std::length_error(msg.str());
        }

        if( m_num_walkers < 2*m_num_dims ) {
            std::stringstream msg;
            msg << "Maximum number of walkers = "
                << m_num_walkers
                << " should be more than twice the "
                << " number of dimensions.";
            throw std::length_error(msg.str());
        }

        if( m_scale_a <= 1 or m_scale_a > 10 ) {
            std::stringstream msg;
            msg << "Scale parameter a = "
                << m_scale_a
                << " should be a value between 0 and 10.";
            throw std::domain_error(msg.str());
        }
    }

    template<class rng_type>
    chain_type run_sampler(
        size_t const num_samples,
        real_matrix_type const & start_points,
        rng_type & rng
    ) {
        using namespace boost::numeric::ublas;
        using namespace mpp::utils;
        BOOST_ASSERT_MSG(
            start_points.size1() == m_num_walkers,
            "Number of rows should be equal to the number of walkers."
        );
        BOOST_ASSERT_MSG(
            start_points.size2() == m_num_dims,
            "Number of columns should be equal to the number of dimensions."
        );
        BOOST_ASSERT_MSG(
            num_samples <=
                size_t(MPP_MAXIMUM_NUMBER_OF_SAMPLES_PER_RUN_SAMPLER_CALL),
            "num_samples too big. Please modify the config and recompile."
        );
        chain_type rb_chain(num_samples,m_num_dims);
        std::size_t half_walkers = (std::size_t) m_num_walkers/2;
        std::uniform_int_distribution<std::size_t> uni_int_dist(
            0,
            half_walkers - 1
        );
        std::uniform_real_distribution<real_scalar_type> uni_real_dist;
        real_matrix_type ensbl_S(start_points);
        real_vector_type lp_ensbl_S(m_num_walkers);
        for(std::size_t wkr_i = 0; wkr_i < m_num_walkers; ++wkr_i){
            matrix_row<real_matrix_type> X_i(
                ensbl_S,
                wkr_i
            );
            lp_ensbl_S(wkr_i) = m_log_posterior(X_i);
        }

        std::size_t num_accepted(0);
        std::size_t num_rejected(0);
        while( num_accepted < num_samples ) {
            for(std::size_t half_i = 0; half_i < 2; ++ half_i){
                std::size_t compl_i = half_i == 0 ? half_walkers : 0 ;
                std::size_t wkr_start = half_i == 0 ? 0 : half_walkers;
                std::size_t wkr_end = half_i == 0 ?half_walkers :m_num_walkers;

                for(std::size_t wkr_k = wkr_start; wkr_k < wkr_end; ++wkr_k){
                    matrix_row<real_matrix_type> X_j(
                        ensbl_S,
                        compl_i + uni_int_dist(rng)
                    );
                    real_scalar_type u_rand = uni_real_dist(rng);
                    real_scalar_type z_val
                        = (u_rand*(m_scale_a -1.)+1.)
                            *(u_rand*(m_scale_a -1.)+1.)/m_scale_a;
                    matrix_row<real_matrix_type> X_k(
                        ensbl_S,
                        wkr_k
                    );
                    real_scalar_type lp_X_k = lp_ensbl_S(wkr_k);
                    real_vector_type prop_Y = X_j + z_val*(X_k - X_j);
                    real_scalar_type lp_prop_Y = m_log_posterior(prop_Y);
                    real_scalar_type log_ratio_q
                        = (m_num_dims-1)*std::log(z_val) + lp_prop_Y - lp_X_k;
                    real_scalar_type log_uni_r = std::log(uni_real_dist(rng));
                    if(log_uni_r <=  log_ratio_q*m_beta) {
                        X_k = prop_Y;
                        lp_ensbl_S(wkr_k) = lp_prop_Y;
                        if( wkr_k == 0){
                            rb_chain.set_sample(num_accepted,prop_Y,lp_prop_Y);
                            ++num_accepted;
                            load_progress_bar(num_accepted, num_samples);
                        }
                    }
                    else{
                        if( wkr_k == 0){
                            ++num_rejected;
                        }
                    }
                }
            }
        }

        m_acc_rate = real_scalar_type(num_accepted)
            / real_scalar_type(num_accepted + num_rejected);

        return rb_chain;
    }

    inline real_scalar_type acc_rate() const {
        return m_acc_rate;
    }

private:
    log_post_func_type m_log_posterior;
    std::size_t m_num_dims;
    std::size_t m_num_walkers;
    real_scalar_type m_scale_a;
    real_scalar_type m_beta;
    real_scalar_type m_acc_rate;
};

}}

#endif //MPP_ENSEMBLE_RED_BLUE_HPP
