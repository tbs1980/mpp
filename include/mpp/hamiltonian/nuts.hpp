#ifndef MPP_HAMILTONIAN_NUTS_HPP
#define MPP_HAMILTONIAN_NUTS_HPP

#include <exception>
#include <sstream>
#include <string>
#include <random>
#include <type_traits>
#include <cmath>
#include <functional>
#include <cstddef>
#include <boost/assert.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "../config.hpp"
#include "../chains/mcmc_chain.hpp"
#include "classic_hamiltonian.hpp"
#include "kinetic_energy_multivar_normal.hpp"

namespace mpp { namespace hamiltonian {

template<class real_scalar_t>
class nut_sampler {
public:
    typedef boost::numeric::ublas::vector<real_scalar_t> real_vector_t;
    typedef mpp::chains::mcmc_chain<real_scalar_t> chain_t;
    typedef typename std::function<
        real_scalar_t (real_vector_t const &) > log_post_func_t;
    typedef typename std::function<
        real_vector_t (real_vector_t const &)> grad_log_post_func_t;
    typedef std::normal_distribution<real_scalar_t> normal_dist_t;
    typedef std::exponential_distribution<real_scalar_t> exp_dist_t;
    typedef std::uniform_real_distribution<real_scalar_t> uni_real_dist_t;

    nut_sampler(
        log_post_func_t & log_posterior,
        grad_log_post_func_t & grad_log_posterior,
        size_t const num_dims,
        real_scalar_t const delta
    )
    : m_log_posterior(log_posterior)
    , m_grad_log_posterior(grad_log_posterior)
    , m_num_dims(num_dims)
    , m_delta(delta) {

        if( m_num_dims == size_t(0) ) {
            std::stringstream msg;
            msg << "The number of dimensions = "
                << m_num_dims
                << " should be greater than zero.";
            throw std::length_error(msg.str());
        }
        BOOST_ASSERT_MSG(
            m_num_dims <= size_t(MPP_MAXIMUM_NUMBER_OF_DIMENSIONS),
            "num_dims too big. Please modify the config.hpp and recompile."
        );

    }

    template<class rng_t>
    chain_t run_sampler(
        std::size_t const num_samples,
        real_vector_t const & theta_0,
        rng_t & rng
    ) {
        BOOST_ASSERT_MSG(
            num_samples <=
                size_t(MPP_MAXIMUM_NUMBER_OF_SAMPLES_PER_RUN_SAMPLER_CALL),
            "num_samples too big. Please modify the config and recompile."
        );
        if( theta_0.size() !=  m_num_dims){
            std::stringstream msg;
            msg << "The number of dimensions = "
                << m_num_dims
                << " is not equal to the length of theta_0 = "
                << theta_0.size();
            throw std::length_error(msg.str());
        }

        chain_t hmc_chain(num_samples,m_num_dims);
        real_scalar_t epsilon = find_reasonable_epsilon(
            m_log_posterior,
            m_grad_log_posterior,
            theta_0,
            rng
        );
        real_scalar_t const gamma = 0.5;
        std::size_t t_0 = 10;
        real_scalar_t kappa = 0.75;
        real_scalar_t mu = std::log(10*epsilon);
        real_scalar_t epsilon_bar = 1;
        real_scalar_t H_bar = 0;
        std::size_t const M = num_samples;
        std::size_t const M_adapt = 0.5*num_samples;
        real_scalar_t log_p = m_log_posterior(theta_0);
        real_vector_t grad_0 = m_grad_log_posterior(theta_0);
        std::size_t count_m=1;
        hmc_chain.set_sample(count_m-1,theta_0,log_p);

        normal_dist_t nrm_dist(0.,1.);
        exp_dist_t exp_dist(1.);
        uni_real_dist_t uni_real_dist(0.,1.);
        for(count_m = 2; count_m <= M+M_adapt; ++count_m){
            real_vector_t r_0(m_num_dims);
            for(std::size_t ind_i = 0; ind_i < m_num_dims; ++ind_i) {
                r_0(ind_i) = nrm_dist(rng);
            }
            real_scalar_t const nrm2_r = inner_prod(r_0,r_0);
            real_scalar_t joint = log_p - 0.5*nrm2_r;
            real_scalar_t logu = joint - exp_dist(rng);

            real_vector_t theta_minus(theta_0);
            real_vector_t theta_plus(theta_0);
            real_vector_t theta_prime(theta_0);
            real_scalar_t log_p_prime(log_p);
            real_vector_t grad_prime(grad_0);
            real_vector_t r_minus(r_0);
            real_vector_t r_plus(r_0);
            real_vector_t grad_minus(grad_0);
            real_vector_t grad_plus(grad_0);
            std::size_t height_j(0);
            std::size_t valid_n(1);
            std::size_t valid_n_prine(1);
            std::size_t build_s(1);
            std::size_t build_s_prime(1);
            std::size_t alpha(1);
            std::size_t n_alpha(1);
            while (build_s) {
                int const dir_v = uni_real_dist(rng) > 0.5 ? 1 : -1;
                if ( dir_v == 1){

                }
                else {

                }

                if ( build_s_prime == 1
                    && uni_real_dist(rng)
                        < (real_scalar_t)valid_n_prine / (real_scalar_t)valid_n
                ){
                    hmc_chain.set_sample(count_m-1,theta_prime,log_p_prime);
                    log_p = log_p_prime;
                    grad_0 = grad_prime;
                }
                valid_n += valid_n_prine;
                build_s = build_s_prime * stop_criterion(
                    theta_minus,
                    theta_plus,
                    r_minus,
                    r_plus
                );
                height_j += 1;
            }
            real_scalar_t eta = 1./(real_scalar_t)(count_m-1 + t_0);
            H_bar = (1. - eta)*H_bar
                + eta * (m_delta - (real_scalar_t)alpha/(real_scalar_t)n_alpha);
            if ( count_m <= M_adapt){
                epsilon = std::exp( mu
                    - std::sqrt((real_scalar_t)count_m-1)/gamma*H_bar
                );
                eta = std::pow( (real_scalar_t)count_m-1,-kappa);
                epsilon_bar = std::exp(
                    (1. - eta)*std::log(epsilon_bar) + eta*std::log(epsilon)
                );
            }
            else {
                epsilon = epsilon_bar;
            }

        }
        return hmc_chain;
    }

    void build_tree(
        real_vector_t & theta_pm,
        real_vector_t & r_pm,
        real_vector_t & grad_pm,
        real_vector_t & theta_prime,
        real_vector_t & grad_prime,
        std::size_t & n_prine,
        std::size_t & s_prime,
        real_scalar_t & alpha_prime,
        std::size_t & n_alpha_prime,
        real_vector_t & theta,
        real_vector_t & r,
        real_vector_t & grad,
        real_scalar_t & log_u,
        int & v,
        std::size_t & j,
        real_scalar_t & epsilon,
        log_post_func_t & log_posterior,
        grad_log_post_func_t & grad_log_posterior,
        real_scalar_t & joint_0
    ){
        if (j ==0 ){
            theta_prime = theta,
            real_vector_t r_prime(r);

        }
    }

    std::size_t stop_criterion(
        real_vector_t const & theta_minus,
        real_vector_t const & theta_plus,
        real_vector_t const & r_minus,
        real_vector_t const & r_plus
    ) {
        real_vector_t theta_vec = theta_plus - theta_minus;
        if(
            inner_prod(theta_vec,r_minus) >= 0.
            && inner_prod(theta_vec,r_plus) >= 0.
        ){
            return std::size_t(1);
        }

        return std::size_t(0);
    }

    template<class rng_t>
    static real_scalar_t find_reasonable_epsilon(
        log_post_func_t & log_posterior,
        grad_log_post_func_t & grad_log_posterior,
        real_vector_t const & theta_0,
        rng_t & rng
    ){
        BOOST_ASSERT(
            theta_0.size() > 0
                and theta_0.size() <= MPP_MAXIMUM_NUMBER_OF_DIMENSIONS
        );
        using namespace boost::numeric::ublas;
        std::size_t const num_dims = theta_0.size();
        real_scalar_t log_p_0 = log_posterior(theta_0);
        real_vector_t theta_prm(theta_0);
        real_scalar_t epsilon = 1.;
        real_vector_t r_prm(num_dims);
        normal_dist_t nrm_dist;
        for(std::size_t ind_i = 0; ind_i < num_dims; ++ind_i) {
            r_prm(ind_i) = nrm_dist(rng);
        }
        real_scalar_t nrm_r2 = inner_prod(r_prm,r_prm);
        leap_frog(
            grad_log_posterior,
            theta_prm,
            r_prm,
            epsilon
        );
        real_scalar_t log_p_prm = log_posterior(theta_prm);
        real_scalar_t nrm_r2_prm = inner_prod(r_prm,r_prm);
        real_scalar_t acc_prob = std::exp(
            log_p_prm - log_p_0 - 0.5*( nrm_r2_prm -  nrm_r2 )
        );
        BOOST_ASSERT_MSG(
            std::isfinite(acc_prob),
            "Acceptance probability value is not finite."
        );
        real_scalar_t func_I_val = acc_prob > 0.5 ? 1. : 0.;
        real_scalar_t a = 2.*func_I_val - 1.;

        log_p_0 = log_p_prm;
        nrm_r2 = nrm_r2_prm;
        while( std::pow(acc_prob,a) > std::pow(2.,-a) ){
            epsilon = epsilon*std::pow(2.,a);
            leap_frog(
                grad_log_posterior,
                theta_prm,
                r_prm,
                epsilon
            );
            log_p_prm = log_posterior(theta_prm);
            nrm_r2_prm = inner_prod(r_prm,r_prm);
            acc_prob = std::exp(
                log_p_prm - log_p_0 - 0.5*( nrm_r2_prm -  nrm_r2 )
            );
            BOOST_ASSERT_MSG(
                std::isfinite(acc_prob),
                "Acceptance probability value is not finite."
            );
        }
        return epsilon;
    }

    static void leap_frog(
        grad_log_post_func_t & grad_log_posterior,
        real_vector_t & theta,
        real_vector_t & r,
        real_scalar_t const epsilon
    ){
        r += 0.5*epsilon*grad_log_posterior(theta);
        theta += epsilon*r;
        r += 0.5*epsilon*grad_log_posterior(theta);
    }
private:
    log_post_func_t m_log_posterior;
    grad_log_post_func_t m_grad_log_posterior;
    std::size_t m_num_dims;
    real_scalar_t m_delta;
};

}}


#endif //MPP_HAMILTONIAN_NUTS_HPP
