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
    typedef std::normal_distribution<real_scalar_t> normal_distribution_t;
    typedef mpp::hamiltonian::hmc_sampler<real_scalar_t> classic_hmc_sampler_t;
    typedef mpp::hamiltonian::multivariate_normal<real_scalar_t> kin_energy_t;

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

    }
private:
    template<class rng_t>
    static real_scalar_t find_reasonable_epsilon(
        log_post_func_t & log_posterior,
        grad_log_post_func_t & grad_log_posterior,
        real_vector_t & theta_0,
        real_scalar_t const eps,
        rng_t & rng
    ){
        using namespace boost::numeric::ublas;
        real_scalar_t epsilon = 1.;
        scalar_vector<real_scalar_t> m_inv(m_num_dims,1.);
        kin_energy_t kin_eng(m_inv);
        real_vector_t r_0 = kin_eng.generate_sample(rng);
        std::size_t const num_steps = 1;

        real_scalar_t const h_0 = -log_posterior(theta_0)
            -kin_eng.log_posterior(r_0);
        classic_hmc_sampler_t::leap_frog(
            grad_log_posterior,
            kin_eng,
            theta_0,
            r_0,
            eps,
            num_steps
        )



    }
    log_post_func_t m_log_posterior;
    grad_log_post_func_t m_grad_log_posterior;
    std::size_t m_num_dims;
    real_scalar_t m_delta;
};

}}


#endif //MPP_HAMILTONIAN_NUTS_HPP
