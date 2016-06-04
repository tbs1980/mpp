#ifndef MPP_HAMILTONIAN_CLASSIC_HAMILTONIAN_HPP
#define MPP_HAMILTONIAN_CLASSIC_HAMILTONIAN_HPP

#include <exception>
#include <sstream>
#include <string>
#include <random>
#include <type_traits>
#include <cmath>
#include <functional>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/assert.hpp>
#include <mpp/config.hpp>
#include <mpp/chains/mcmc_chain.hpp>
#include <mpp/hamiltonian/kinetic_energy_multivar_normal.hpp>

namespace mpp { namespace hamiltonian {

template<class real_scalar_type>
class hmc_sampler
{
public:
    typedef boost::numeric::ublas::vector<real_scalar_type> real_vector_type;
    typedef mpp::chains::mcmc_chain<real_scalar_type> chain_type;
    typedef std::mt19937 rng_type;
    typedef std::uniform_real_distribution<real_scalar_type> uni_real_dist_type;
    typedef std::uniform_int_distribution<size_t> uni_int_dist_type;
    typedef mpp::hamiltonian::multivariate_normal<
        real_scalar_type> kinetic_energy_type;

    typedef typename std::function< 
        real_scalar_type (real_vector_type const &) > log_post_func_type;

    typedef typename std::function< 
        real_vector_type (real_vector_type const &)> grad_log_post_func_type;

    static_assert(
        std::is_floating_point<real_scalar_type>::value,
        "The real scalar is expected to be a floating point type."
    );

    hmc_sampler (
        log_post_func_type & log_posterior,
        grad_log_post_func_type & grad_log_posterior,
        size_t const num_dims,
        size_t const max_num_steps,
        real_scalar_type const max_eps,
        real_vector_type const & inv_mass_mat
    ) throw()
    : m_log_posterior(log_posterior)
    , m_grad_log_posterior(grad_log_posterior)
    , m_num_dims(num_dims)
    , m_max_num_steps(max_num_steps)
    , m_max_eps(max_eps)
    , m_inv_mass_mat(inv_mass_mat)
    , m_beta(1)
    ,m_acc_rate(0) {

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

        if( m_max_num_steps == size_t(0) ) {
            std::stringstream msg;
            msg << "Maximum number of steps = "
                << m_max_num_steps
                << " in the discretisation of Hamiltonian should be"
                << " greater than zero.";
            throw std::length_error(msg.str());
        }

        BOOST_ASSERT_MSG(
            m_max_num_steps <= size_t(MPP_CLASSIC_HAMILTONIAN_MAX_NUM_STEPS),
            "max_num_steps too big. Please modify the config.hpp and recompile."
        );

        if( m_max_eps <= real_scalar_type(0) 
            or m_max_eps > real_scalar_type(1)) {
            std::stringstream msg;
            msg << "Maximum value of epsilon = "
                << m_max_eps
                << " in the discretisation of Hamiltonian should be"
                << " in the intervale (0,1].";
            throw std::domain_error(msg.str());
        }

    }

    chain_type run_sampler(size_t const num_samples,
        real_vector_type const & start_point) throw() {
        BOOST_ASSERT_MSG(
            num_samples <=
                size_t(MPP_MAXIMUM_NUMBER_OF_SAMPLES_PER_RUN_SAMPLER_CALL),
            "num_samples too big. Please modify the config and recompile."
        );

        chain_type hmc_chain(num_samples,m_num_dims);
        uni_real_dist_type uni_real_dist;
        uni_int_dist_type uni_int_dist( size_t(1),m_max_num_steps+size_t(1) );
        kinetic_energy_type kin_eng(m_inv_mass_mat);
        real_vector_type q_1(start_point);
        size_t num_accepted(0);
        size_t num_rejected(0);
        while( num_accepted < num_samples ) {
            real_vector_type q_0(q_1);
            real_scalar_type const eps = m_max_eps*uni_real_dist(m_rng);
            size_t const num_steps = uni_int_dist(m_rng);
            real_vector_type p_0 = kin_eng.generate_sample(m_rng);

            real_scalar_type const h_0 = -m_log_posterior(q_0)
                -kin_eng.log_posterior(p_0);
            leap_frog(m_grad_log_posterior,kin_eng,q_0,p_0,eps,num_steps);
            real_scalar_type const log_post_val = m_log_posterior(q_0);
            real_scalar_type const h_1 = -log_post_val
                -kin_eng.log_posterior(p_0);
            real_scalar_type const delta_h = h_1 - h_0;

            if( not std::isfinite(delta_h) ) {
                std::stringstream msg;
                msg << "delta(H) value is not finite";
                throw std::out_of_range(msg.str());
            }
            real_scalar_type const uni_rand = uni_real_dist(m_rng);
            if(std::log(uni_rand) < -delta_h*m_beta) {
                q_1 = q_0;
                hmc_chain.set_sample(num_accepted,q_1,log_post_val);
                ++num_accepted;
            } 
            else {
                ++num_rejected;
            }
        }

        m_acc_rate = real_scalar_type(num_accepted)
            / real_scalar_type(num_accepted + num_rejected);

        return hmc_chain;
    }

    inline real_scalar_type acc_rate() const {
        return m_acc_rate;
    }

private:

    static void leap_frog(
        grad_log_post_func_type & grad_log_posterior,
        kinetic_energy_type & kin_eng,
        real_vector_type & q,
        real_vector_type & p,
        real_scalar_type const eps,
        size_t const num_steps
    ) {
        BOOST_ASSERT_MSG(
            num_steps <= size_t(MPP_CLASSIC_HAMILTONIAN_MAX_NUM_STEPS),
            "m_max_num_steps too big. Please modify the config and recompile."
        );

        BOOST_ASSERT_MSG(
            eps > real_scalar_type(0) and eps <= real_scalar_type(1),
            "epsilon value should be a real number in [0,1]"
        );

        BOOST_ASSERT_MSG(
            p.size() == q.size(),
            "p and q should have identical dimensions"
        );

        real_vector_type dq = grad_log_posterior(q);
        real_vector_type dp = kin_eng.grad_log_posterior(p);
        size_t const num_dims = p.size();

        for(size_t i=0;i<num_dims;++i) {
            p(i) = p(i) + real_scalar_type(0.5)*eps*dq(i);
        }

        for(size_t j=0;j<num_steps;++j) {
            dp = kin_eng.grad_log_posterior(p);
            for(size_t i=0;i<num_dims;++i) {
                q(i) = q(i) - eps*dp(i);
            }
            dq = grad_log_posterior(q);
            for(size_t i=0;i<num_dims;++i) {
                p(i) = p(i) + eps*dq(i);
            }
        }

        for(size_t i=0;i<num_dims;++i) {
            p(i) = p(i) - real_scalar_type(0.5)*eps*dq(i);
        }

    }

    log_post_func_type m_log_posterior;
    grad_log_post_func_type m_grad_log_posterior;
    size_t m_num_dims;
    size_t m_max_num_steps;
    real_scalar_type m_max_eps;
    real_vector_type m_inv_mass_mat;
    rng_type m_rng;
    real_scalar_type m_beta;
    real_scalar_type m_acc_rate;
};

}}

#endif // MPP_HAMILTONIAN_CLASSIC_HAMILTONIAN_HPP
