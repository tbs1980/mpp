#ifndef MPP_REIMANN_MANIFOLD_HMC_HPP
#define MPP_REIMANN_MANIFOLD_HMC_HPP

#include <cstddef>
#include <random>
#include <cmath>
#include <iostream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/multi_array.hpp>
#include "../config.hpp"
#include "../utils/lin_alg_utils.hpp"
#include "../chains/mcmc_chain.hpp"

namespace mpp{ namespace hamiltonian {

template<class real_scalar_t>
class rm_hmc_sampler{
public:
    typedef boost::numeric::ublas::vector<real_scalar_t> real_vector_t;
    typedef boost::numeric::ublas::matrix<real_scalar_t> real_matrix_t;
    typedef boost::numeric::ublas::unbounded_array<real_matrix_t>
        real_matrix_array_t;
    typedef mpp::chains::mcmc_chain<real_scalar_t> chain_t;
    typedef std::uniform_real_distribution<real_scalar_t> uni_real_dist_t;
    typedef std::uniform_int_distribution<size_t> uni_int_dist_t;
    typedef std::normal_distribution<real_scalar_t> normal_distribution_t;
    typedef typename std::function<
        real_scalar_t (real_vector_t const &) > log_post_func_t;
    typedef typename std::function<
        real_vector_t (real_vector_t const &)> grad_log_post_func_t;
    typedef typename std::function<
        real_matrix_t (real_vector_t const &)> mtr_tnsr_log_post_func_t;
    typedef typename std::function<
        real_matrix_array_t (real_vector_t const&)> der_mtr_tnsr_log_post_func_t;
    typedef std::normal_distribution<real_scalar_t> normal_dist_t;

    rm_hmc_sampler(
        log_post_func_t & log_posterior,
        grad_log_post_func_t & grad_log_posterior,
        mtr_tnsr_log_post_func_t & mtr_tnsr_log_posterior,
        der_mtr_tnsr_log_post_func_t & drv_mtr_tnsr_log_posterior,
        std::size_t const num_dims,
        real_scalar_t const max_epsilon,
        std::size_t const max_leap_frog_steps,
        std::size_t const max_fixed_point_steps
    )
    : m_log_posterior(log_posterior)
    , m_grad_log_posterior(grad_log_posterior)
    , m_mtr_tnsr_log_posterior(mtr_tnsr_log_posterior)
    , m_drv_mtr_tnsr_log_posterior(drv_mtr_tnsr_log_posterior)
    , m_num_dims(num_dims)
    , m_max_epsilon(max_epsilon)
    , m_max_num_leap_frog_steps(max_leap_frog_steps)
    , m_num_fixed_point_steps(max_fixed_point_steps)
    , m_beta(1)
    , m_acc_rate(0) {

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

        if( m_max_num_leap_frog_steps == size_t(0) ) {
            std::stringstream msg;
            msg << "Maximum number of steps = "
                << m_max_num_leap_frog_steps
                << " in the discretisation of Hamiltonian should be"
                << " greater than zero.";
            throw std::length_error(msg.str());
        }

        BOOST_ASSERT_MSG(
            m_max_num_leap_frog_steps
            <= size_t(MPP_RMHMC_MAX_NUM_LEAPFROG_STEPS),
            "max_num_steps too big. Please modify the config.hpp and recompile."
        );

        if( m_max_epsilon <= real_scalar_t(0)
            or m_max_epsilon > real_scalar_t(1)) {
            std::stringstream msg;
            msg << "Maximum value of epsilon = "
                << m_max_epsilon
                << " in the discretisation of Hamiltonian should be"
                << " in the interval (0,1].";
            throw std::domain_error(msg.str());
        }

    }

    template<class rng_t>
    chain_t run_sampler(
        std::size_t const num_samples,
        real_vector_t const & start_point,
        rng_t & rng
    ) {
        using namespace boost::numeric::ublas;
        using namespace mpp::utils;

        BOOST_ASSERT_MSG(
            num_samples <=
                size_t(MPP_MAXIMUM_NUMBER_OF_SAMPLES_PER_RUN_SAMPLER_CALL),
            "num_samples too big. Please modify the config and recompile."
        );
        if( start_point.size() !=  m_num_dims){
            std::stringstream msg;
            msg << "The number of dimensions = "
                << m_num_dims
                << " is not equal to the length of start_point = "
                << start_point.size();
            throw std::length_error(msg.str());
        }

        chain_t rmhmc_chain(num_samples,m_num_dims);
        uni_real_dist_t uni_real_dist;
        uni_int_dist_t uni_int_dist( size_t(1),
            m_max_num_leap_frog_steps+size_t(1)
        );
        normal_distribution_t norm_dist;
        real_vector_t q_1(start_point);
        size_t num_accepted(0);
        size_t num_rejected(0);
        while( num_accepted < num_samples ) {
            real_vector_t q_0(q_1);
            real_matrix_t G = m_mtr_tnsr_log_posterior(q_0);
            real_matrix_t chol_G = zero_matrix<real_scalar_t>(
                G.size1(),
                G.size2()
            );
            std::size_t res = cholesky_decompose<real_scalar_t>(G,chol_G);
            BOOST_ASSERT_MSG(res == 0,"Matrix G is not positive definite.");
            real_scalar_t const step_size = m_max_epsilon*uni_real_dist(rng);
            size_t const num_leap_frog_steps = uni_int_dist(rng);
            real_vector_t p_0(m_num_dims);
            for(std::size_t dim_i = 0; dim_i < m_num_dims; ++dim_i){
                p_0(dim_i) = norm_dist(rng);
            }
            p_0 = prod(p_0,chol_G);

            real_scalar_t const delta_h = stomer_verlet(
                num_leap_frog_steps,
                m_num_fixed_point_steps,
                step_size,
                m_log_posterior,
                m_grad_log_posterior,
                m_mtr_tnsr_log_posterior,
                m_drv_mtr_tnsr_log_posterior,
                p_0,
                q_0
            );

            if( not std::isfinite(delta_h) ) {
                std::stringstream msg;
                msg << "delta(H) value is not finite";
                throw std::out_of_range(msg.str());
            }
            real_scalar_t const uni_rand = uni_real_dist(rng);
            if(std::log(uni_rand) < -delta_h*m_beta) {
                q_1 = q_0;
                real_scalar_t log_post_val = m_log_posterior(q_1);
                rmhmc_chain.set_sample(num_accepted,q_1,log_post_val);
                ++num_accepted;
            }
            else {
                ++num_rejected;
            }
        }
        m_acc_rate = real_scalar_t(num_accepted)
            / real_scalar_t(num_accepted + num_rejected);

        return rmhmc_chain;
    }

    inline real_scalar_t acc_rate() const {
        return m_acc_rate;
    }

    real_scalar_t stomer_verlet(
        std::size_t const num_leap_frog_steps,
        std::size_t const num_fixed_point_steps,
        real_scalar_t const step_size,
        log_post_func_t & log_posterior,
        grad_log_post_func_t & grad_log_posterior,
        mtr_tnsr_log_post_func_t & mtr_tnsr_log_posterior,
        der_mtr_tnsr_log_post_func_t & drv_mtr_tnsr_log_posterior,
        real_vector_t & p_new,
        real_vector_t & x_new
    ){
        using namespace boost::numeric::ublas;
        using namespace mpp::utils;

        real_scalar_t const c_e = 1e-4;
        std::size_t const num_dims = p_new.size();

        real_matrix_t G = mtr_tnsr_log_posterior(x_new);
        real_scalar_t det_G = compute_determinant<real_scalar_t>(G);
        real_matrix_t invG(G);
        compute_inverse<real_scalar_t>(G,invG);
        real_scalar_t log_post_x0 = log_posterior(x_new);
        BOOST_ASSERT(det_G > 0);
        real_scalar_t const H_0 = -log_post_x0 + std::log(det_G)
            + 0.5*inner_prod(p_new,prod(invG,p_new));
        BOOST_ASSERT(std::isfinite(H_0));

        real_matrix_array_t d_G = drv_mtr_tnsr_log_posterior(x_new);
        for(std::size_t lf_i = 0; lf_i < num_leap_frog_steps ; ++lf_i){
            real_matrix_array_t invG_dG_invG(d_G.size());
            real_vector_t tr_invG_dG(num_dims);
            for(std::size_t dim_i = 0; dim_i < num_dims; ++dim_i ){
                real_matrix_t d_G_i = d_G[dim_i];
                real_matrix_t invG_dG_invG_i = prod(invG,d_G_i);
                tr_invG_dG(dim_i)
                    = compute_trace<real_scalar_t>(invG_dG_invG_i);
                invG_dG_invG_i = prod(invG_dG_invG_i,invG);
                invG_dG_invG[dim_i] = invG_dG_invG_i;
            }

            real_vector_t p_0(p_new);
            real_scalar_t norm_p_0 = norm_2(p_0);
            real_vector_t pT_invG_dG_invG_p(num_dims);
            for(std::size_t fp_i = 0; fp_i < num_fixed_point_steps; ++fp_i){
                for(std::size_t dim_i = 0; dim_i < num_dims; ++ dim_i) {
                    real_matrix_t invG_dG_invG_i = invG_dG_invG[dim_i];
                    pT_invG_dG_invG_p(dim_i)
                        = inner_prod( p_new, prod(invG_dG_invG_i,p_new) );
                }
                p_new = p_0 - step_size*0.5*(
                    -grad_log_posterior(x_new)
                    + 0.5*tr_invG_dG
                    - 0.5*pT_invG_dG_invG_p
                );
                real_scalar_t norm_p_new = norm_2(p_new);
                if( norm_2(p_0/norm_p_0 - p_new/norm_p_new) < c_e ){
                    break;
                }
            }

            real_vector_t x_0(x_new);
            real_matrix_t invG_new(invG);
            real_scalar_t norm_x_0 = norm_2(x_0);
            for(std::size_t fp_i = 0; fp_i < num_fixed_point_steps; ++fp_i){
                real_vector_t prod_invG_invG_new_p_new = prod(invG+invG_new,p_new);
                x_new = x_0 + step_size*0.5*( prod_invG_invG_new_p_new );
                // TODO check for nans here.
                G = mtr_tnsr_log_posterior(x_new);
                compute_inverse<real_scalar_t>(G,invG_new);
                real_scalar_t norm_x_new = norm_2(x_new);
                if( norm_2(x_0/norm_x_0 - x_new/norm_x_new) < c_e ){
                    break;
                }
            }

            invG = invG_new;
            d_G = drv_mtr_tnsr_log_posterior(x_new);
            for(std::size_t dim_i = 0; dim_i < num_dims; ++dim_i ){
                real_matrix_t d_G_i = d_G[dim_i];
                real_matrix_t invG_dG_invG_i = prod(invG,d_G_i);
                tr_invG_dG(dim_i)
                    = compute_trace<real_scalar_t>(invG_dG_invG_i);
                invG_dG_invG_i = prod(invG_dG_invG_i,invG);
                invG_dG_invG[dim_i] = invG_dG_invG_i;
            }

            for(std::size_t dim_i = 0; dim_i < num_dims; ++ dim_i) {
                real_matrix_t invG_dG_invG_i = invG_dG_invG[dim_i];
                pT_invG_dG_invG_p(dim_i)
                    = inner_prod( p_new, prod(invG_dG_invG_i,p_new) );
            }
            p_new = p_new - step_size*0.5*(
                -grad_log_posterior(x_new)
                + 0.5*tr_invG_dG
                - 0.5*pT_invG_dG_invG_p
            );
        }
        real_scalar_t const log_post_x_new = log_posterior(x_new);
        det_G = compute_determinant<real_scalar_t>(G);
        if (std::isfinite(log_post_x_new) == false or det_G <= 0.){
            return std::numeric_limits<real_scalar_t>::max();
        }
        real_scalar_t const H_new = -log_post_x_new + std::log(det_G)
            + 0.5*inner_prod(p_new,prod(invG,p_new));
        BOOST_ASSERT(std::isfinite(H_new));
        return (H_new - H_0);
    }

private:
    log_post_func_t & m_log_posterior;
    grad_log_post_func_t & m_grad_log_posterior;
    mtr_tnsr_log_post_func_t & m_mtr_tnsr_log_posterior;
    der_mtr_tnsr_log_post_func_t & m_drv_mtr_tnsr_log_posterior;
    std::size_t m_num_dims;
    real_scalar_t m_max_epsilon;
    std::size_t m_max_num_leap_frog_steps;
    std::size_t m_num_fixed_point_steps;
    real_scalar_t m_beta;
    real_scalar_t m_acc_rate;
};

}}

#endif //MPP_REIMANN_MANIFOLD_HMC_HPP
