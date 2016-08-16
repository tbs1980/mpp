#ifndef MPP_REIMANN_MANIFOLD_HMC_HPP
#define MPP_REIMANN_MANIFOLD_HMC_HPP

#include <cstddef>
#include <random>
#include <cmath>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/multi_array.hpp>
#include "../utils/lin_alg_utils.hpp"

namespace mpp{ namespace hamiltonian {

template<class real_scalar_t>
class rm_hmc_sampler{
public:
    typedef boost::numeric::ublas::vector<real_scalar_t> real_vector_t;
    typedef boost::numeric::ublas::matrix<real_scalar_t> real_matrix_t;
    typedef mpp::chains::mcmc_chain<real_scalar_t> chain_type;
    typedef typename std::function<
        real_scalar_t (real_vector_type const &) > log_post_func_t;
    typedef typename std::function<
        real_vector_t (real_vector_type const &)> grad_log_post_func_t;
    typedef typename std::function<
        real_matrix_t (real_vector_type const &)> mtr_tnsr_log_post_func_t;
    typedef typename std::function<
        real_matrix_t (real_vector_type const &)> mtr_tnsr_der_log_post_func_t;
    typedef std::normal_distribution<real_scalar_t> normal_dist_t;

    rm_hmc_sampler(
        log_post_func_t & log_posterior,
        grad_log_post_func_t & grad_log_posterior,
        mtr_tnsr_log_post_func_t & mtr_tnsr_log_posterior,
        mtr_tnsr_der_log_post_func_t & mtr_tnsr_der_log_posterior,
        std::size_t const num_dims,
        std::size_t const max_epsilon,
        std::size_t const max_leap_frog_steps,
        std::size_t const max_fixed_point_steps
    ){

    }

    template<class rng_t>
    chain_type run_sampler(
        std::size_t const num_samples,
        real_vector_t const & start_point,
        rng_t & rng
    ) {
        using mpp::utils;
        using boost::numeric::ublas;

        normal_dist_t nrm_dist(0.,1.);
        std::size_t num_accepted = 0;
        std::size_t num_rejected = 0;
        real_vector_t state_x(start_point);
        real_matrix_t G = mtr_tnsr_log_posterior(state_x);
        real_matrix_t dG = mtr_tnsr_der_log_posterior(state_x);
        while( num_accepted < num_samples ) {

            real_matrix_t cholG = cholesky_decompose<real_scalar_t>(G);
            real_matrix_t invG = compute_inverse<real_scalar_t>(G);

            real_vector_t p(m_num_dims);
            for(std::size_t ind_i = 0; ind_i < m_num_dims; ++ind_i) {
                p(ind_i) = nrm_dist(rng);
            }
            p = prod(cholG,p);
        }
    }

    static real_scalar_t stomer_verlet(
        std::size_t const num_leap_frog_steps,
        std::size_t const num_fixed_point_steps,
        real_scalar_t const step_size,
        log_post_func_t & log_posterior,
        grad_log_post_func_t & grad_log_posterior,
        mtr_tnsr_log_post_func_t & mtr_tnsr_log_posterior,
        mtr_tnsr_der_log_post_func_t & mtr_tnsr_der_log_posterior,
        real_vector_t & p_new,
        real_vector_t & x_new
    ) {
        using namespace boost::numeric::ublas;
        using namespace mpp::utils;
        typedef boost::multi_array<real_scalar_t, num_dims> multi_array_t;
        real_scalar_t const c_e = 1e-4;
        std::size_t const num_dims = num_dims;
        real_scalar_t log_post_x0 = log_posterior(x_new);
        real_matrix_t G = mtr_tnsr_log_posterior(x_new);
        real_scalar_t det_G = compute_determinant<real_scalar_t>(G);
        real_matrix_t invG = compute_inverse<real_scalar_t>(G);
        real_scalar_t H_0 = -log_post_x0 + std::log(det_G)
            + 0.5*prod(p_new,prod(invG,p_new));
        multi_array_t d_G = mtr_tnsr_der_log_posterior(x_new);
        multi_array_t invG_dG_invG(d_G);
        real_vector_t tr_invG_dG(num_dims);
        for(std::size_t ind_i = 0; ind_i < num_dims; ++ind_i ){
            real_matrix_t d_G_i(num_dims,num_dims);
            for(std::size_t ind_j = 0; ind_j < num_dims; ++ind_j){
                for(std::size_t ind_k =0; ind_k < num_dims; ++ind_k){
                    d_G_i(ind_j,ind_k) = d_G[ind_i][ind_j][ind_k];
                }
            }
            real_matrix_t invG_dG_invG_i = prod(invG,d_G_i);
            tr_invG_dG(ind_i) = 0.;
            for(std::size_t ind_j = 0; ind_j < num_dims; ++ind_j){
                tr_invG_dG(ind_i) += invG_dG_invG_i(ind_j,ind_j);
            }
            invG_dG_invG_i = prod(invG_dG_invG_i,invG);
            for(std::size_t ind_j = 0; ind_j < num_dims; ++ind_j){
                for(std::size_t ind_k =0; ind_k < num_dims; ++ind_k){
                    invG_dG_invG[ind_i][ind_j][ind_k]
                        = invG_dG_invG_i(ind_j,ind_k);
                }
            }
        }

        real_vector_t p_0(p_new);
        real_scalar_t norm_p_0 = norm_2(p_0);
        real_vector_t pT_invG_dG_invG_p(num_dims);
        for(std::size_t ind_i = 0; ind_i < num_fixed_point_steps; ++i){
            for(std::size_t ind_j = 0; ind_j < num_dims; ++ ind_j) {
                real_matrix_t invG_dG_invG_i(num_dims,num_dims);
                for(std::size_t ind_k = 0; ind_k < num_dims; ++ind_k){
                    for(std::size_t ind_l =0; ind_l < num_dims; ++ind_l){
                        invG_dG_invG_i(ind_k,ind_l)
                            = invG_dG_invG[ind_j][ind_k][ind_l];
                    }
                }
                pT_invG_dG_invG_p(ind_j)
                    = inner_prod( p_new, prod(invG_dG_invG_i,p_new) );
            }
            p_new = p_0 - step_size*0.5*(
                grad_log_posterior(x_new)
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
        for(std::size_t ind_i = 0; ind_i < num_fixed_point_steps; ++i){
            x_new = x_0 + step_size*0.5*( prod(invG+invG_new,p_new) );
            G = mtr_tnsr_log_posterior(x_new);
            invG_new = compute_inverse<real_scalar_t>(G);
            real_scalar_t norm_x_new = norm_2(x_new);
            if( norm_2(x_0/norm_x_0 - x_new/norm_x_new) < c_e ){
                break;
            }
        }

        invG = invG_new;
        d_G = mtr_tnsr_der_log_posterior(x_new);
        for(std::size_t ind_i = 0; ind_i < num_dims; ++ind_i ){
            real_matrix_t d_G_i(num_dims,num_dims);
            for(std::size_t ind_j = 0; ind_j < num_dims; ++ind_j){
                for(std::size_t ind_k =0; ind_k < num_dims; ++ind_k){
                    d_G_i(ind_j,ind_k) = d_G[ind_i][ind_j][ind_k];
                }
            }
            real_matrix_t invG_dG_invG_i = prod(invG,d_G_i);
            tr_invG_dG(ind_i) = 0.;
            for(std::size_t ind_j = 0; ind_j < num_dims; ++ind_j){
                tr_invG_dG(ind_i) += invG_dG_invG_i(ind_j,ind_j);
            }
            invG_dG_invG_i = prod(invG_dG_invG_i,invG);
            for(std::size_t ind_j = 0; ind_j < num_dims; ++ind_j){
                for(std::size_t ind_k =0; ind_k < num_dims; ++ind_k){
                    invG_dG_invG[ind_i][ind_j][ind_k]
                        = invG_dG_invG_i(ind_j,ind_k);
                }
            }
        }

        for(std::size_t ind_j = 0; ind_j < num_dims; ++ ind_j) {
            real_matrix_t invG_dG_invG_i(num_dims,num_dims);
            for(std::size_t ind_k = 0; ind_k < num_dims; ++ind_k){
                for(std::size_t ind_l =0; ind_l < num_dims; ++ind_l){
                    invG_dG_invG_i(ind_k,ind_l)
                        = invG_dG_invG[ind_j][ind_k][ind_l];
                }
            }
            pT_invG_dG_invG_p(ind_j)
                = inner_prod( p_new, prod(invG_dG_invG_i,p_new) );
        }
        p_new = p_new - step_size*0.5*(
            grad_log_posterior(x_new)
            + 0.5*tr_invG_dG
            - 0.5*pT_invG_dG_invG_p
        );

        log_post_x0 = log_posterior(x_new);
        det_G = compute_determinant<real_scalar_t>(G);
        real_scalar_t H_new = -log_post_x0 + std::log(det_G)
            + 0.5*prod(p_new,prod(invG,p_new));

        return -H_new + H_0;
    }

    static real_scalar_t stomer_verlet(
        std::size_t const num_leap_frog_steps,
        std::size_t const num_fixed_point_steps,
        real_scalar_t const step_size,
        log_post_func_t & log_posterior,
        grad_log_post_func_t & grad_log_posterior,
        mtr_tnsr_log_post_func_t & mtr_tnsr_log_posterior,
        mtr_tnsr_der_log_post_func_t & mtr_tnsr_der_log_posterior,
        real_vector_t & p_new,
        real_vector_t & x_new
    ){
        using namespace boost::numeric::ublas;
        using namespace mpp::utils;
        typedef unbounded_array<real_matrix_t> real_matrix_array_t;
        real_scalar_t const c_e = 1e-4;
        std::size_t const num_dims = num_dims;

        real_scalar_t log_post_x0 = log_posterior(x_new);
        real_matrix_t G = mtr_tnsr_log_posterior(x_new);
        real_scalar_t det_G = compute_determinant<real_scalar_t>(G);
        real_matrix_t invG = compute_inverse<real_scalar_t>(G);
        real_scalar_t const H_0 = -log_post_x0 + std::log(det_G)
            + 0.5*prod(p_new,prod(invG,p_new));
        real_matrix_array_t d_G = mtr_tnsr_der_log_posterior(x_new);

        for(std::size_t lf_i = 0; lf_i < num_leap_frog_steps ; ++lf_i){
            real_matrix_array_t invG_dG_invG(d_G.size());
            real_vector_t tr_invG_dG(num_dims);
            for(std::size_t dim_i = 0; dim_i < num_dims; ++dim_i ){
                // real_matrix_t d_G_i(num_dims,num_dims);
                // for(std::size_t ind_j = 0; ind_j < num_dims; ++ind_j){
                //     for(std::size_t ind_k =0; ind_k < num_dims; ++ind_k){
                //         d_G_i(ind_j,ind_k) = d_G[dim_i][ind_j][ind_k];
                //     }
                // }
                real_matrix_t d_G_i = d_G[dim_i];
                real_matrix_t invG_dG_invG_i = prod(invG,d_G_i);
                tr_invG_dG(dim_i) = 0.;
                for(std::size_t ind_j = 0; ind_j < num_dims; ++ind_j){
                    tr_invG_dG(dim_i) += invG_dG_invG_i(ind_j,ind_j);
                }
                invG_dG_invG_i = prod(invG_dG_invG_i,invG);
                for(std::size_t ind_j = 0; ind_j < num_dims; ++ind_j){
                    for(std::size_t ind_k =0; ind_k < num_dims; ++ind_k){
                        invG_dG_invG[dim_i][ind_j][ind_k]
                            = invG_dG_invG_i(ind_j,ind_k);
                    }
                }
            }
        }
    }
private:
    log_post_func_t & m_log_posterior;
    grad_log_post_func_t & m_grad_log_posterior;
    mtr_tnsr_log_post_func_t & m_mtr_tnsr_log_posterior;
    mtr_tnsr_der_log_post_func_t & m_mtr_tnsr_der_log_posterior;
    std::size_t m_num_dims;
    std::size_t m_max_epsilon;
    std::size_t m_max_num_leap_frog_steps;
    std::size_t num_fixed_point_steps;
};

}}

#endif //MPP_REIMANN_MANIFOLD_HMC_HPP
