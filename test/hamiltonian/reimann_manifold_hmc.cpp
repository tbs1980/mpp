#define BOOST_TEST_MODULE rmhmc sampler
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <cstddef>
#include <functional>
#include <iostream>
#include <random>
#include <mpp/dists/multivariate_normal.hpp>
#include <mpp/hamiltonian/reimann_manifold_hmc.hpp>

template<typename real_scalar_t>
void test_rmhmc_stomer_verlet() {
    using namespace mpp::dists;
    using namespace boost::numeric::ublas;
    using namespace mpp::hamiltonian;
    typedef diag_multivar_normal<real_scalar_t> diag_multivar_normal_t;
    typedef vector<real_scalar_t> real_vector_t;
    typedef matrix<real_scalar_t> real_matrix_t;
    typedef boost::numeric::ublas::unbounded_array<real_matrix_t>
        real_matrix_array_t;
    typedef rm_hmc_sampler<real_scalar_t> rm_hmc_sampler_t;
    typedef typename rm_hmc_sampler_t::log_post_func_t log_post_func_t;
    typedef typename rm_hmc_sampler_t::grad_log_post_func_t grad_log_post_func_t;
    typedef typename
        rm_hmc_sampler_t::mtr_tnsr_log_post_func_t mtr_tnsr_log_post_func_t;
    typedef typename
        rm_hmc_sampler_t::der_mtr_tnsr_log_post_func_t
            der_mtr_tnsr_log_post_func_t;
    typedef std::normal_distribution<real_scalar_t> normal_distribution_t;
    typedef std::mt19937 rng_type;

    size_t const num_dims(10);
    scalar_vector<real_scalar_t> mean(num_dims,0.);
    scalar_vector<real_scalar_t> var(num_dims,1.);
    diag_multivar_normal_t dmn(mean,var);
    using std::placeholders::_1;
    log_post_func_t log_posterior
        = std::bind (&diag_multivar_normal_t::log_posterior,&dmn,_1);
    grad_log_post_func_t grad_log_posterior
        = std::bind (&diag_multivar_normal_t::grad_log_posterior,&dmn,_1);
    mtr_tnsr_log_post_func_t metric_tensor_log_posterior = std::bind (
        &diag_multivar_normal_t::metric_tensor_log_posterior,
        &dmn,_1
    );
    der_mtr_tnsr_log_post_func_t deriv_metric_tensor_log_posterior = std::bind(
        &diag_multivar_normal_t::deriv_metric_tensor_log_posterior,
        &dmn,_1
    );

    // scalar_vector<real_scalar_t> q_test(num_dims,0);
    //
    // std::cout << " log posterior = " << dmn.log_posterior(q_test) << std::endl;
    //
    // std::cout << " gradient = " << dmn.grad_log_posterior(q_test) << std::endl;
    //
    // std::cout << " metric tensor = " << dmn.metric_tensor_log_posterior(q_test)
    //     << std::endl;
    //
    // std::cout << " derivative wrt each dimensionality of the metric tensor"
    //     << std::endl;
    // real_matrix_array_t d_G = dmn.deriv_metric_tensor_log_posterior(q_test);
    // for(std::size_t dim_i = 0; dim_i < num_dims; ++dim_i) {
    //     std::cout << dim_i << "\t" << d_G[dim_i] << std::endl;
    // }
    //

    std::size_t const num_leap_frog_steps = 1;
    std::size_t const num_fixed_point_steps = 1;
    real_scalar_t const step_size = 1;
    real_vector_t p_0(num_dims);
    real_vector_t x_0(num_dims);
    rng_type rng;
    normal_distribution_t norm_dist(0,1);
    for(std::size_t dim_i = 0; dim_i < num_dims; ++dim_i){
        p_0(dim_i) = norm_dist(rng);
        x_0(dim_i) = norm_dist(rng);
    }

    rm_hmc_sampler_t rm_hmc_spr(
        log_posterior,
        grad_log_posterior,
        metric_tensor_log_posterior,
        deriv_metric_tensor_log_posterior,
        num_dims,
        step_size,
        num_leap_frog_steps,
        num_fixed_point_steps
    );

    rm_hmc_spr.stomer_verlet(
        num_leap_frog_steps,
        num_fixed_point_steps,
        step_size,
        log_posterior,
        grad_log_posterior,
        metric_tensor_log_posterior,
        deriv_metric_tensor_log_posterior,
        p_0,
        x_0
    );

}

BOOST_AUTO_TEST_CASE(rmhmc_stomer_verlet) {
    test_rmhmc_stomer_verlet<double>();
}
