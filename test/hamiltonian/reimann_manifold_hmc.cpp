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
#include <limits>
#include <string>
#include <mpp/dists/multivariate_normal.hpp>
#include <mpp/dists/banana.hpp>
#include <mpp/hamiltonian/reimann_manifold_hmc.hpp>
#include <mpp/hamiltonian/classic_hamiltonian.hpp>

template<typename real_scalar_t>
void test_rmhmc_stomer_verlet() {
    using namespace mpp::dists;
    using namespace boost::numeric::ublas;
    using namespace mpp::hamiltonian;
    typedef diag_multivar_normal<real_scalar_t> diag_multivar_normal_t;
    typedef vector<real_scalar_t> real_vector_t;
    typedef rm_hmc_sampler<real_scalar_t> rm_hmc_sampler_t;
    typedef typename rm_hmc_sampler_t::log_post_func_t log_post_func_t;
    typedef typename rm_hmc_sampler_t::grad_log_post_func_t grad_log_post_func_t;
    typedef typename
        rm_hmc_sampler_t::mtr_tnsr_log_post_func_t mtr_tnsr_log_post_func_t;
    typedef typename
        rm_hmc_sampler_t::der_mtr_tnsr_log_post_func_t
            der_mtr_tnsr_log_post_func_t;
    typedef std::normal_distribution<real_scalar_t> normal_distribution_t;
    typedef std::mt19937 rng_t;

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

    std::size_t const num_leap_frog_steps = 6;
    std::size_t const num_fixed_point_steps = 4;
    real_scalar_t const step_size = 1;
    real_vector_t p_0(num_dims);
    real_vector_t x_0(num_dims);
    rng_t rng;
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

    real_vector_t x_test(x_0);
    real_scalar_t delta_h = rm_hmc_spr.stomer_verlet(
        num_leap_frog_steps,
        num_fixed_point_steps,
        step_size,
        log_posterior,
        grad_log_posterior,
        metric_tensor_log_posterior,
        deriv_metric_tensor_log_posterior,
        p_0,
        x_test
    );

    BOOST_CHECK(std::isfinite(delta_h));

    real_scalar_t eps = std::numeric_limits<real_scalar_t>::epsilon();
    for(std::size_t dim_i = 0; dim_i < num_dims; ++dim_i){
        BOOST_CHECK(
            std::abs(x_test(dim_i) - x_0(dim_i)) <= eps*10.
        );
    }

}

template<typename real_scalar_t>
void test_rmhmc(std::string const & chn_file_name){
    using namespace mpp::dists;
    using namespace boost::numeric::ublas;
    using namespace mpp::hamiltonian;
    using namespace mpp::chains;

    typedef diag_multivar_normal<real_scalar_t> diag_multivar_normal_t;
    typedef vector<real_scalar_t> real_vector_t;
    typedef rm_hmc_sampler<real_scalar_t> rm_hmc_sampler_t;
    typedef typename rm_hmc_sampler_t::log_post_func_t log_post_func_t;
    typedef typename rm_hmc_sampler_t::grad_log_post_func_t grad_log_post_func_t;
    typedef typename
        rm_hmc_sampler_t::mtr_tnsr_log_post_func_t mtr_tnsr_log_post_func_t;
    typedef typename
        rm_hmc_sampler_t::der_mtr_tnsr_log_post_func_t
            der_mtr_tnsr_log_post_func_t;
    typedef std::normal_distribution<real_scalar_t> normal_distribution_t;
    typedef std::mt19937 rng_t;
    typedef mcmc_chain<real_scalar_t> chain_t;

    size_t const num_dims(10);
    scalar_vector<real_scalar_t> mean(num_dims, 0.);
    scalar_vector<real_scalar_t> var(num_dims, 1.);
    diag_multivar_normal_t dmn(mean, var);
    using std::placeholders::_1;
    log_post_func_t log_posterior
        = std::bind (&diag_multivar_normal_t::log_posterior, &dmn, _1);
    grad_log_post_func_t grad_log_posterior
        = std::bind (&diag_multivar_normal_t::grad_log_posterior, &dmn, _1);
    mtr_tnsr_log_post_func_t metric_tensor_log_posterior = std::bind (
        &diag_multivar_normal_t::metric_tensor_log_posterior,
        &dmn,
        _1
    );
    der_mtr_tnsr_log_post_func_t deriv_metric_tensor_log_posterior = std::bind(
        &diag_multivar_normal_t::deriv_metric_tensor_log_posterior,
        &dmn,
        _1
    );
    std::size_t const num_leap_frog_steps = 6;
    std::size_t const num_fixed_point_steps = 4;
    real_scalar_t const step_size = 1;
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

    size_t const num_samples(100);
    real_vector_t x_0(num_dims);
    rng_t rng;
    normal_distribution_t norm_dist(0,1);
    for(std::size_t dim_i = 0; dim_i < num_dims; ++dim_i){
        x_0(dim_i) = norm_dist(rng);
    }
    chain_t chn = rm_hmc_spr.run_sampler(num_samples,x_0,rng);
    chn.write_samples_to_csv(chn_file_name);
    // std::cout << "acc rate = " << rm_hmc_spr.acc_rate() << std::endl;
}

// BOOST_AUTO_TEST_CASE(rmhmc_stomer_verlet) {
//     test_rmhmc_stomer_verlet<float>();
//     test_rmhmc_stomer_verlet<double>();
//     test_rmhmc_stomer_verlet<long double>();
// }
//
// BOOST_AUTO_TEST_CASE(rmhmc){
//     test_rmhmc<float>(std::string("float.chain"));
//     test_rmhmc<double>(std::string("double.chain"));
//     test_rmhmc<long double>(std::string("long-double.chain"));
// }

template<typename real_scalar_t>
void test_rmhmc_banana(std::string const & chn_file_name)
{
    using namespace mpp::dists;
    using namespace boost::numeric::ublas;
    using namespace mpp::hamiltonian;
    using namespace mpp::chains;

    typedef banana<real_scalar_t> banana_t;
    typedef vector<real_scalar_t> real_vector_t;
    typedef rm_hmc_sampler<real_scalar_t> rm_hmc_sampler_t;
    typedef typename rm_hmc_sampler_t::log_post_func_t log_post_func_t;
    typedef typename rm_hmc_sampler_t::grad_log_post_func_t grad_log_post_func_t;
    typedef typename
        rm_hmc_sampler_t::mtr_tnsr_log_post_func_t mtr_tnsr_log_post_func_t;
    typedef typename
        rm_hmc_sampler_t::der_mtr_tnsr_log_post_func_t
            der_mtr_tnsr_log_post_func_t;
    typedef std::normal_distribution<real_scalar_t> normal_distribution_t;
    typedef std::mt19937 rng_t;
    typedef mcmc_chain<real_scalar_t> chain_t;

    real_scalar_t const theta_1_plus_theta_2_sq = 1.;
    real_scalar_t const sigma_y = 2.;
    real_scalar_t const sigma_theta = 1.;
    std::size_t const num_data_points = 100;
    std::size_t const random_seed = 12345;
    bool const force_diag_metric_tensor = false;

    banana_t bna(
        theta_1_plus_theta_2_sq,
        sigma_y,
        sigma_theta,
        num_data_points,
        random_seed,
        force_diag_metric_tensor
    );
    using std::placeholders::_1;
    log_post_func_t log_posterior
        = std::bind (&banana_t::log_posterior, &bna, _1);
    grad_log_post_func_t grad_log_posterior
        = std::bind (&banana_t::grad_log_posterior, &bna, _1);
    mtr_tnsr_log_post_func_t metric_tensor_log_posterior = std::bind (
        &banana_t::metric_tensor_log_posterior,
        &bna,
        _1
    );
    der_mtr_tnsr_log_post_func_t deriv_metric_tensor_log_posterior = std::bind(
        &banana_t::deriv_metric_tensor_log_posterior,
        &bna,
        _1
    );

    std::size_t const num_leap_frog_steps = 5;
    std::size_t const num_fixed_point_steps = 5;
    real_scalar_t const step_size = 1.2/50.;
    std::size_t const num_dims = 2;
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

    size_t const num_samples(1000);
    real_vector_t x_0(num_dims);
    rng_t rng;
    normal_distribution_t norm_dist(0,1);
    for(std::size_t dim_i = 0; dim_i < num_dims; ++dim_i){
        x_0(dim_i) = norm_dist(rng);
    }
    chain_t chn = rm_hmc_spr.run_sampler(num_samples,x_0,rng);
    chn.write_samples_to_csv(chn_file_name);
    std::cout << "acc rate = " << rm_hmc_spr.acc_rate() << std::endl;
}

template<typename real_scalar_t>
void test_hmc_banana(std::string const & chn_file_name) {
    using namespace mpp::hamiltonian;
    using namespace mpp::chains;
    using namespace mpp::dists;
    using namespace boost::numeric::ublas;

    typedef banana<real_scalar_t> banana_t;
    typedef hmc_sampler<real_scalar_t> hmc_sampler_t;
    typedef vector<real_scalar_t> real_vector_t;
    typedef std::normal_distribution<real_scalar_t> normal_distribution_t;
    typedef std::mt19937 rng_t;
    typedef mcmc_chain<real_scalar_t> chain_t;
    typedef typename hmc_sampler_t::log_post_func_type log_post_func_t;
    typedef typename hmc_sampler_t::grad_log_post_func_type grad_log_post_func_t;

    size_t const num_dims(2);
    real_scalar_t const theta_1_plus_theta_2_sq = 1.;
    real_scalar_t const sigma_y = 2.;
    real_scalar_t const sigma_theta = 1.;
    std::size_t const num_data_points = 100;
    std::size_t const random_seed = 12345;
    bool const force_diag_metric_tensor = false;

    banana_t bna(
        theta_1_plus_theta_2_sq,
        sigma_y,
        sigma_theta,
        num_data_points,
        random_seed,
        force_diag_metric_tensor
    );


    using std::placeholders::_1;
    log_post_func_t log_posterior
        = std::bind (&banana_t::log_posterior, &bna, _1);

    grad_log_post_func_t grad_log_posterior
        = std::bind (&banana_t::grad_log_posterior, &bna, _1);
    size_t const max_num_steps(10);
    real_scalar_t const max_eps(0.1);
    real_vector_t inv_mass_mat(num_dims);
    for(size_t i=0;i<num_dims;++i) {
        inv_mass_mat(i) = sigma_y*sigma_y*sigma_theta*sigma_theta
            /(sigma_y*sigma_y + sigma_theta*sigma_theta);
    }

    hmc_sampler_t hmc_spr(
        log_posterior,
        grad_log_posterior,
        num_dims,
        max_num_steps,
        max_eps,
        inv_mass_mat
    );

    size_t const num_samples(1000);
    rng_t rng;
    normal_distribution_t nrm_dist;
    real_vector_t q_0(num_dims);
    for(size_t i=0;i<num_dims;++i) {
        q_0(i) = nrm_dist(rng);
    }
    chain_t chn = hmc_spr.run_sampler(num_samples,q_0,rng);
    chn.write_samples_to_csv(chn_file_name);

}


BOOST_AUTO_TEST_CASE(rmhmc_banana){
    test_rmhmc_banana<float>(std::string("banana.float.chain"));
    // test_rmhmc_banana<double>(std::string("banana.double.chain"));
    // test_rmhmc_banana<long double>(std::string("banana.long-double.chain"));
}

// BOOST_AUTO_TEST_CASE(hmc_banana){
//     test_hmc_banana<float>(std::string("banana.float.chain"));
//     test_hmc_banana<double>(std::string("banana.double.chain"));
//     test_hmc_banana<long double>(std::string("banana.long-double.chain"));
//
// }
