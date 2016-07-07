#define BOOST_TEST_MODULE nuts sampler
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <random>
#include <cstddef>
#include <functional>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/test/unit_test.hpp>
#include <mpp/chains/mcmc_chain.hpp>
#include <mpp/hamiltonian/nuts.hpp>
#include <mpp/dists/multivariate_normal.hpp>

template<typename real_scalar_t>
void test_nuts(std::string const & chn_file_name) {
    using namespace mpp::hamiltonian;
    using namespace mpp::chains;
    using namespace mpp::dists;
    using namespace boost::numeric::ublas;
    typedef vector<real_scalar_t> real_vector_t;
    typedef diag_multivar_normal<real_scalar_t> diag_multivar_normal_t;
    typedef nut_sampler<real_scalar_t> nut_sampler_t;
    typedef typename nut_sampler_t::log_post_func_t log_post_func_t;
    typedef typename nut_sampler_t::grad_log_post_func_t grad_log_post_func_t;
    typedef std::mt19937 rng_t;
    typedef std::normal_distribution<real_scalar_t> normal_distribution_t;
    typedef mcmc_chain<real_scalar_t> chain_t;

    std::size_t const num_dims(10);
    real_vector_t mean(num_dims);
    real_vector_t var(num_dims);
    for(size_t i=0;i<num_dims;++i) {
        mean(i) = real_scalar_t(0);
        var(i) = real_scalar_t(1);
    }
    diag_multivar_normal_t dmn(mean,var);
    using std::placeholders::_1;
    log_post_func_t log_posterior
        = std::bind (&diag_multivar_normal_t::log_posterior,&dmn,_1);
    grad_log_post_func_t grad_log_posterior
        = std::bind (&diag_multivar_normal_t::grad_log_posterior,&dmn,_1);
    real_scalar_t const delta = 0.65;
    nut_sampler_t nuts_spr(
        log_posterior,
        grad_log_posterior,
        num_dims,
        delta
    );
    std::size_t const num_samples(100);
    rng_t rng;
    normal_distribution_t nrm_dist;
    real_vector_t theta_0(num_dims);
    for(std::size_t i=0;i<num_dims;++i) {
        theta_0(i) = nrm_dist(rng);
    }
    real_scalar_t res_eps = nuts_spr.find_reasonable_epsilon(
        log_posterior,
        grad_log_posterior,
        theta_0,
        rng
    );
    std::cout<<"realsonable epsilon = " << res_eps << std::endl;
    chain_t chn = nuts_spr.run_sampler(num_samples,theta_0,rng);
}

BOOST_AUTO_TEST_CASE(nuts) {
    test_nuts<float>(std::string("float.chain"));
    // test_nuts<double>(std::string("double.chain"));
    // test_nuts<long double>(std::string("long-double.chain"));
}
