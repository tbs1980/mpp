#define BOOST_TEST_MODULE classic hamiltonian
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <random>
#include <functional>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/test/unit_test.hpp>
#include <mpp/chains/mcmc_chain.hpp>
#include <mpp/hamiltonian/classic_hamiltonian.hpp>
#include <mpp/dists/multivariate_normal.hpp>

template<typename real_scalar_type>
void test_classic_hamiltonian(std::string const & chn_file_name) {
    using namespace mpp::hamiltonian;
    using namespace mpp::chains;
    using namespace mpp::dists;
    using namespace boost::numeric::ublas;

    typedef diag_multivar_normal<real_scalar_type> diag_multivar_normal_type;
    typedef hmc_sampler<real_scalar_type> hmc_sampler_type;
    typedef vector<real_scalar_type> real_vector_type;
    typedef std::normal_distribution<real_scalar_type> normal_distribution_type;
    typedef std::mt19937 rng_type;
    typedef mcmc_chain<real_scalar_type> chain_type;
    typedef typename hmc_sampler_type::log_post_func_type log_post_func_type;
    typedef typename hmc_sampler_type::grad_log_post_func_type grad_log_post_func_type;

    size_t const num_dims(10);
    real_vector_type mean(num_dims);
    real_vector_type var(num_dims);
    for(size_t i=0;i<num_dims;++i) {
        mean(i) = real_scalar_type(0);
        var(i) = real_scalar_type(1);
    }
    diag_multivar_normal_type dmn(mean,var);
    using std::placeholders::_1;
    log_post_func_type log_posterior
        = std::bind (&diag_multivar_normal_type::log_posterior,&dmn,_1);

    grad_log_post_func_type grad_log_posterior
        = std::bind (&diag_multivar_normal_type::grad_log_posterior,&dmn,_1);
    size_t const max_num_steps(10);
    real_scalar_type const max_eps(1);
    real_vector_type inv_mass_mat(num_dims);
    for(size_t i=0;i<num_dims;++i) {
        inv_mass_mat(i) = real_scalar_type(1);
    }

    hmc_sampler_type hmc_spr(
        log_posterior,
        grad_log_posterior,
        num_dims,
        max_num_steps,
        max_eps,
        inv_mass_mat
    );

    size_t const num_samples(100);
    rng_type rng;
    normal_distribution_type nrm_dist;
    real_vector_type q_0(num_dims);
    for(size_t i=0;i<num_dims;++i) {
        q_0(i) = nrm_dist(rng);
    }
    chain_type chn = hmc_spr.run_sampler(num_samples,q_0,rng);
    chn.write_samples_to_csv(chn_file_name);

}

BOOST_AUTO_TEST_CASE(classic_hamiltonian) {
    test_classic_hamiltonian<float>(std::string("float.chain"));
    test_classic_hamiltonian<double>(std::string("double.chain"));
    test_classic_hamiltonian<long double>(std::string("long-double.chain"));
}
