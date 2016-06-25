#define BOOST_TEST_MODULE mcmc chain
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <random>
#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <mpp/chains/mcmc_chain.hpp>
#include <mpp/ensemble/red_blue.hpp>
#include <mpp/dists/multivariate_normal.hpp>

template<typename real_scalar_type>
void test_red_blue(std::string const & chn_file_name) {
    using namespace mpp::ensemble;
    using namespace mpp::chains;
    using namespace mpp::dists;
    using namespace boost::numeric::ublas;

    typedef diag_multivar_normal<real_scalar_type> diag_multivar_normal_type;
    typedef red_blue_sampler<real_scalar_type> rb_sampler_type;
    typedef vector<real_scalar_type> real_vector_type;
    typedef matrix<real_scalar_type> real_matrix_type;
    typedef std::normal_distribution<real_scalar_type> normal_distribution_type;
    typedef std::mt19937 rng_type;
    typedef mcmc_chain<real_scalar_type> chain_type;
    typedef typename rb_sampler_type::log_post_func_type log_post_func_type;
    typedef typename rb_sampler_type::grad_log_post_func_type grad_log_post_func_type;

    size_t const num_dims(4);
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
    std::size_t const num_walkers = 2*num_dims;
    real_scalar_type const scale_a = 2;

    rb_sampler_type rb_spr(
        log_posterior,
        num_dims,
        num_walkers,
        scale_a
    );

    size_t const num_samples(100000);
    rng_type rng;
    normal_distribution_type nrm_dist;
    real_matrix_type start_ensemble(num_walkers,num_dims);
    for(std::size_t wkr_i=0; wkr_i < num_walkers; ++wkr_i){
        for(std::size_t dim_j = 0; dim_j < num_dims; ++dim_j){
            start_ensemble(wkr_i,dim_j) = nrm_dist(rng);
        }
    }
    chain_type chn = rb_spr.run_sampler(num_samples,start_ensemble,rng);
    chn.write_samples_to_csv(chn_file_name);

}

BOOST_AUTO_TEST_CASE(classic_hamiltonian) {
    test_red_blue<float>(std::string("float.chain"));
    // test_red_blue<double>(std::string("double.chain"));
    // test_red_blue<long double>(std::string("long-double.chain"));
}
