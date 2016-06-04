#define BOOST_TEST_MODULE mcmc chain
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <random>
#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <mpp/chains/mcmc_chain.hpp>

template<typename real_scalar_type>
void test_mcmc_chain(std::string const & file_name)
{
    using namespace mpp::chains;
    using namespace boost::numeric::ublas;
    typedef vector<real_scalar_type> real_vector_type;
    typedef mcmc_chain<real_scalar_type> mcmc_chain_type;
    typedef std::mt19937 rng_type;
    typedef std::normal_distribution<real_scalar_type> normal_distribution_type;

    size_t const num_dims = 3;
    size_t const num_samples = 10;
    mcmc_chain_type mc_chn(num_samples,num_dims);
    real_vector_type q_0(num_dims);
    rng_type rng;
    normal_distribution_type norm_dist;
    for(size_t i=0;i<num_samples;++i)
    {
        for(size_t j=0;j<num_dims;++j)
        {
            q_0(j) = norm_dist(rng);
        }
        mc_chn.set_sample(i,q_0,real_scalar_type(1));
    }
    mc_chn.write_samples_to_csv(file_name);

}

BOOST_AUTO_TEST_CASE(mcmc_chain)
{
    test_mcmc_chain<float>(std::string("float.chain"));
    test_mcmc_chain<double>(std::string("double.chain"));
    test_mcmc_chain<long double>(std::string("long-double.chain"));
}
