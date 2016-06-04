#define BOOST_TEST_MODULE kinetic energy multivar normal
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <random>
#include <limits>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/test/unit_test.hpp>
#include <mpp/hamiltonian/kinetic_energy_multivar_normal.hpp>

template<typename real_scalar_type>
void test_multivariate_normal_diag()
{
    using namespace mpp::hamiltonian;
    using namespace boost::numeric::ublas;

    typedef multivariate_normal<real_scalar_type> multivariate_normal_type;
    typedef vector<real_scalar_type> real_vector_type;
    typedef std::mt19937 rng_type;

    std::size_t const num_dims = 1000000;

    real_vector_type sigma_inv(num_dims);
    for(std::size_t i=0;i<num_dims;++i)
    {
        sigma_inv(i) = real_scalar_type(1);
    }

    multivariate_normal_type mlt_nr(sigma_inv);

    // compute the log_posterior for a ones vector
    real_vector_type p(num_dims);
    for(std::size_t i=0;i<num_dims;++i)
    {
        p(i) = real_scalar_type(1);
    }

    real_scalar_type log_posterior_c = mlt_nr.log_posterior(p);

    real_scalar_type log_posterior_e(0);
    for(std::size_t i=0;i<num_dims;++i)
    {
        log_posterior_e -= p(i)*p(i)*sigma_inv(i);
    }
    log_posterior_e *= real_scalar_type(0.5);

    real_scalar_type eps = std::numeric_limits<real_scalar_type>::epsilon();
    BOOST_CHECK(
        std::abs(log_posterior_c - log_posterior_e) <= eps
    );

    real_vector_type grad_p_c = mlt_nr.grad_log_posterior(p);

    real_vector_type grad_p_e(num_dims);
    for(std::size_t i=0;i<num_dims;++i)
    {
        grad_p_e(i) = -sigma_inv(i)*p(i);
    }

    for(std::size_t i=0;i<num_dims;++i)
    {
        BOOST_CHECK(
            std::abs(grad_p_c(i) - grad_p_e(i) ) <= eps
        );
    }

    rng_type rng;
    real_vector_type samp_p = mlt_nr.generate_sample(rng);

    real_scalar_type sum(0);
    real_scalar_type sum2(0);
    for(std::size_t i=0;i<num_dims;++i)
    {
        sum += samp_p(i);
        sum2 += samp_p(i)*samp_p(i);
    }
    real_scalar_type mean = sum/real_scalar_type(num_dims);
    real_scalar_type std = std::sqrt( sum2/real_scalar_type(num_dims)
        - mean*mean );

    BOOST_CHECK(
        std::abs(mean) <=
            real_scalar_type(1)/std::sqrt(real_scalar_type(num_dims))
    );

    BOOST_CHECK(
        std::abs( std - real_scalar_type(1) ) <= real_scalar_type(1)/std::sqrt(real_scalar_type(num_dims))
    );

}

BOOST_AUTO_TEST_CASE(kinetic_energy_multivar_normal_diag)
{
    test_multivariate_normal_diag<float>();
    test_multivariate_normal_diag<double>();
    test_multivariate_normal_diag<long double>();
}
