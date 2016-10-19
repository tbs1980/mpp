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

template<class _real_scalar_t>
class log_post_taylor_2008_2d{
public:
    typedef _real_scalar_t real_scalar_t;
    typedef boost::numeric::ublas::vector<real_scalar_t> real_vector_t;
    typedef boost::numeric::ublas::matrix<real_scalar_t> real_matrix_t;
    typedef boost::numeric::ublas::unbounded_array<real_matrix_t>
        real_matrix_array_t;

    static_assert(
        std::is_floating_point<real_scalar_t>::value,
        "The real scalar is expected to be a floating point type."
    );

    log_post_taylor_2008_2d(
        real_scalar_t const N,
        real_scalar_t const d
    )
    : m_N(N)
    , m_d(d)
    {
        if( m_N < 0 ) {
            std::stringstream msg;
            msg << "The value of N "
                << m_N
                << " should be greater than zero.";
            throw std::domain_error(msg.str());
        }

    }

    real_scalar_t log_posterior(real_vector_t const & x) const {
        real_scalar_t const a  = x(0);
        real_scalar_t const G = x(1);

    }

    real_vector_t grad_log_posterior(real_vector_t const & x) const {
        real_scalar_t const a  = x(0);
        real_scalar_t const G = x(1);
        return -0.5(m_d - a)*(m_d - a)/m_N - 0.5*G - 0.5*a*a*std::exp(-G) + G;
    }

    real_matrix_t metric_tensor_log_posterior(real_vector_t const & x)const {
        real_scalar_t const a  = x(0);
        real_scalar_t const G = x(1);
        real_scalar_t const d_a = (m_d - a)/m_N - a*std::exp(-G);
        real_scalar_t const d_G = -0.5 -0.5*a*a*std::exp(-G) + 1.;
        real_vector_t d_x(2);
        d_x(0) = d_a;
        d_x(1) = d_G;
    }

    real_matrix_array_t deriv_metric_tensor_log_posterior(
        real_vector_t const & x
    ) const {
        real_scalar_t const a  = x(0);
        real_scalar_t const G = x(1);

    }

private:
    real_scalar_t m_N;
    real_scalar_t m_d;
};

BOOST_AUTO_TEST_CASE(rmhmc_banana){
    // test_rmhmc_banana<float>(std::string("banana.float.chain"));
    // test_rmhmc_banana<double>(std::string("banana.double.chain"));
    // test_rmhmc_banana<long double>(std::string("banana.long-double.chain"));
}
