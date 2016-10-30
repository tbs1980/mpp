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
#include <cmath>
#include <mpp/dists/multivariate_normal.hpp>
#include <mpp/dists/banana.hpp>
#include <mpp/hamiltonian/reimann_manifold_hmc.hpp>
#include <mpp/hamiltonian/classic_hamiltonian.hpp>

template<class _real_scalar_t>
class log_post_balan_2016_2d {
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

    log_post_balan_2016_2d (
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

    real_scalar_t log_posterior(real_vector_t const & v) const {
        BOOST_ASSERT_MSG(
            v.size() == 2,
            "v should be a two dimensional vector."
        );
        real_scalar_t const x = v(0);
        real_scalar_t const F = v(1);
        BOOST_ASSERT_MSG(F > 1., "F should be a positive real number > 1");
        real_scalar_t const log_F = std::log(F);

        real_scalar_t const log_post =
            -0.5*(m_d - log_F*x)*(m_d - log_F*x)
            -0.5*log_F*x*x
            +0.5*std::log(log_F)
            -log_F;
        return log_post;
    }

    real_vector_t grad_log_posterior(real_vector_t const & v) const {
        BOOST_ASSERT_MSG(
            v.size() == 2,
            "v should be a two dimensional vector."
        );
        real_scalar_t const x = v(0);
        real_scalar_t const F = v(1);
        BOOST_ASSERT_MSG(F > 1., "F should be a positive real number > 1");
        real_scalar_t const log_F = std::log(F);
        real_vector_t d_v(2);
        real_scalar_t const d_x = (m_d - log_F*x)*log_F/m_N - log_F*x;
        real_scalar_t const d_F = (m_d - log_F*x)*x/m_N/F - 0.5*x*x/F
            +0.5/log_F/F - 1./F;
        d_v(0) = d_x;
        d_v(1) = d_F;
        return d_v;
    }

    real_matrix_t metric_tensor_log_posterior(real_vector_t const & v)const {
        BOOST_ASSERT_MSG(
            v.size() == 2,
            "v should be a two dimensional vector."
        );
        real_scalar_t const x = v(0);
        real_scalar_t const F = v(1);
        BOOST_ASSERT_MSG(F > 1., "F should be a positive real number > 1");
        real_scalar_t const log_F = std::log(F);
        real_matrix_t G(2,2);
        G(0,0) = log_F*log_F;
        G(0,1) = log_F*x/F;
        G(1,0) = log_F*x/F;
        G(1,1) = x*x/F/F;
        G /= m_N;
        G(0,0) += log_F;
        G(0,1) += x/F;
        G(1,0) += x/F;
        G(1,1) += 0.5/F/F/log_F/log_F + 0.5/F/F/log_F - 0.5*x*x/F/F - 1./F/F;

        std::cout << 'v = ' << v << std::endl;
        std::cout << "G = " << G << std::endl;

        return G;
    }

    real_matrix_array_t deriv_metric_tensor_log_posterior(
        real_vector_t const & v
    ) const {
        BOOST_ASSERT_MSG(
            v.size() == 2,
            "v should be a two dimensional vector."
        );
        using namespace boost::numeric::ublas;
        real_scalar_t const x = v(0);
        real_scalar_t const F = v(1);
        BOOST_ASSERT_MSG(F > 1., "F should be a positive real number > 1");
        real_scalar_t const log_F = std::log(F);
        real_matrix_array_t d_G( v.size(),
            zero_matrix<real_scalar_t>( v.size(),v.size() )
        );
        d_G[0](0,0) = 0.;
        d_G[0](0,1) = log_F/F;
        d_G[0](1,0) = log_F/F;
        d_G[0](1,1) = 2.*x/F/F;
        d_G[0] /= m_N;
        d_G[0](0,0) += 0.;
        d_G[0](0,1) += 1./F;
        d_G[0](1,0) += 1./F;
        d_G[0](1,1) += -x/F/F;

        d_G[1](0,0) = 2.*log_F/F;
        d_G[1](0,1) = x/F/F - log_F*x/F/F;
        d_G[1](1,0) = x/F/F - log_F*x/F/F;
        d_G[1](1,1) = -2.*x*x/F/F/F;
        d_G[1] /= m_N;
        d_G[1](0,0) += 1./F;
        d_G[1](0,1) += -x/F/F;
        d_G[1](1,0) += -x/F/F;
        d_G[1](1,1) += -1./F/F/F/log_F/log_F/log_F -1.5/F/F/F/log_F/log_F
            -1./F/F/F/log_F + x*x/F/F/F + 2./F/F/F;
        return d_G;
    }

private:
    real_scalar_t m_N;
    real_scalar_t m_d;
};

template<typename real_scalar_t>
void test_log_post_balan_2016_rmhmc(std::string const & chn_file_name){
    using namespace mpp::dists;
    using namespace boost::numeric::ublas;
    using namespace mpp::hamiltonian;
    using namespace mpp::chains;

    typedef log_post_balan_2016_2d<real_scalar_t> log_post_t;
    typedef vector<real_scalar_t> real_vector_t;
    typedef rm_hmc_sampler<real_scalar_t> rm_hmc_sampler_t;
    typedef typename rm_hmc_sampler_t::log_post_func_t log_post_func_t;
    typedef typename rm_hmc_sampler_t::grad_log_post_func_t grad_log_post_func_t;
    typedef typename
        rm_hmc_sampler_t::mtr_tnsr_log_post_func_t mtr_tnsr_log_post_func_t;
    typedef typename
        rm_hmc_sampler_t::der_mtr_tnsr_log_post_func_t
            der_mtr_tnsr_log_post_func_t;
    typedef std::mt19937 rng_t;
    typedef mcmc_chain<real_scalar_t> chain_t;

    real_scalar_t const d = 3.;
    real_scalar_t const N = 1.;

    log_post_t lp_balan_16(d,N);
    using std::placeholders::_1;
    log_post_func_t log_posterior
        = std::bind (&log_post_t::log_posterior, &lp_balan_16, _1);
    grad_log_post_func_t grad_log_posterior
        = std::bind (&log_post_t::grad_log_posterior, &lp_balan_16, _1);
    mtr_tnsr_log_post_func_t metric_tensor_log_posterior = std::bind (
        &log_post_t::metric_tensor_log_posterior,
        &lp_balan_16,
        _1
    );
    der_mtr_tnsr_log_post_func_t deriv_metric_tensor_log_posterior = std::bind(
        &log_post_t::deriv_metric_tensor_log_posterior,
        &lp_balan_16,
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
    x_0(0) = 1.1;
    x_0(1) = 4.5;
    chain_t chn = rm_hmc_spr.run_sampler(num_samples,x_0,rng);
    chn.write_samples_to_csv(chn_file_name);
    std::cout << "acc rate = " << rm_hmc_spr.acc_rate() << std::endl;

}

BOOST_AUTO_TEST_CASE(lp_balan_2016_2d_rmhmc){
    test_log_post_balan_2016_rmhmc<float>(std::string("lp_blalan_16.float.chain"));
    // test_rmhmc_banana<double>(std::string("banana.double.chain"));
    // test_rmhmc_banana<long double>(std::string("banana.long-double.chain"));
}
