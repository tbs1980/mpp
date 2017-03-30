#define BOOST_TEST_MODULE var_est gaussian_noise
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/assert.hpp>

#include <cstddef>
#include <random>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include <mpp/hamiltonian/reimann_manifold_hmc.hpp>

template<class _real_scalar_t>
class variance_estimation {
public:
    typedef _real_scalar_t real_scalar_t;
    typedef boost::numeric::ublas::vector<real_scalar_t> real_vector_t;
    typedef boost::numeric::ublas::matrix<real_scalar_t> real_matrix_t;
    typedef boost::numeric::ublas::unbounded_array<real_matrix_t> real_matrix_array_t;
    typedef std::mt19937 rng_t;
    typedef std::normal_distribution<real_scalar_t> normal_distribution_t;

    variance_estimation(
        std::size_t const arg_n,
        real_scalar_t const arg_omega,
        real_scalar_t const arg_zeta,
        std::size_t const arg_seed
    ){
        using namespace boost::numeric::ublas;

        BOOST_ASSERT(arg_n > 0);
        BOOST_ASSERT(arg_omega > 0);
        BOOST_ASSERT(arg_zeta > 0);
        m_n = arg_n;
        m_zeta = arg_zeta;

        // draw {x_i} from a normal distribution with mu=0 and var = arg_omega
        rng_t rng(arg_seed);
        normal_distribution_t nrm_dist(0., 1.);
        m_y = real_vector_t(arg_n);
        for(std::size_t i=0; i<arg_n; ++i){
            m_y(i) = std::sqrt(arg_omega)*nrm_dist(rng)
                + std::sqrt(arg_zeta)*nrm_dist(rng);
        }

    }

    ~variance_estimation(){

    }

    real_scalar_t log_posterior(real_vector_t const & arg_z) const {
        BOOST_ASSERT(arg_z.size() == m_n + 1); // x and omega

        // arg_z -> x and omega
        real_vector_t x(m_n);
        for(std::size_t i = 0; i < m_n; ++i){
            x(i) = arg_z(i);
        }
        real_scalar_t omega = arg_z(m_n);

        // compute the log posterior
        real_scalar_t log_y_x(0);
        for(std::size_t i = 0; i < m_n; ++i) {
            real_scalar_t const diff = (m_y(i) - x(i));
            log_y_x -= diff*diff;
        }
        log_y_x /= (2.*m_zeta);
        real_scalar_t log_x_omega(0);
        for(std::size_t i = 0; i < m_n; ++i) {
            log_x_omega -= x(i)*x(i);
        }
        log_x_omega /= (2.*omega);
        log_x_omega -= 0.5*static_cast<real_scalar_t>(m_n)*std::log(omega);

        return log_y_x + log_x_omega;
    }

    real_vector_t grad_log_posterior(real_vector_t const & arg_z) const {
        BOOST_ASSERT(arg_z.size() == m_n + 1); // x and omega

        // arg_z -> x and omega
        real_vector_t x(m_n);
        for(std::size_t i = 0; i < m_n; ++i){
            x(i) = arg_z(i);
        }
        real_scalar_t omega = arg_z(m_n);

        // compute the derivatives with respect to x
        real_vector_t d_x(m_n);
        for(std::size_t i = 0; i < m_n; ++i) {
            d_x(i) = (m_y(i) - x(i))/m_zeta - x(i)/omega;
        }

        // compute the derivatives with respect to omega
        real_scalar_t d_omega(0);
        for(std::size_t i = 0; i < m_n; ++i) {
            d_omega += x(i)*x(i);
        }
        d_omega /= (2.*omega*omega);
        d_omega -= 0.5*static_cast<real_scalar_t>(m_n)/omega;

        // assign d_x and d_omega to d_z
        real_vector_t d_z(m_n + 1);
        for(std::size_t i = 0; i < m_n; ++i) {
            d_z(i) = d_x(i);
        }
        d_z(m_n) = d_omega;

        return d_z;
    }

    real_matrix_t metric_tensor_log_posterior(
        real_vector_t const & arg_z
    ) const {
        using namespace boost::numeric::ublas;
        BOOST_ASSERT(arg_z.size() == m_n + 1); // x and omega

        // get omega from z
        real_scalar_t const omega = arg_z(m_n);

        // assign the top right block of the metric tensor
        real_matrix_t G = zero_matrix<real_scalar_t>(m_n+1 ,m_n+1);
        for(std::size_t i = 0; i < m_n; ++i) {
            G(i,i) = (1./m_zeta + 1./omega);
        }
        // assign the bottom left blcok of the metric tensor
        G(m_n,m_n) = 0.5*static_cast<real_scalar_t>(m_n)/omega/omega;

        return G;
    }

    real_matrix_array_t deriv_metric_tensor_log_posterior(
        real_vector_t const & arg_z
    ) const {
        using namespace boost::numeric::ublas;
        BOOST_ASSERT(arg_z.size() == m_n + 1); // x and omega

        // get omega from z
        real_scalar_t const omega = arg_z(m_n);

        // assign the derivatives with respect to omega
        real_matrix_array_t d_G(m_n + 1,zero_matrix<real_scalar_t>(m_n + 1,m_n + 1));
        for(std::size_t i = 0; i < m_n; ++i) {
            d_G[m_n](i,i) = -1./omega/omega;
        }
        d_G[m_n](m_n,m_n) = -static_cast<real_scalar_t>(m_n)/omega/omega/omega;

        return d_G;
    }

private:
    real_scalar_t m_n;
    real_scalar_t m_zeta;
    real_vector_t m_y;
};

template<typename real_scalar_t>
void test_var_est_gaussian_noise(std::string const & chn_file_name){
    using namespace boost::numeric::ublas;
    using namespace mpp::hamiltonian;
    using namespace mpp::chains;

    typedef rm_hmc_sampler<real_scalar_t> rm_hmc_sampler_t;
    typedef typename rm_hmc_sampler_t::log_post_func_t log_post_func_t;
    typedef typename rm_hmc_sampler_t::grad_log_post_func_t
        grad_log_post_func_t;
    typedef typename rm_hmc_sampler_t::mtr_tnsr_log_post_func_t
        mtr_tnsr_log_post_func_t;
    typedef typename rm_hmc_sampler_t::der_mtr_tnsr_log_post_func_t
        der_mtr_tnsr_log_post_func_t;
    typedef variance_estimation<real_scalar_t> var_est_t;
    typedef std::mt19937 rng_t;
    typedef boost::numeric::ublas::vector<real_scalar_t> real_vector_t;
    typedef mcmc_chain<real_scalar_t> chain_t;

    // define the posterior distribution
    std::size_t const n = 10;
    real_scalar_t const omega(1.);
    real_scalar_t const zeta(0.01);
    std::size_t const seed = 31415;
    var_est_t var_est(n, omega, zeta, seed);

    // create the functors for RMHMC
    using std::placeholders::_1;
    log_post_func_t log_posterior
        = std::bind (&var_est_t::log_posterior, &var_est, _1);
    grad_log_post_func_t grad_log_posterior
        = std::bind (&var_est_t::grad_log_posterior, &var_est, _1);
    mtr_tnsr_log_post_func_t metric_tensor_log_posterior = std::bind (
            &var_est_t::metric_tensor_log_posterior,
            &var_est,
            _1
    );
    der_mtr_tnsr_log_post_func_t deriv_metric_tensor_log_posterior = std::bind(
            &var_est_t::deriv_metric_tensor_log_posterior,
            &var_est,
            _1
    );

    // define the sampler
    std::size_t const num_leap_frog_steps = 5;
    std::size_t const num_fixed_point_steps = 5;
    real_scalar_t const step_size = 0.75;
    std::size_t const num_dims = n + 1;
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

    // define the number samples required and starting point
    size_t const num_samples(1000);
    real_vector_t x_0(num_dims);
    for(std::size_t i = 0; i < num_dims - 1; ++i){
        x_0(i) = 0.;
    }
    x_0(num_dims - 1) = 1.;
    rng_t rng;
    chain_t chn = rm_hmc_spr.run_sampler(num_samples, x_0, rng);
    std::cout << "acc rate = " << rm_hmc_spr.acc_rate() << std::endl;
    chn.write_samples_to_csv(chn_file_name);
}

BOOST_AUTO_TEST_CASE(variance_estimation_gaussian_noise) {
    test_var_est_gaussian_noise<float>(
        std::string("var_est_gauss_noise_float.chain")
    );
}
