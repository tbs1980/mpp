#define BOOST_TEST_MODULE var_est gaussian_noise
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/assert.hpp>

#include <cstddef>
#include <random>
#include <cmath>

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
        d_x(m_n) = d_omega;

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
            for(std::size_t j = 0; j< m_n; ++j){
                G(i,j) = (1./m_zeta + 1./omega);
            }
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
void test_var_est_gaussian_noise(){

    std::size_t const n = 1000;
    real_scalar_t const omega(1.);
    real_scalar_t const zeta(1.);
    std::size_t const seed = 31415;
    variance_estimation<real_scalar_t> var_est(n, omega, zeta, seed);
}

BOOST_AUTO_TEST_CASE(variance_estimation_gaussian_noise) {
    test_var_est_gaussian_noise<float>();
}
