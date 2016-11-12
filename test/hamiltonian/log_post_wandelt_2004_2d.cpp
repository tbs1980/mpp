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
#include <exception>
#include <mpp/dists/multivariate_normal.hpp>
#include <mpp/dists/banana.hpp>
#include <mpp/hamiltonian/reimann_manifold_hmc.hpp>
#include <mpp/hamiltonian/classic_hamiltonian.hpp>

template<class _real_scalar_t>
class log_post_wandelt_2004_2d{
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

    log_post_wandelt_2004_2d(
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
        real_scalar_t const C = x(1);
        // std::cout << x << std::endl;
        // BOOST_ASSERT_MSG(C > 0., "C should be a real number > 0");
//        if(C <= 0. and C > 2000.){
//            // return -std::numeric_limits<real_scalar_t>::max();
//            std::string msg("C out of range. It should be 0<C<20.");
//            throw std::invalid_argument(msg);
//        }
        // if(std::isfinite(a)==false || std::isfinite(C)==false || C <=0. or C>20.){
        //     std::string msg("C out of range. It should be 0<C<20.");
        //     throw std::invalid_argument(msg);
        // }

        if(C<=0){
            return -std::numeric_limits<real_scalar_t>::max();
        }

        return -0.5*(m_d - a)*(m_d - a)/m_N - 0.5*std::log(C) - 0.5*a*a/C;
    }

    real_vector_t grad_log_posterior(real_vector_t const & x) const {
        real_scalar_t const a  = x(0);
        real_scalar_t const C = x(1);
        // BOOST_ASSERT_MSG(C > 0., "C should be a real number > 0");
//        if(C <= 0. and C > 2000.){
//            std::string msg("C out of range. It should be 0<C<20.");
//            throw std::invalid_argument(msg);
//        }
        real_vector_t d_x(2);
        // if(C <= 0. ){
        //     d_x(0) = 0.;
        //     d_x(1) = 0.;
        //     return d_x;
        // }
        real_scalar_t const d_a = (m_d - a)/m_N - a/C;
        real_scalar_t const d_C = -0.5/C + 0.5*a*a/C/C;
        d_x(0) = d_a;
        d_x(1) = d_C;
        return d_x;
    }

    real_matrix_t metric_tensor_log_posterior(real_vector_t const & x)const {
        real_scalar_t const a  = x(0);
        real_scalar_t const C = x(1);
        // BOOST_ASSERT_MSG(C > 0., "C should be a real number > 0");
//        if(C <= 0. and C > 2000.){
//            std::string msg("C out of range. It should be 0<C<20.");
//            throw std::invalid_argument(msg);
//        }
        real_matrix_t mtr_tnsr(2,2);
        // if(C <= 0. ){
        //     mtr_tnsr(0,0) = 1e-20;
        //     mtr_tnsr(1,0) = 0.;
        //     mtr_tnsr(0,1) = 0.;
        //     mtr_tnsr(1,1) = 1e-20;
        //     return mtr_tnsr;
        // }
        mtr_tnsr(0,0) = 1./m_N + 1./C;
        mtr_tnsr(1,0) = 0;//-a/C/C;
        mtr_tnsr(0,1) = 0;//-a/C/C;
        mtr_tnsr(1,1) = 1./C;//a*a/C/C/C - 0.5/C/C;

        return mtr_tnsr;
    }

    real_matrix_array_t deriv_metric_tensor_log_posterior(
        real_vector_t const & x
    ) const {
        real_scalar_t const a  = x(0);
        real_scalar_t const C = x(1);
        // BOOST_ASSERT_MSG(C > 0., "C should be a real number > 0");
//        if(C <= 0. and C > 2000.){
//            std::string msg("C out of range. It should be 0<C<20.");
//            throw std::invalid_argument(msg);
//        }
        real_matrix_array_t drv_mtr_tnsr( x.size(),
            real_matrix_t( x.size(),x.size() )
        );
        // if(C <= 0. ){
        //     drv_mtr_tnsr[0](0,0) = 1e-20;
        //     drv_mtr_tnsr[0](0,1) = 0.;
        //     drv_mtr_tnsr[0](1,0) = 0.;
        //     drv_mtr_tnsr[0](1,1) = 1e-20;
        //
        //     drv_mtr_tnsr[1](0,0) = 1e-20;
        //     drv_mtr_tnsr[1](0,1) = 0.;
        //     drv_mtr_tnsr[1](1,0) = 0.;
        //     drv_mtr_tnsr[1](1,1) = 1e-20;
        //
        //     return drv_mtr_tnsr;
        // }
        drv_mtr_tnsr[0](0,0) = 0.;
        drv_mtr_tnsr[0](0,1) = 0.;//-1./C/C;
        drv_mtr_tnsr[0](1,0) = 0.;//-1./C/C;
        drv_mtr_tnsr[0](1,1) = 0.;//2.*a/C/C/C;

        drv_mtr_tnsr[1](0,0) = -1./C/C;
        drv_mtr_tnsr[1](0,1) = 0.;//2.*a/C/C/C;
        drv_mtr_tnsr[1](1,0) = 0.;//2.*a/C/C/C;
        drv_mtr_tnsr[1](1,1) = -1./C/C;//-3.*a*a/C/C/C/C + 1./C/C/C;

        return drv_mtr_tnsr;

    }

private:
    real_scalar_t m_N;
    real_scalar_t m_d;
};

template<typename real_scalar_t>
void test_log_post_wandelt_2004_hmc(std::string const & chn_file_name){
    using namespace boost::numeric::ublas;
    using namespace mpp::hamiltonian;
    using namespace mpp::chains;

    typedef log_post_wandelt_2004_2d<real_scalar_t> log_post_t;
    typedef vector<real_scalar_t> real_vector_t;
    typedef hmc_sampler<real_scalar_t> hmc_sampler_t;
    typedef vector<real_scalar_t> real_vector_t;
    typedef std::mt19937 rng_t;
    typedef mcmc_chain<real_scalar_t> chain_t;
    typedef typename hmc_sampler_t::log_post_func_type log_post_func_t;
    typedef typename hmc_sampler_t::grad_log_post_func_type grad_log_post_func_t;

    size_t const num_dims(2);
    real_scalar_t const N = 1.;
    real_scalar_t const d = 3.;
    log_post_t  lp_wdt_04(N,d);

    using std::placeholders::_1;
    log_post_func_t log_posterior
            = std::bind (&log_post_t::log_posterior, &lp_wdt_04, _1);

    grad_log_post_func_t grad_log_posterior
            = std::bind (&log_post_t::grad_log_posterior, &lp_wdt_04, _1);
    size_t const max_num_steps(10);
    real_scalar_t const max_eps(1.);
    real_vector_t inv_mass_mat(num_dims);
    inv_mass_mat(0) = 1;//0.87 ;
    inv_mass_mat(1) = 1;//93.95;

    hmc_sampler_t hmc_spr(
            log_posterior,
            grad_log_posterior,
            num_dims,
            max_num_steps,
            max_eps,
            inv_mass_mat
    );

    size_t const num_samples(5000);
    rng_t rng;
    real_vector_t q_0(num_dims);
    q_0(0) = 2.2;
    q_0(1) = 4.8;
    chain_t chn = hmc_spr.run_sampler(num_samples,q_0,rng);
    chn.write_samples_to_csv(chn_file_name);

}

template<typename real_scalar_t>
void test_log_post_wandelt_2004_rmhmc(std::string const & chn_file_name){
    using namespace boost::numeric::ublas;
    using namespace mpp::hamiltonian;
    using namespace mpp::chains;

    typedef log_post_wandelt_2004_2d<real_scalar_t> log_post_t;
    typedef vector<real_scalar_t> real_vector_t;
    typedef rm_hmc_sampler<real_scalar_t> rm_hmc_sampler_t;
    typedef vector<real_scalar_t> real_vector_t;
    typedef std::mt19937 rng_t;
    typedef mcmc_chain<real_scalar_t> chain_t;
    typedef typename rm_hmc_sampler_t::log_post_func_t log_post_func_t;
    typedef typename rm_hmc_sampler_t::grad_log_post_func_t grad_log_post_func_t;
    typedef typename
        rm_hmc_sampler_t::mtr_tnsr_log_post_func_t mtr_tnsr_log_post_func_t;
    typedef typename
        rm_hmc_sampler_t::der_mtr_tnsr_log_post_func_t
            der_mtr_tnsr_log_post_func_t;

    real_scalar_t const N = 1.;
    real_scalar_t const d = 3.;
    log_post_t  lp_wdt_04(N,d);

    using std::placeholders::_1;
    log_post_func_t log_posterior
        = std::bind (&log_post_t::log_posterior, &lp_wdt_04, _1);
    grad_log_post_func_t grad_log_posterior
        = std::bind (&log_post_t::grad_log_posterior, &lp_wdt_04, _1);
    mtr_tnsr_log_post_func_t metric_tensor_log_posterior = std::bind (
        &log_post_t::metric_tensor_log_posterior,
        &lp_wdt_04,
        _1
    );
    der_mtr_tnsr_log_post_func_t deriv_metric_tensor_log_posterior = std::bind(
        &log_post_t::deriv_metric_tensor_log_posterior,
        &lp_wdt_04,
        _1
    );

    std::size_t const num_leap_frog_steps = 5;
    std::size_t const num_fixed_point_steps = 5;
    real_scalar_t const step_size = 0.07;
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

    size_t const num_samples(5000);
    rng_t rng;
    real_vector_t q_0(num_dims);
    q_0(0) = 2.2;
    q_0(1) = 4.8;
    chain_t chn = rm_hmc_spr.run_sampler(num_samples,q_0,rng);
    chn.write_samples_to_csv(chn_file_name);
    std::cout << "acc rate = " << rm_hmc_spr.acc_rate() << std::endl;

}


BOOST_AUTO_TEST_CASE(log_post_wandelt_2004){
     test_log_post_wandelt_2004_hmc<float>(std::string("lp_wdt_04.float.chain"));
//    test_log_post_wandelt_2004_rmhmc<float>(
//        std::string("rmhmc_lp_wdt_04.float.chain")
//    );
}
