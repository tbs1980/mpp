#define BOOST_TEST_MODULE normal_distribution rmhmc
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
#include <mpp/hamiltonian/reimann_manifold_hmc.hpp>

template<typename  _real_scalar_t>
class normal_distribution{
public:
    typedef _real_scalar_t real_scalar_t;
    typedef boost::numeric::ublas::vector<real_scalar_t> real_vector_t;
    typedef boost::numeric::ublas::matrix<real_scalar_t> real_matrix_t;
    typedef boost::numeric::ublas::unbounded_array<real_matrix_t> real_matrix_array_t;
    typedef std::mt19937 rng_t;
    typedef std::normal_distribution<real_scalar_t> normal_distribution_t;

    normal_distribution(real_scalar_t const mu_fid,
                        real_scalar_t const sigma_fid,
                        std::size_t const num_data_points,
                        std::size_t const random_seed
    )
    :m_mu_fid(mu_fid)
    ,m_sigma_fid(sigma_fid)
    ,m_num_data_points(num_data_points) {
        m_data_x = real_vector_t(m_num_data_points);
        rng_t rng(random_seed);
        normal_distribution_t nrm_dist(m_mu_fid, m_sigma_fid);
        for(std::size_t dim_i = 0; dim_i < m_data_x.size(); ++dim_i){
            m_data_x(dim_i) = nrm_dist(rng);
        }
    }
    ~normal_distribution(){

    }

    real_scalar_t log_posterior(real_vector_t const & x) const {
        real_scalar_t const mu = x(0);
        real_scalar_t const sigma = x(1);
        real_scalar_t log_lik(0);
        for(std::size_t dim_i = 0; dim_i < m_data_x.size(); ++dim_i) {
            real_scalar_t const diff = m_data_x(dim_i) - mu;
            log_lik -= diff*diff;
        }
        log_lik *= 0.5/sigma/sigma;
        log_lik -= m_num_data_points*std::log(sigma);
        return log_lik;
    }

    real_vector_t grad_log_posterior(real_vector_t const & x) const {
        real_scalar_t const mu = x(0);
        real_scalar_t const sigma = x(1);
        real_vector_t d_x(x.size());
        real_scalar_t d_mu = ( sum(m_data_x) - m_num_data_points*mu )/sigma/sigma;
        real_scalar_t d_sigma = 0;
        for(std::size_t dim_i = 0; dim_i < m_data_x.size(); ++dim_i){
            real_scalar_t const diff = m_data_x(dim_i) - mu;
            d_sigma += diff*diff;
        }
        d_sigma *= 1./sigma/sigma/sigma;
        d_sigma -= m_num_data_points/sigma;
        d_x(0) = d_mu;
        d_x(1) = d_sigma;
        return d_x;
    }

    real_matrix_t metric_tensor_log_posterior(real_vector_t const & x)const {
        real_scalar_t const sigma = x(1);
        real_matrix_t G(x.size(),x.size());
        G(0,0) = m_num_data_points/sigma/sigma;
        G(0,1) = 0.;
        G(1,0) = 0.;
        G(1,1) = 2*m_num_data_points/sigma/sigma;
        return G;
    }

    real_matrix_array_t deriv_metric_tensor_log_posterior(real_vector_t const & x ) const {
        using namespace boost::numeric::ublas;
        real_scalar_t const sigma = x(1);
        real_matrix_array_t d_G( x.size(), zero_matrix<real_scalar_t>( x.size(),x.size() ) );
        d_G[1](0,0) = -2.*m_num_data_points/sigma/sigma/sigma;
        d_G[1](0,1) = 0.;
        d_G[1](1,0) = 0.;
        d_G[1](1,1) = -4.*m_num_data_points/sigma/sigma/sigma;
        return d_G;
    }
private:
    real_scalar_t m_mu_fid;
    real_scalar_t m_sigma_fid;
    std::size_t m_num_data_points;
    real_vector_t m_data_x;
};

template<typename real_scalar_t>
void test_normal_distribution_rmhmc(std::string const & chn_file_name){
    using namespace boost::numeric::ublas;
    using namespace mpp::hamiltonian;
    using namespace mpp::chains;

    typedef vector<real_scalar_t> real_vector_t;
    typedef normal_distribution<real_scalar_t> norm_dist_t;
    typedef rm_hmc_sampler<real_scalar_t> rm_hmc_sampler_t;
    typedef typename rm_hmc_sampler_t::log_post_func_t log_post_func_t;
    typedef typename rm_hmc_sampler_t::grad_log_post_func_t grad_log_post_func_t;
    typedef typename rm_hmc_sampler_t::mtr_tnsr_log_post_func_t mtr_tnsr_log_post_func_t;
    typedef typename rm_hmc_sampler_t::der_mtr_tnsr_log_post_func_t der_mtr_tnsr_log_post_func_t;
    typedef std::mt19937 rng_t;
    typedef mcmc_chain<real_scalar_t> chain_t;

    real_scalar_t const mu_fid = 0.;
    real_scalar_t const sigma_fid = 10.;
    std::size_t  const num_data_points = 100;
    std::size_t  const random_seed = 1234;
    norm_dist_t nrm_dst(mu_fid, sigma_fid, num_data_points, random_seed);

    using std::placeholders::_1;
    log_post_func_t log_posterior = std::bind (&norm_dist_t::log_posterior, &nrm_dst, _1);
    grad_log_post_func_t grad_log_posterior = std::bind (&norm_dist_t::grad_log_posterior, &nrm_dst, _1);
    mtr_tnsr_log_post_func_t metric_tensor_log_posterior = std::bind (
            &norm_dist_t::metric_tensor_log_posterior,
            &nrm_dst,
            _1
    );
    der_mtr_tnsr_log_post_func_t deriv_metric_tensor_log_posterior = std::bind(
            &norm_dist_t::deriv_metric_tensor_log_posterior,
            &nrm_dst,
            _1
    );

    std::size_t const num_leap_frog_steps = 5;
    std::size_t const num_fixed_point_steps = 5;
    real_scalar_t const step_size = 0.75;
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
    x_0(0) = 5.;
    x_0(1) = 40.;
    rng_t rng;
    chain_t chn = rm_hmc_spr.run_sampler(num_samples,x_0,rng);
    chn.write_samples_to_csv(chn_file_name);
    std::cout << "acc rate = " << rm_hmc_spr.acc_rate() << std::endl;

}

BOOST_AUTO_TEST_CASE(normal_distribution_rmhmc) {
    test_normal_distribution_rmhmc<float>(std::string("rmhmc_nrm_dist.float.chain"));
}