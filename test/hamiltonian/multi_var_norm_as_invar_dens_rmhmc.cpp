#define BOOST_TEST_MODULE multi_var_normal_distribution rmhmc
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
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
class multivariate_normal{
public:
    typedef _real_scalar_t real_scalar_t;
    typedef boost::numeric::ublas::vector<real_scalar_t> real_vector_t;
    typedef boost::numeric::ublas::matrix<real_scalar_t> real_matrix_t;
    typedef boost::numeric::ublas::unbounded_array<real_matrix_t> real_matrix_array_t;
    typedef std::mt19937 rng_t;
    typedef std::normal_distribution<real_scalar_t> normal_distribution_t;

    multivariate_normal(
        real_vector_t const & mu_fid,
        real_matrix_t const & sigma_fid,
        std::size_t const num_data_points,
        std::size_t const random_seed
    )
    : m_mu_fid(mu_fid)
    , m_sigma_fid(sigma_fid)
    , m_num_data_points(num_data_points) {
        using namespace mpp::utils;
        using namespace boost::numeric::ublas;

        BOOST_ASSERT(m_mu_fid.size() == 2);
        BOOST_ASSERT(m_mu_fid.size() == m_sigma_fid.size1());
        BOOST_ASSERT(m_sigma_fid.size1() == m_sigma_fid.size2());

        m_data_x = real_matrix_t(m_num_data_points, m_mu_fid.size());
        real_matrix_t chol_sigma_fid(m_sigma_fid.size1(), m_sigma_fid.size2());
        std::size_t const has_chol
            = cholesky_decompose<real_scalar_t>(m_sigma_fid, chol_sigma_fid);
        BOOST_ASSERT(has_chol == 0);

        rng_t rng(random_seed);
        normal_distribution_t nrm_dist(0., 1.);
        for(std::size_t dim_i = 0; dim_i < m_data_x.size1(); ++dim_i) {
            real_vector_t data_row(m_mu_fid.size());
            for(std::size_t dim_j = 0; dim_j < data_row.size(); ++dim_j) {
                data_row(dim_j) = nrm_dist(rng);
            }

            data_row = prod(data_row, chol_sigma_fid);
            data_row += m_mu_fid;
            for(std::size_t dim_j = 0; dim_j < m_data_x.size2(); ++dim_j) {
                m_data_x(dim_i, dim_j) = data_row(dim_j);
            }
        }

        m_dup_mat_m = duplication_matrix(m_mu_fid.size());
        m_dup_mat_mm = duplication_matrix(m_mu_fid.size()*m_mu_fid.size());
        m_comm_mat_mm = commutation_matrix(m_mu_fid.size()*m_mu_fid.size(), m_mu_fid.size()*m_mu_fid.size());
        m_dm_T_kron_dm_T = kron(trans(m_dup_mat_m), trans(m_dup_mat_m));
        m_mat_I_m = identity_matrix<real_scalar_t>(m_mu_fid.size());
        m_mat_I_m2 = identity_matrix<real_scalar_t>(m_mu_fid.size()*m_mu_fid.size());
        m_Im_kron_Kmm_kron_Im = kron(m_mat_I_m, kron(m_comm_mat_mm, m_mat_I_m) );
        m_dm_T_kron_dm_T_prod_Im_kron_Kmm_kron_Im = prod(m_dm_T_kron_dm_T, m_Im_kron_Kmm_kron_Im);
    }

    ~multivariate_normal(){

    }

    real_scalar_t log_posterior(real_vector_t const & arg_x) const {
        // TODO we need refer to the equations in the paper
        using namespace boost::numeric::ublas;
        using namespace mpp::utils;

        BOOST_ASSERT( arg_x.size() == m_mu_fid.size() + m_sigma_fid.size1()*(m_sigma_fid.size1()+1)/2 );

        // convert the argument to mu and sigma
        real_vector_t mu(m_mu_fid.size());
        std::size_t ind_i = 0;
        for(std::size_t dim_i = 0; dim_i < mu.size(); ++dim_i) {
            mu(dim_i) = arg_x(ind_i);
            ++ind_i;
        }//TODO we need to copy this in a better way
        real_matrix_t sigma(m_sigma_fid.size1(), m_sigma_fid.size2());
        for(std::size_t dim_i = 0; dim_i < m_sigma_fid.size1(); ++ dim_i) {
            for(std::size_t dim_j = dim_i; dim_j < m_sigma_fid.size2(); ++dim_j) {
                sigma(dim_i, dim_j) = arg_x(ind_i);
                sigma(dim_j, dim_i) = arg_x(ind_i);
                ++ind_i;
            }
        }//TODO we need to copy this in a better way

        // compute the determinant of sigma and check for positive definiteness
        real_scalar_t const det_sigma
            = compute_determinant<real_scalar_t>(sigma);
        if(det_sigma <= 0) {
            return -std::numeric_limits<real_scalar_t>::max();
        }

        // find the the inverse of the sigma
        real_matrix_t sigma_inv(sigma.size1(), sigma.size2());
        bool has_inv = compute_inverse<real_scalar_t>(sigma, sigma_inv);
        if(has_inv == false){
            return -std::numeric_limits<real_scalar_t>::max();
        }

        // compute the matrix S
        real_matrix_t mat_s
            = zero_matrix<real_scalar_t>(mu.size(), mu.size());
        for(std::size_t dim_i = 0; dim_i < m_num_data_points; ++ dim_i) {
            real_vector_t diff_x_mu(mu.size());
            for(std::size_t dim_j = 0; dim_j < mu.size(); ++dim_j){
                diff_x_mu(dim_j) = m_data_x(dim_i, dim_j) - mu(dim_j);
            }
            mat_s += outer_prod(diff_x_mu, diff_x_mu);
        }

        // compute the log-posterior
        real_scalar_t const log_post = -real_scalar_t(0.5)*(
            m_num_data_points*std::log(det_sigma)
            + compute_trace<real_scalar_t>(prod(sigma_inv, mat_s))
        );

        return log_post;
    }

    real_vector_t grad_log_posterior(real_vector_t const & arg_x) const {
        // TODO we need refer to the equations in the paper
        using namespace boost::numeric::ublas;
        using namespace mpp::utils;

        BOOST_ASSERT( arg_x.size() == m_mu_fid.size() + m_sigma_fid.size1()*(m_sigma_fid.size1()+1)/2 );

        // convert the argument to mu and sigma
        real_vector_t mu(m_mu_fid.size());
        std::size_t ind_ms_i = 0;
        for(std::size_t dim_i = 0; dim_i < mu.size(); ++dim_i) {
            mu(dim_i) = arg_x(ind_ms_i);
            ++ind_ms_i;
        } // TODO we need to copy this in a better way
        real_matrix_t sigma(m_sigma_fid.size1(), m_sigma_fid.size2());
        for(std::size_t dim_i = 0; dim_i < m_sigma_fid.size1(); ++ dim_i) {
            for(std::size_t dim_j = dim_i; dim_j < m_sigma_fid.size2(); ++dim_j) {
                sigma(dim_i, dim_j) = arg_x(ind_ms_i);
                sigma(dim_j, dim_i) = arg_x(ind_ms_i);
                ++ind_ms_i;
            }
        } // TODO we need to copy this in a better way

        // find the inverse of sigma
        real_matrix_t sigma_inv(sigma.size1(), sigma.size2());
        bool has_inv = compute_inverse<real_scalar_t>(sigma, sigma_inv);
        BOOST_ASSERT(has_inv == true);

        // compute the matrix S
        real_matrix_t mat_s
            = zero_matrix<real_scalar_t>(mu.size(), mu.size());
        for(std::size_t dim_i = 0; dim_i < m_num_data_points; ++ dim_i) {
            real_vector_t diff_x_mu(mu.size());
            for(std::size_t dim_j = 0; dim_j < mu.size(); ++dim_j){
                diff_x_mu(dim_j) = m_data_x(dim_i, dim_j) - mu(dim_j);
            }
            mat_s += outer_prod(diff_x_mu, diff_x_mu);
        }

        // compute the vector y_bar
        real_vector_t y_bar = zero_vector<real_scalar_t>(mu.size());
        for(std::size_t dim_i = 0; dim_i < m_num_data_points; ++ dim_i) {
            real_vector_t data_row(mu.size());
            for(std::size_t dim_j = 0; dim_j < mu.size(); ++dim_j) {
                data_row(dim_j) = m_data_x(dim_i, dim_j);
            }
            y_bar += data_row;
        }
        y_bar /= static_cast<real_scalar_t>(m_num_data_points);

        // compute the gradient of the log-posterior wrt mu d L/ d mu^T
        real_vector_t const d_mu = prod(sigma_inv, (y_bar - mu));

        // compute vec(S - n*Sigma)
        real_matrix_t const diff_s_n_sigma = mat_s - m_num_data_points*sigma;
        real_vector_t vec_diff_s_n_sigma(sigma.size1()*sigma.size1());
        std::size_t ind_vs_i = 0;
        for(std::size_t dim_i = 0; dim_i < sigma.size1(); ++dim_i) {
            for(std::size_t dim_j = 0; dim_j < sigma.size2(); ++dim_j) {
                vec_diff_s_n_sigma(ind_vs_i) = diff_s_n_sigma(dim_i, dim_j);
                ++ind_vs_i;
            }
        } // TODO we need to copy this in a better way

        // compute Sigma^-1 kron Sigma^-1
        real_matrix_t const sig_inv_kron_sig_inv = kron(sigma_inv, sigma_inv);

        // copute the gradient wrt to vech(Sigma) ie d L / d vech(Sigma)^T
        real_vector_t const d_vech_sigma = prod(
            trans(m_dup_mat_m),
            prod<real_vector_t>(sig_inv_kron_sig_inv, vec_diff_s_n_sigma)
        );

        // copy back to return gradeint vector
        real_vector_t d_arg_x(arg_x.size());
        std::size_t ind_dx_i = 0;
        for(std::size_t dim_i = 0; dim_i < d_mu.size(); ++dim_i) {
            d_arg_x(ind_dx_i) = d_mu(dim_i);
            ++ind_dx_i;
        }
        for(std::size_t dim_i = 0; dim_i < d_vech_sigma.size(); ++dim_i) {
            d_arg_x(ind_dx_i) = d_vech_sigma(dim_i);
            ++ind_dx_i;
        }

        return d_arg_x;
    }

    real_matrix_t metric_tensor_log_posterior(real_vector_t const & arg_x) const {
        // TODO we need refer to the equations in the paper
        using namespace boost::numeric::ublas;
        using namespace mpp::utils;

        BOOST_ASSERT( arg_x.size() == m_mu_fid.size() + m_sigma_fid.size1()*(m_sigma_fid.size1()+1)/2 );

        // convert the argument to mu and sigma
        real_vector_t mu(m_mu_fid.size());
        std::size_t index_i = 0;
        for(std::size_t dim_i = 0; dim_i < mu.size(); ++dim_i) {
            mu(dim_i) = arg_x(index_i);
            ++index_i;
        } // TODO we need to copy this in a better way
        real_matrix_t sigma(m_sigma_fid.size1(), m_sigma_fid.size2());
        for(std::size_t dim_i = 0; dim_i < m_sigma_fid.size1(); ++ dim_i) {
            for(std::size_t dim_j = dim_i; dim_j < m_sigma_fid.size2(); ++dim_j) {
                sigma(dim_i, dim_j) = arg_x(index_i);
                sigma(dim_j, dim_i) = arg_x(index_i);
                ++index_i;
            }
        } // TODO we need to copy this in a better way

        // compute the ivnerse of Sigma
        real_matrix_t sigma_inv(sigma.size1(), sigma.size2());
        bool const has_inv = compute_inverse<real_scalar_t>(sigma, sigma_inv);
        BOOST_ASSERT(has_inv == true);

        // compute Sigma^-1 kron Sigma^-1
        real_matrix_t const sig_inv_kron_sig_inv = kron(sigma_inv, sigma_inv);

        real_matrix_t mtrc_tnsr_G
            = zero_matrix<real_scalar_t>(arg_x.size(), arg_x.size());

        // create the top-left block of G
        for(std::size_t dim_i = 0; dim_i < sigma_inv.size1(); ++dim_i) {
            for(std::size_t dim_j = 0; dim_j < sigma_inv.size2(); ++dim_j) {
                mtrc_tnsr_G(dim_i, dim_j)
                    = m_num_data_points*sigma_inv(dim_i, dim_j);
            }
        }

        real_matrix_t dm_t_sig_inv_kron_sig_inv_dm
            = prod(trans(m_dup_mat_m), real_matrix_t( prod(sig_inv_kron_sig_inv, m_dup_mat_m)));

        BOOST_ASSERT(dm_t_sig_inv_kron_sig_inv_dm.size1() == dm_t_sig_inv_kron_sig_inv_dm.size2());

        // create the bottom-left block of G
        for(std::size_t dim_i = 0; dim_i < dm_t_sig_inv_kron_sig_inv_dm.size1(); ++dim_i) {
            for(std::size_t dim_j = 0; dim_j < dm_t_sig_inv_kron_sig_inv_dm.size2(); ++dim_j) {
                mtrc_tnsr_G(dim_i+sigma_inv.size1(), dim_j+sigma_inv.size2())
                    = m_num_data_points*real_scalar_t(0.5)
                        *dm_t_sig_inv_kron_sig_inv_dm(dim_i, dim_j);
            }
        }

        return mtrc_tnsr_G;

    }

    real_matrix_array_t deriv_metric_tensor_log_posterior(real_vector_t const & arg_x ) const {
        // TODO we need refer to the equations in the paper
        using namespace boost::numeric::ublas;
        using namespace mpp::utils;

        BOOST_ASSERT( arg_x.size() == m_mu_fid.size() + m_sigma_fid.size1()*(m_sigma_fid.size1()+1)/2 );

        // convert the argument to mu and sigma
        real_vector_t mu(m_mu_fid.size());
        std::size_t index_i = 0;
        for(std::size_t dim_i = 0; dim_i < mu.size(); ++dim_i) {
            mu(dim_i) = arg_x(index_i);
            ++index_i;
        } // TODO we need to copy this in a better way
        real_matrix_t sigma(m_sigma_fid.size1(), m_sigma_fid.size2());
        for(std::size_t dim_i = 0; dim_i < m_sigma_fid.size1(); ++ dim_i) {
            for(std::size_t dim_j = dim_i; dim_j < m_sigma_fid.size2(); ++dim_j) {
                sigma(dim_i, dim_j) = arg_x(index_i);
                sigma(dim_j, dim_i) = arg_x(index_i);
                ++index_i;
            }
        } // TODO we need to copy this in a better way

        // compute the ivnerse of Sigma
        real_matrix_t sigma_inv(sigma.size1(), sigma.size2());
        bool has_inv = compute_inverse<real_scalar_t>(sigma, sigma_inv);
        BOOST_ASSERT(has_inv == true);

        // compute Sigma^-1 kron Sigma^-1
        real_matrix_t const sig_inv_kron_sig_inv = kron(sigma_inv, sigma_inv);

        // compute the matrix theta
        real_matrix_t const theta
            = -real_scalar_t(m_num_data_points)*prod(sig_inv_kron_sig_inv, m_dup_mat_m);

        // compute vec(Sigma^-1)
        real_matrix_t const vec_sigma_inv
            = real_matrix_t(sigma.size1()*sigma.size2(), std::size_t(1));
        std::size_t ind_vsi=0;
        for(std::size_t dim_j = 0; dim_j < sigma_inv.size1(); ++dim_j){
            for(std::size_t dim_i = 0; dim_i < sigma_inv.size2(); ++dim_i){
                vec_sigma_inv(ind_vsi,0) = vec_sigma_inv(dim_i, dim_j);
                ++ind_vsi;
            }
        }

        // compute I_m2 kron vec(Sigma^-1) + vec(Sigma^-1) kron I_m2
        real_matrix_t Im2_kron_vec_sig_inv_plus_vec_sig_inv_kron_Im2
            = kron(m_mat_I_m2, vec_sigma_inv);
        Im2_kron_vec_sig_inv_plus_vec_sig_inv_kron_Im2
            += kron(vec_sigma_inv, m_mat_I_m2);

        real_matrix_t temp_mat_1
            = -real_scalar_t(0.5)*m_dm_T_kron_dm_T_prod_Im_kron_Kmm_kron_Im;
        real_matrix_t temp_mat_2
            = prod(Im2_kron_vec_sig_inv_plus_vec_sig_inv_kron_Im2, theta);

        real_matrix_array_t der_mtrc_tnsr_G( arg_x.size(), zero_matrix<real_scalar_t>( arg_x.size(), arg_x.size() ) );

        return der_mtrc_tnsr_G;
    }

    real_matrix_t duplication_matrix(std::size_t const dim_n) const {
        using namespace boost::numeric::ublas;
        BOOST_ASSERT(dim_n > 0);

        std::size_t const num_dup_mat_cols = dim_n*(dim_n+1)/2;
        // std::size_t const uvh_size = (size_t) 0.5*(-1. + std::sqrt(1 + 8*num_dup_mat_cols));
        std::size_t const uvh_size = dim_n;

        real_matrix_t dup_mat(dim_n*dim_n, dim_n*(dim_n+1)/2);

        real_matrix_t temp_mat = identity_matrix<real_scalar_t>(num_dup_mat_cols);
        real_vector_t row_x(temp_mat.size1());
        for(std::size_t dim_i = 0; dim_i < temp_mat.size1(); ++dim_i) {
            for(std::size_t dim_j = 0; dim_j < temp_mat.size2(); ++dim_j) {
                row_x(dim_j) = temp_mat(dim_i, dim_j);
            }

            real_matrix_t res_mat = zero_matrix<real_scalar_t>(uvh_size, uvh_size);
            std::size_t index = 0;
            for(size_t ind_i = 0; ind_i < res_mat.size1(); ++ind_i) {
                for(std::size_t ind_j = ind_i ; ind_j < res_mat.size2(); ++ind_j) {
                    res_mat(ind_i, ind_j) = row_x(index);
                    ++index;
                }
            }
            res_mat = res_mat + trans(res_mat);
            for(size_t ind_i = 0; ind_i < res_mat.size1(); ++ind_i) {
                res_mat(ind_i, ind_i) *= 0.5;
            }

            index = 0;
            for(size_t ind_i = 0; ind_i < res_mat.size1(); ++ind_i) {
                for(std::size_t ind_j = 0 ; ind_j < res_mat.size2(); ++ind_j) {
                    dup_mat(index ,dim_i) = res_mat(ind_i, ind_j);
                    ++index;
                }
            }
        }

        return dup_mat;
    }

    real_matrix_t commutation_matrix(std::size_t const dim_p, std::size_t const dim_q) {
        using namespace boost::numeric::ublas;
        BOOST_ASSERT(dim_p > 0);
        BOOST_ASSERT(dim_q > 0);
        real_matrix_t comm_mat(dim_p*dim_q, dim_p*dim_q);

        real_matrix_t mat_k = identity_matrix<real_scalar_t>(dim_p*dim_q);
        real_matrix_t ind_mat(dim_p, dim_q);
        std::size_t index = 0;
        for(std::size_t dim_j =0; dim_j < ind_mat.size2(); ++dim_j) {
            for(std::size_t dim_i = 0; dim_i < ind_mat.size1(); ++dim_i) {
                ind_mat(dim_i, dim_j) = static_cast<real_scalar_t>(index);
                ++index;
            }
        }

        real_vector_t indices( ind_mat.size1()*ind_mat.size2() );
        index = 0;
        for(std::size_t dim_i = 0; dim_i < ind_mat.size1(); ++dim_i) {
            for(std::size_t dim_j =0; dim_j < ind_mat.size2(); ++dim_j) {
                indices(index) = ind_mat(dim_i, dim_j);
                ++index;
            }
        }

        for(std::size_t dim_i = 0; dim_i < mat_k.size1(); ++dim_i) {
            for(std::size_t dim_j =0; dim_j < mat_k.size2(); ++dim_j) {
                comm_mat(dim_i, dim_j) = mat_k(static_cast<std::size_t>(indices(dim_i)) , dim_j);
            }
        }

        return comm_mat;
    }

private:
    real_vector_t m_mu_fid;
    real_matrix_t m_sigma_fid;
    std::size_t m_num_data_points;
    real_matrix_t m_data_x;
    real_matrix_t m_dup_mat_m;
    real_matrix_t m_dup_mat_mm;
    real_matrix_t m_comm_mat_mm;
    real_matrix_t m_dm_T_kron_dm_T;
    real_matrix_t m_Im_kron_Kmm_kron_Im;
    real_matrix_t m_dm_T_kron_dm_T_prod_Im_kron_Kmm_kron_Im
    real_matrix_t m_mat_I_m;
    real_matrix_t m_mat_I_m2;
};

template<typename real_scalar_t>
void test_multivariate_normal(std::string const & chn_file_name) {
    using namespace boost::numeric::ublas;

    typedef vector<real_scalar_t> real_vector_t;
    typedef matrix<real_scalar_t> real_matrix_t;
    typedef multivariate_normal<real_scalar_t> multivariate_normal_t;
    typedef boost::numeric::ublas::unbounded_array<real_matrix_t> real_matrix_array_t;

    std::size_t const num_dims = 2;
    real_vector_t const mu_fid = zero_vector<real_scalar_t>(num_dims);
    real_matrix_t const sigma_fid = identity_matrix<real_scalar_t>(num_dims);
    std::size_t const num_data_points = 1;
    std::size_t const random_seed = 31415;
    multivariate_normal_t mvnrm(mu_fid, sigma_fid, num_data_points, random_seed);

    real_vector_t arg_x(num_dims + num_dims*(num_dims+1)/2);
    std::size_t ind_i = 0;
    for(std::size_t dim_i = 0; dim_i < mu_fid.size(); ++dim_i){
        arg_x(ind_i) = mu_fid(dim_i);
        ++ind_i;
    }
    for(std::size_t dim_i = 0; dim_i < sigma_fid.size1(); ++dim_i) {
        for(std::size_t dim_j = dim_i; dim_j < sigma_fid.size2(); ++dim_j) {
            arg_x(ind_i) = sigma_fid(dim_i, dim_j);
            ++ind_i;
        }
    }

    real_scalar_t const log_post_val = mvnrm.log_posterior(arg_x);
    std::cout <<"log_post_val = " << std::endl;
    real_vector_t const d_arg_x = mvnrm.grad_log_posterior(arg_x);
    real_matrix_t const mtrc_tnsr_G = mvnrm.metric_tensor_log_posterior(arg_x);

    real_matrix_t dup_mat = mvnrm.duplication_matrix(3);
    std::cout << dup_mat << std::endl;

    real_matrix_t comm_mat = mvnrm.commutation_matrix(3,2);
    std::cout << "\n" << comm_mat << std::endl;

    real_matrix_array_t der_mtrc_tnsr_G = mvnrm.deriv_metric_tensor_log_posterior(arg_x);
}

template<typename real_scalar_t>
void test_kron_prod() {
    using namespace boost::numeric::ublas;
    using namespace mpp::utils;
    typedef matrix<real_scalar_t> real_matrix_t;

    real_matrix_t mat_A(2,2);
    real_matrix_t mat_B(2,2);

    std::size_t ind = 0;
    for(std::size_t i=0; i < 2; ++i){
        for(std::size_t j=0; j < 2; ++j){
            mat_A(i,j) = static_cast<real_scalar_t>(ind);
            mat_B(i,j) = 1.;
            ++ind;
        }
    }

    std::cout << mat_A << std::endl;
    std::cout << mat_B << std::endl;

    real_matrix_t mat_C = kron(mat_A, mat_B);

    std::cout << mat_C << std::endl;
}

BOOST_AUTO_TEST_CASE(multivariate_normal_distribution_rmhmc) {
    test_multivariate_normal<float>(std::string("multivar_nrm_dist_rmhmc.float.chain"));
    // test_kron_prod<double>();
}
