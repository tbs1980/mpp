#ifndef MPP_DISTS_BANANA_HPP
#define MPP_DISTS_BANANA_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cmath>
#include <string>
#include <sstream>
#include <cstddef>
#include <type_traits>
#include <random>

namespace mpp { namespace dists {

template<class _real_scalar_t>
class banana {
public:
    typedef _real_scalar_t real_scalar_t;
    typedef boost::numeric::ublas::vector<real_scalar_t> real_vector_t;
    typedef boost::numeric::ublas::matrix<real_scalar_t> real_matrix_t;
    typedef boost::numeric::ublas::unbounded_array<real_matrix_t>
        real_matrix_array_t;
    typedef std::normal_distribution<real_scalar_t> normal_distribution_t;
    typedef std::mt19937 rng_type;

    static_assert(
        std::is_floating_point<real_scalar_t>::value,
        "The real scalar is expected to be a floating point type."
    );

    banana(
        real_scalar_t const theta_1_plus_theta_2_sq,
        real_scalar_t const sigma_y,
        real_scalar_t const sigma_theta,
        std::size_t const num_data_points,
        std::size_t random_seed
    )
    : m_theta_1_plus_theta_2_sq(theta_1_plus_theta_2_sq)
    , m_sigma_y(sigma_y)
    , m_sigma_theta(sigma_theta)
    , m_num_data_points(num_data_points)
    {
        if( m_sigma_y < 0 ) {
            std::stringstream msg;
            msg << "The std dvn of y "
                << m_sigma_y
                << " should be greater than zero.";
            throw std::domain_error(msg.str());
        }

        if( m_sigma_theta < 0 ) {
            std::stringstream msg;
            msg << "The std dvn of theta "
                << m_sigma_theta
                << " should be greater than zero.";
            throw std::domain_error(msg.str());
        }

        if( num_data_points == 0 ) {
            std::stringstream msg;
            msg << "The number of data points "
                << num_data_points
                << " cannot be zero.";
            throw std::length_error(msg.str());
        }

        m_data_y = real_vector_t(m_num_data_points);
        rng_type rng(random_seed);
        normal_distribution_t norm_dist(m_theta_1_plus_theta_2_sq,m_sigma_y);
        for(std::size_t dim_i = 0; dim_i < m_num_data_points; ++ dim_i) {
            m_data_y(dim_i) = norm_dist(rng);
        }
    }

    real_scalar_t log_posterior(real_vector_t const & theta) const {
        BOOST_ASSERT_MSG(
            theta.size() == 2,
            "theta should be a two dimensional vector."
        );
        real_scalar_t const theta_1 = theta(0);
        real_scalar_t const theta_2 = theta(1);
        real_scalar_t const mu = theta_1 + theta_2*theta_2;

        real_scalar_t log_pr =
            -0.5*(theta_1*theta_1 + theta_2*theta_2)
            /m_sigma_theta/m_sigma_theta;


        real_scalar_t log_lik(0);
        for(std::size_t dim_i = 0; dim_i < m_num_data_points; ++ dim_i){
            real_scalar_t const diff = m_data_y(dim_i) - mu;
            log_lik -= diff*diff/m_sigma_y/m_sigma_y;
        }
        log_lik *= 0.5;

        return log_pr + log_lik;
    }

    real_vector_t grad_log_posterior(real_vector_t const & theta) const {
        BOOST_ASSERT_MSG(
            theta.size() == 2,
            "theta should be a two dimensional vector."
        );
        real_vector_t d_log_pr = -theta/m_sigma_theta/m_sigma_theta;
        real_vector_t d_log_lik = scalar_vector(theta.size(),1.);
        d_log_lik(1) = 2*theta(1);
        real_scalar_t const theta_1 = theta(0);
        real_scalar_t const theta_2 = theta(1);
        real_scalar_t const mu = theta_1 + theta_2*theta_2;
        d_log_lik *= ( sum(m_data_y) - m_num_data_points*mu )
            /m_sigma_y/m_sigma_y;
        real_vector_t d_theta = d_log_pr + d_log_lik;
        return d_theta;
    }

    real_matrix_t metric_tensor_log_posterior(real_vector_t const & theta)const{
        BOOST_ASSERT_MSG(
            theta.size() == 2,
            "theta should be a two dimensional vector."
        );
        using namespace boost::numeric::ublas;
        real_matrix_t G(theta.size(),theta.size());
        real_vector_t theta_tmp = scalar_vector(theta.size(),1.);
        theta_tmp(1) = 2*theta(1);
        G = outer_prod(theta_tmp,theta_tmp);
        BOOST_ASSERT(G.size1() == 2);
        BOOST_ASSERT(G.size2() == 2);
        return G;
    }

    real_matrix_array_t deriv_metric_tensor_log_posterior(
        real_vector_t const & q
    ) const {
        BOOST_ASSERT_MSG(
            theta.size() == 2,
            "theta should be a two dimensional vector."
        );
        using namespace boost::numeric::ublas;
        real_matrix_array_t d_G( m_var.size(),
            zero_matrix<real_scalar_t>( m_var.size(),m_var.size() )
        );
        return d_G;
    }

private:
    real_scalar_t m_theta_1_plus_theta_2_sq;
    real_scalar_t m_sigma_y;
    real_scalar_t m_sigma_theta;
    std::size_t m_num_data_points;
    real_vector_t m_data_y;
};

}}

#endif
