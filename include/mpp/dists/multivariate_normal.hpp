#ifndef MPP_DISTS_MULTIVARIATE_NORMAL_HPP
#define MPP_DISTS_MULTIVARIATE_NORMAL_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cmath>
#include <string>
#include <cstddef>

namespace mpp { namespace dists {

template<class _real_scalar_t>
class diag_multivar_normal {
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

    diag_multivar_normal(
        real_vector_t const & mean,
        real_vector_t const & var
    ) throw()
    :m_mean(mean),m_var(var) {
        if( m_mean.size() != m_var.size() ) {
            std::stringstream msg;
            msg << "The number of dimensions "
                << "of mean and var should be identical.";
            throw std::length_error(msg.str());
        }

        if( m_var.size() == std::size_t(0)) {
            std::stringstream msg;
            msg << "The number of dimensions = "
                << m_var.size()
                << " of the multivariate normal should be"
                << " greater than zero.";
            throw std::length_error(msg.str());
        }

        for(std::size_t i=0;i<m_mean.size();++i) {
            if( not std::isfinite(m_mean (i)) ) {
                std::stringstream msg;
                msg << i << "th value of mean is not finite";
                throw std::out_of_range(msg.str());
            }
        }

        for(std::size_t i=0;i<m_var.size();++i) {
            if( not std::isfinite(m_var (i)) ) {
                std::stringstream msg;
                msg << i << "th value of var is not finite";
                throw std::out_of_range(msg.str());
            }
        }

        for(std::size_t i=0;i<m_var.size();++i) {
            if( m_var (i) <= real_scalar_t(0) ) {
                std::stringstream msg;
                msg << i << "th value of var should be a positive real number";
                throw std::out_of_range(msg.str());
            }
        }
    }

    real_scalar_t log_posterior(real_vector_t const & q) const {
        BOOST_ASSERT_MSG(
            q.size() == m_var.size(),
            "q should have the same dimensionality of the log_posterior."
        );
        real_scalar_t val(0);
        for(std::size_t i=0;i<q.size();++i) {
            val -= q(i)*q(i)/m_var(i);
        }
        return real_scalar_t(0.5)*val;
    }

    real_vector_t grad_log_posterior(real_vector_t const & q) const {
        BOOST_ASSERT_MSG(
            q.size() == m_var.size(),
            "q should have the same dimensionality of the log_posterior."
        );
        real_vector_t dq(q.size());
        for(std::size_t i=0;i<q.size();++i) {
            dq(i) = -q(i)/m_var(i);
        }
        return dq;
    }

    real_matrix_t metric_tensor_log_posterior(real_vector_t const & q) const {
        using namespace boost::numeric::ublas;
        real_matrix_t G = identity_matrix<real_scalar_t>(m_var.size());
        for(std::size_t ind_i = 0; ind_i < m_var.size(); ++ind_i){
            G(ind_i,ind_i) = 1./m_var(ind_i);
        }
        return G;
    }

    real_matrix_array_t deriv_metric_tensor_log_posterior(
        real_vector_t const & q
    ) const {
        using namespace boost::numeric::ublas;
        real_matrix_array_t d_G( m_var.size(),
            zero_matrix<real_scalar_t>( m_var.size(),m_var.size() )
        );
        return d_G;
    }

private:
    real_vector_t m_mean;
    real_vector_t m_var;
};

}}

#endif //MPP_DISTS_MULTIVARIATE_NORMAL_HPP
