#ifndef MPP_DISTS_MULTIVARIATE_NORMAL_HPP
#define MPP_DISTS_MULTIVARIATE_NORMAL_HPP

namespace mpp { namespace dists {

template<class _real_scalar_type>
class diag_multivar_normal {
public:
    typedef _real_scalar_type real_scalar_type;
    typedef boost::numeric::ublas::vector<real_scalar_type> real_vector_type;

    static_assert(
        std::is_floating_point<real_scalar_type>::value,
        "The real scalar is expected to be a floating point type."
    );

    diag_multivar_normal(
        real_vector_type const & mean,
        real_vector_type const & var
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
            if( m_var (i) <= real_scalar_type(0) ) {
                std::stringstream msg;
                msg << i << "th value of var should be a positive real number";
                throw std::out_of_range(msg.str());
            }
        }
    }

    real_scalar_type log_posterior(real_vector_type const & q) const {
        BOOST_ASSERT_MSG(
            q.size() == m_var.size(),
            "q should have the same dimensionality of the log_posterior."
        );
        real_scalar_type val(0);
        for(std::size_t i=0;i<q.size();++i) {
            val -= q(i)*q(i)/m_var(i);
        }
        return real_scalar_type(0.5)*val;
    }

    real_vector_type grad_log_posterior(real_vector_type const & q) const {
        BOOST_ASSERT_MSG(
            q.size() == m_var.size(),
            "q should have the same dimensionality of the log_posterior."
        );
        real_vector_type dq(q.size());
        for(std::size_t i=0;i<q.size();++i) {
            dq(i) = -q(i)/m_var(i);
        }
        return dq;
    }

private:
    real_vector_type m_mean;
    real_vector_type m_var;
};

}}

#endif //MPP_DISTS_MULTIVARIATE_NORMAL_HPP
