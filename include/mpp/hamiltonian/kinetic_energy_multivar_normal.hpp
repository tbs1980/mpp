#ifndef MPP_HAMILTONIAN_KINETIC_ENERGY_MULTIVAR_NORMAL_HPP
#define MPP_HAMILTONIAN_KINETIC_ENERGY_MULTIVAR_NORMAL_HPP

#include <random>
#include <exception>
#include <sstream>
#include <string>
#include <cmath>
#include <cstddef>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/assert.hpp>

namespace mpp { namespace hamiltonian {

template<class real_scalar_type>
class multivariate_normal {
public:
    typedef boost::numeric::ublas::vector<real_scalar_type> real_vector_type;
    typedef std::normal_distribution<real_scalar_type> normal_distribution_type;

    explicit multivariate_normal(real_vector_type const & sigma_inv)
    :m_sigma_inv(sigma_inv) {
        if( sigma_inv.size() == std::size_t(0)) {
            std::stringstream msg;
            msg << "The number of dimensions = "
                << m_sigma_inv.size()
                << " of the multivariate normal kinetic energy should be"
                << " greater than zero.";
            throw std::length_error(msg.str());
        }

        for(std::size_t i=0;i<sigma_inv.size();++i) {
            if( not std::isfinite(sigma_inv (i)) ) {
                std::stringstream msg;
                msg << i << "th value of sigma_inv is not finite";
                throw std::out_of_range(msg.str());
            }
        }
    }

    real_scalar_type log_posterior(real_vector_type const & p) const {
        BOOST_ASSERT_MSG( p.size() == m_sigma_inv.size(),
            "p should have the same dimensionality of the log_posterior.");
        real_scalar_type val(0);
        for(std::size_t i=0;i<p.size();++i) {
            val -= p(i)*p(i)*m_sigma_inv(i);
        }
        return real_scalar_type(0.5)*val;
    }

    real_vector_type grad_log_posterior(real_vector_type const & p) const {
        BOOST_ASSERT_MSG( p.size() == m_sigma_inv.size(),
            "p should have the same dimensionality of the log_posterior.");
        real_vector_type dp(p.size());
        for(std::size_t i=0;i<p.size();++i) {
            dp(i) = -m_sigma_inv(i)*p(i);
        }
        return dp;
    }

    template<class rng_type>
    real_vector_type generate_sample(rng_type & rng) {
        real_vector_type sample(m_sigma_inv.size());
        for(std::size_t i=0;i<m_sigma_inv.size();++i) {
            real_scalar_type scale
                = m_sigma_inv(i) > real_scalar_type(0) ?
                    std::sqrt( real_scalar_type(1)/m_sigma_inv(i) ) : 0 ;
            sample(i) = scale*m_norm_dist(rng);
        }
        return sample;
    }
private:
    real_vector_type m_sigma_inv;
    normal_distribution_type m_norm_dist;
};

}}

#endif //MPP_HAMILTONIAN_KINETIC_ENERGY_MULTIVAR_NORMAL_HPP
