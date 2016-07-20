#ifndef MPP_CHAIN_MCMC_CHAIN_HPP
#define MPP_CHAIN_MCMC_CHAIN_HPP

#include <fstream>
#include <iomanip>
#include <exception>
#include <sstream>
#include <string>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/assert.hpp>

namespace mpp{ namespace chains{

template<class real_scalar_type>
class mcmc_chain {
public:
    typedef boost::numeric::ublas::vector<real_scalar_type> real_vector_type;
    typedef boost::numeric::ublas::matrix<real_scalar_type> real_matrix_type;

    mcmc_chain(size_t const num_samples, size_t const num_dims)
    : m_num_dims(num_dims)
    , m_num_samples(num_samples)
    , m_chain(num_samples, num_dims)
    , m_weights(num_samples) {
    }

    inline void set_sample(std::size_t const sample_id,
        real_vector_type const & sample,
        real_scalar_type const weight
    ) {
        BOOST_ASSERT(sample_id < m_num_samples );
        boost::numeric::ublas::row(m_chain,sample_id) = sample;
        m_weights(sample_id) = weight;
    }

    inline real_matrix_type const & get_samples() const {
        return m_chain;
    }

    inline real_vector_type const & get_weights() const {
        return m_weights;
    }

    void write_samples_to_csv( std::string const & file_name ) const {
        write_samples_to_csv(file_name,0,m_num_dims);
    }

    void write_samples_to_csv(
        std::string const & file_name,
        std::size_t const from_dim,
        std::size_t const to_dim
    ) const {
        if ( from_dim > to_dim) {
            std::stringstream msg;
            msg << "from_dim  = "
                << from_dim
                << " should be less than to_dim = " << to_dim << ".";
            throw std::length_error(msg.str());
        }
        if ( to_dim > m_num_dims ) {
            std::stringstream msg;
            msg << "to_dim  = "
                << to_dim
                << " should be greater than or equal to "
                << " the number of diemnsions = " << m_num_dims << ".";
            throw std::length_error(msg.str());
        }

        std::string delimiter(",");
        std::ofstream out_file;
        out_file.open(file_name,std::ios::trunc);
        if( out_file.is_open() )
        {
            out_file << std::scientific;
            out_file << std::setprecision(10);
            for(std::size_t i = 0; i < m_num_samples; ++i)
            {
                out_file << m_weights(i) << delimiter;
                for(std::size_t j = from_dim; j < size_t(to_dim-1); ++j)
                {
                    out_file << m_chain(i,j) << delimiter;
                }
                out_file << m_chain(i,size_t(to_dim-1)) <<std::endl;
            }
        }

        out_file.close();
    }

private:
    size_t m_num_dims;
    size_t m_num_samples;
    real_matrix_type m_chain;
    real_vector_type m_weights;
};

}}

#endif // MPP_CHAIN_MCMC_CHAIN_HPP
