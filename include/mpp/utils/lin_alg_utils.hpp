#ifndef MPP_LIN_ALG_UTILS_HPP
#define MPP_LIN_ALG_UTILS_HPP

#include <cmath>
#include <cstddef>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace mpp { namespace utils {

template<typename real_scalar_t>
real_scalar_t compute_trace(
    boost::numeric::ublas::matrix<real_scalar_t>  m
){
    // http://boost.2283326.n4.nabble.com/Matrix-Trace-td2710079.html
    using namespace boost::numeric::ublas;
    typedef matrix<real_scalar_t> real_matrix_t;
    matrix_vector_range<real_matrix_t> diag (
        m, 
        range(0,m.size1()), 
        range(0,m.size2())
    );
    return sum(diag);
}

template<typename real_scalar_type>
real_scalar_type compute_determinant(
    boost::numeric::ublas::matrix<real_scalar_type>  m
) {
    // http://www.richelbilderbeek.nl/CppUblasMatrixExample7.htm
    using namespace boost::numeric::ublas;
    BOOST_ASSERT( m.size1() == m.size2() );
    permutation_matrix<std::size_t> pivots( m.size1() );
    int const is_singular = lu_factorize(m, pivots);
    if (is_singular){
        return real_scalar_type(0);
    }
    real_scalar_type d(1);
    std::size_t const sz = pivots.size();
    for (std::size_t i=0; i != sz; ++i){
        if ( pivots(i) != i ){
            d *= -1.;
        }
        d *= m(i,i);
    }
    return d;
}

template <typename real_scalar_t>
std::size_t cholesky_decompose(
    boost::numeric::ublas::matrix<real_scalar_t> const & mat_A,
    boost::numeric::ublas::matrix<real_scalar_t> &  mat_L
) {
    using namespace boost::numeric::ublas;
    typedef matrix<real_scalar_t> real_matrix_type;
    // http://www.guwi17.de/ublas/examples/
    BOOST_ASSERT( mat_A.size1() == mat_A.size2() );
    BOOST_ASSERT( mat_A.size1() ==  mat_L.size1() );
    BOOST_ASSERT( mat_A.size2() ==  mat_L.size2() );

    std::size_t const num_rows = mat_A.size1();
    for (std::size_t k=0 ; k < num_rows; k++) {
        real_scalar_t qL_kk = mat_A(k,k) - inner_prod(
            project( row( mat_L, k), range(0, k) ),
            project( row( mat_L, k), range(0, k) )
        );

        if (qL_kk <= 0) {
            return 1 + k;
        }
        else {
            real_scalar_t L_kk = std::sqrt( qL_kk );
             mat_L(k,k) =  L_kk;
            matrix_column<real_matrix_type> cLk(mat_L, k);
            project( cLk, range(k+1, num_rows) ) = (
                project( column( mat_A, k), range(k+1, num_rows) ) - prod(
                    project( mat_L, range(k+1, num_rows), range(0, k) ) ,
                    project( row(mat_L, k), range(0, k) )
                )
            ) / L_kk;
        }
    }
    return 0;
}

template<typename real_scalar_t>
bool compute_inverse(
    boost::numeric::ublas::matrix<real_scalar_t> const & input,
    boost::numeric::ublas::matrix<real_scalar_t> & inverse
) {
    // https://gist.github.com/lilac/2464434
    BOOST_ASSERT( input.size1() == input.size2() );
    BOOST_ASSERT( input.size1() == inverse.size1() );
    BOOST_ASSERT( inverse.size1() == inverse.size2() );
    using namespace boost::numeric::ublas;
    matrix<real_scalar_t> A(input);
    permutation_matrix<std::size_t> pm(A.size1());
    int const res = lu_factorize(A,pm);
    if( res != 0 ) {
        return false;
    }
    inverse.assign( identity_matrix<real_scalar_t>( A.size1() ) );
    lu_substitute(A, pm, inverse);
    return true;
}


}}

#endif //MPP_LIN_ALG_UTILS_HPP
