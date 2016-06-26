#include <mpp.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <functional>
#include <random>
#include <cstddef>

using namespace mpp::hamiltonian;
using namespace mpp::chains;
using namespace boost::numeric::ublas;

double log_posterior(vector<double> const & q){
    double log_post_val(0);
    for(std::size_t i=0;i<q.size();++i) {
        log_post_val -= q(i)*q(i);
    }
    return log_post_val;
}

vector<double> grad_log_posterior(vector<double> const & q) {
    vector<double> dq(q.size());
    for(std::size_t i=0;i<q.size();++i) {
        dq(i) = -q(i);
    }
    return dq;
}

int main() {
    std::size_t const num_dims(10);
    std::size_t const max_num_steps(10);
    double const max_eps(1);
    vector<double> inv_mass_mat(num_dims);
    for(std::size_t i=0;i<num_dims;++i) {
        inv_mass_mat(i) = double(1);
    }

    std::function<double(vector<double> const &)> lp = log_posterior;
    std::function<
        vector<double> (vector<double> const &) > glp = grad_log_posterior;

    hmc_sampler<double>  hmc_spr(
        lp,
        glp,
        num_dims,
        max_num_steps,
        max_eps,
        inv_mass_mat
    );

    std::size_t const num_samples(1000);
    std::mt19937 rng;
    std::normal_distribution<double> nrm_dist;
    vector<double> q_0(num_dims);
    for(size_t i=0;i<num_dims;++i) {
        q_0(i) = nrm_dist(rng);
    }
    mcmc_chain<double> chn = hmc_spr.run_sampler(num_samples,q_0,rng);
    std::string chn_file_name("./eg_classic_hmc.chain");
    chn.write_samples_to_csv(chn_file_name);

    return 0;
}
