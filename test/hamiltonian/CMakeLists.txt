add_compute_test(
    "hamiltonian.kinetic_energy_multivar_normal"
    kinetic_energy_multivar_normal.cpp
)
add_compute_test(
    "hamiltonian.classic_hamiltonian"
    classic_hamiltonian.cpp
)
add_compute_test(
    "hamiltonian.nuts"
    nuts.cpp
)

add_compute_test(
    "hamiltonian.reimann_manifold_hmc"
    reimann_manifold_hmc.cpp
)

add_compute_test(
    "hamiltonian.log_post_taylor_2008_2d"
    log_post_taylor_2008_2d.cpp
)

add_compute_test(
    "hamiltonian.log_post_balan_2016_2d"
    log_post_balan_2016_2d.cpp
)

add_compute_test(
    "hamiltonian.log_post_wandelt_2004_2d"
    log_post_wandelt_2004_2d.cpp
)

add_compute_test(
        "hamiltonian.norm_dist_as_invar_dens_rmhmc"
        norm_dist_as_invar_dens_rmhmc.cpp
)

add_compute_test(
        "hamiltonian.multi_var_norm_as_invar_dens_rmhmc"
        multi_var_norm_as_invar_dens_rmhmc.cpp
)
