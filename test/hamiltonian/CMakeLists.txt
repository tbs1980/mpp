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
