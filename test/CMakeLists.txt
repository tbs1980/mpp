include_directories(../include)

function(add_compute_test TEST_NAME TEST_SOURCE)
    get_filename_component(TEST_TARGET ${TEST_SOURCE} NAME_WE)
    add_executable(test_${TEST_TARGET} ${TEST_SOURCE})
    target_link_libraries(test_${TEST_TARGET} ${Boost_LIBRARIES} )
    add_test(${TEST_NAME} test_${TEST_TARGET})
endfunction()

add_subdirectory(chains)
add_subdirectory(hamiltonian)
add_subdirectory(ensemble)
