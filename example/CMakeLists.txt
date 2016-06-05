include_directories(../include)

function(add_example EXAMPLE_NAME EXAMPLE_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(example_${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})
    target_link_libraries(example_${EXAMPLE_TARGET} ${Boost_LIBRARIES} )
endfunction()

add_subdirectory(quickstart)
