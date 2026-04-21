idf_component_register(
    SRCS "gather_bench_coordinator.c"
    INCLUDE_DIRS "." "../../../common"
)
if(DEFINED ENV{SWARM_N_WORKERS})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE NUM_WORKERS=$ENV{SWARM_N_WORKERS})
endif()
