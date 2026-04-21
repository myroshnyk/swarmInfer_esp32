idf_component_register(
    SRCS "swarm_coordinator.c" "../../../common/tensor_ops.c"
    INCLUDE_DIRS "." "../../../common" "../../../models/c_weights"
)
if(DEFINED ENV{SWARM_N_WORKERS})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE NUM_WORKERS=$ENV{SWARM_N_WORKERS})
endif()
