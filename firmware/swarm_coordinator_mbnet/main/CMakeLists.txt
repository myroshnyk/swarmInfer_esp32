idf_component_register(
    SRCS "swarm_coordinator_mbnet.c" "../../common/tensor_ops.c" "../../common/mbnet_ops.c"
    INCLUDE_DIRS "." "../../common" "../../models/mobilenet"
)
if(DEFINED ENV{SWARM_N_WORKERS})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE NUM_WORKERS=$ENV{SWARM_N_WORKERS})
endif()
