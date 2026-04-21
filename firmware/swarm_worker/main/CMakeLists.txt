idf_component_register(
    SRCS "swarm_worker.c" "../../common/tensor_ops.c"
    INCLUDE_DIRS "." "../../common" "../../models/c_weights"
)
if(DEFINED ENV{SWARM_WORKER_ID})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE WORKER_ID=$ENV{SWARM_WORKER_ID})
endif()
