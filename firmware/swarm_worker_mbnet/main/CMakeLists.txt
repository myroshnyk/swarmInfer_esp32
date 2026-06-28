idf_component_register(
    SRCS "swarm_worker_mbnet.c" "../../common/tensor_ops.c" "../../common/mbnet_ops.c"
    INCLUDE_DIRS "." "../../common" "../../models/mobilenet"
)
if(DEFINED ENV{SWARM_WORKER_ID})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE WORKER_ID=$ENV{SWARM_WORKER_ID})
endif()
