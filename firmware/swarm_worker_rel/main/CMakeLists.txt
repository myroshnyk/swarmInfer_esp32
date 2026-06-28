set(power_srcs "")
if(DEFINED ENV{SWARM_POWER_MEASURE})
    set(power_srcs "../../common/ina219.c" "../../common/power_meter.c")
endif()

idf_component_register(
    SRCS "swarm_worker.c" "../../common/tensor_ops.c" ${power_srcs}
    INCLUDE_DIRS "." "../../common" "../../models/c_weights"
)
if(DEFINED ENV{SWARM_POWER_MEASURE})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE POWER_MEASURE=1)
endif()
if(DEFINED ENV{SWARM_WORKER_ID})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE WORKER_ID=$ENV{SWARM_WORKER_ID})
endif()
if(DEFINED ENV{SWARM_FORCE_DENSE})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FORCE_DENSE=$ENV{SWARM_FORCE_DENSE})
endif()
