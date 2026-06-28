set(power_srcs "")
if(DEFINED ENV{SWARM_POWER_MEASURE})
    set(power_srcs "../../common/ina219.c" "../../common/power_meter.c")
endif()

idf_component_register(
    SRCS "swarm_coordinator_nscale.c" "../../common/tensor_ops.c" ${power_srcs}
    INCLUDE_DIRS "." "../../common" "../../models/c_weights_nscale"
)
if(DEFINED ENV{SWARM_POWER_MEASURE})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE POWER_MEASURE=1)
endif()
if(DEFINED ENV{SWARM_N_WORKERS})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE NUM_WORKERS=$ENV{SWARM_N_WORKERS})
endif()
if(DEFINED ENV{SWARM_SMOKE_IMAGES})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE SMOKE_IMAGES=$ENV{SWARM_SMOKE_IMAGES})
endif()
