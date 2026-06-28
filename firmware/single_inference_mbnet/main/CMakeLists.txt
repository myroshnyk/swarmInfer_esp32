idf_component_register(
    SRCS "single_inference_mbnet.c" "../../common/tensor_ops.c" "../../common/mbnet_ops.c"
    INCLUDE_DIRS "." "../../common" "../../models/mobilenet"
)
