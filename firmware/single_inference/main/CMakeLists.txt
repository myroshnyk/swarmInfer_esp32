idf_component_register(
    SRCS "single_inference.c" "../../../common/tensor_ops.c"
    INCLUDE_DIRS "." "../../../common" "../../../models/c_weights"
)
