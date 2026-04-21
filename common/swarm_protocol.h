#ifndef SWARM_PROTOCOL_H
#define SWARM_PROTOCOL_H
#include <stdint.h>
#define SWARM_MAX_PAYLOAD   250
#define SWARM_HEADER_SIZE   8
#define SWARM_CHUNK_SIZE    232
#define SWARM_PACKET_SIZE   240
#define CMD_LAYER_START     0x01
#define CMD_INPUT_CHUNK     0x02
#define CMD_COMPUTE         0x03
#define CMD_RESULT_CHUNK    0x04
#define CMD_RESULT_DONE     0x05
#define CMD_INFERENCE_DONE  0x06
#define CMD_WORKER_READY    0x07
typedef struct __attribute__((packed)) {
    uint8_t  cmd;
    uint8_t  layer;
    uint16_t chunk_id;
    uint16_t total_chunks;
    uint16_t data_len;
    uint8_t  data[SWARM_CHUNK_SIZE];
} SwarmPacket;
static inline uint16_t swarm_num_chunks(uint32_t total_bytes) {
    return (uint16_t)((total_bytes + SWARM_CHUNK_SIZE - 1) / SWARM_CHUNK_SIZE);
}
#define L1_INPUT_H 32
#define L1_INPUT_W 32
#define L1_INPUT_C 3
#define L1_OUTPUT_C 64
#define L1_KERNEL_H 5
#define L1_KERNEL_W 5
#define L1_STRIDE 1
#define L1_PADDING 2
#define L1_WORKER_OC 16
#define L1_CONV_H 32
#define L1_CONV_W 32
#define L1_POOL_H 16
#define L1_POOL_W 16
#define L1_INPUT_SIZE  (L1_INPUT_H * L1_INPUT_W * L1_INPUT_C)
#define L1_CONV_SIZE   (L1_CONV_H * L1_CONV_W * L1_WORKER_OC)
#define L1_POOL_SIZE   (L1_POOL_H * L1_POOL_W * L1_WORKER_OC)
#define L1_FULL_OUTPUT (L1_POOL_H * L1_POOL_W * L1_OUTPUT_C)

/* ── Layer 2: Conv2 3×3, 64→128, pad=1, stride=1, MaxPool 2×2 ── */
#define L2_INPUT_H    16
#define L2_INPUT_W    16
#define L2_INPUT_C    64
#define L2_OUTPUT_C   128
#define L2_KERNEL_H   3
#define L2_KERNEL_W   3
#define L2_STRIDE     1
#define L2_PADDING    1
#define L2_WORKER_OC  32
#define L2_CONV_H     16
#define L2_CONV_W     16
#define L2_POOL_H     8
#define L2_POOL_W     8
#define L2_INPUT_SIZE  (L2_INPUT_H * L2_INPUT_W * L2_INPUT_C)
#define L2_CONV_SIZE   (L2_CONV_H * L2_CONV_W * L2_WORKER_OC)
#define L2_POOL_SIZE   (L2_POOL_H * L2_POOL_W * L2_WORKER_OC)
#define L2_FULL_OUTPUT (L2_POOL_H * L2_POOL_W * L2_OUTPUT_C)

/* ── Layer 3: Conv3 3×3, 128→256, pad=1, stride=1, ReLU + GAP ── */
#define L3_INPUT_H    8
#define L3_INPUT_W    8
#define L3_INPUT_C    128
#define L3_OUTPUT_C   256
#define L3_KERNEL_H   3
#define L3_KERNEL_W   3
#define L3_STRIDE     1
#define L3_PADDING    1
#define L3_WORKER_OC  64
#define L3_CONV_H     8
#define L3_CONV_W     8
#define L3_INPUT_SIZE  (L3_INPUT_H * L3_INPUT_W * L3_INPUT_C)
#define L3_CONV_SIZE   (L3_CONV_H * L3_CONV_W * L3_WORKER_OC)
#define L3_GAP_SIZE    L3_WORKER_OC
#define L3_FULL_GAP    L3_OUTPUT_C

/* ── Dense layers (coordinator only) ── */
#define DENSE1_IN     256
#define DENSE1_OUT    128
#define DENSE2_IN     128
#define DENSE2_OUT    10

#endif
