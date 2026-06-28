/* R2-6 (part B): on-device (Xtensa LX7) microbenchmark of the bitmap
 * sparsification encode/decode cost vs the unicast transmit time it saves,
 * per FatCNN gather layer. ISOLATED — reuses common/tensor_ops.c
 * (sparse_encode / sparse_decode); touches no released firmware.
 *
 * For each layer we build a per-worker (N=4) gather payload of the measured
 * size and measured post-ReLU sparsity, time encode + decode (averaged), and
 * compare against the dense vs sparse unicast transfer time at the measured
 * 81.3 KB/s ESP-NOW unicast throughput. Net benefit = transmit saved
 * - (encode + decode). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "tensor_ops.h"
#include "swarm_protocol.h"   /* SWARM_CHUNK_SIZE */

static const char *TAG = "SPARSEBENCH";

#define ITERS 200
#define UNICAST_BPS 81300.0    /* measured ESP-NOW unicast throughput, bytes/s */

/* Per-worker (N=4) gather payloads: {name, size_elems, sparsity_percent} */
typedef struct { const char *name; int size; int sparsity_pct; } LayerBench;
static const LayerBench LAYERS[] = {
    {"Conv1", 16 * 16 * 16, 54},   /* 4096, 54% zeros */
    {"Conv2", 8 * 8 * 32, 75},     /* 2048, 75% zeros */
    {"Conv3 (GAP)", 64, 42},       /* 64,  42% zeros */
};
#define NLAYERS (sizeof(LAYERS) / sizeof(LAYERS[0]))

static const int8_t ZP = -128;     /* ReLU zero-point */

/* Build a tensor with EXACTLY sparsity_pct% of elements == ZP (zeros), the rest
 * a non-zero value, spread evenly via a Bresenham-style accumulator (no RNG). */
static void fill_pattern(int8_t *buf, int size, int sparsity_pct) {
    int nnz = size * (100 - sparsity_pct) / 100;   /* number of non-zeros */
    int acc = 0;
    for (int i = 0; i < size; i++) {
        acc += nnz;
        if (acc >= size) { acc -= size; buf[i] = (int8_t)42; }
        else buf[i] = ZP;
    }
}

/* transmit time (us) for `bytes` over unicast, including chunk framing rounding */
static double tx_us(int bytes) {
    return (double)bytes / UNICAST_BPS * 1e6;
}

void app_main(void) {
    ESP_LOGI(TAG, "=== Sparsification encode/decode microbenchmark (LX7) ===");
    ESP_LOGI(TAG, "ITERS=%d, unicast=%.0f B/s", ITERS, UNICAST_BPS);
    ESP_LOGI(TAG, "CSV,layer,size,sparsity,enc_size,enc_us,dec_us,tx_dense_us,tx_sparse_us,saved_us,net_us");

    for (int l = 0; l < NLAYERS; l++) {
        int size = LAYERS[l].size;
        int8_t *src = malloc(size);
        int8_t *enc = malloc(size + size / 8 + 16);
        int8_t *dec = malloc(size);
        if (!src || !enc || !dec) { ESP_LOGE(TAG, "malloc fail"); return; }
        fill_pattern(src, size, LAYERS[l].sparsity_pct);

        /* encode timing */
        uint32_t enc_size = 0;
        int64_t t0 = esp_timer_get_time();
        for (int it = 0; it < ITERS; it++)
            enc_size = sparse_encode(src, size, enc, ZP);
        double enc_us = (double)(esp_timer_get_time() - t0) / ITERS;

        /* decode timing */
        t0 = esp_timer_get_time();
        for (int it = 0; it < ITERS; it++)
            sparse_decode(enc, dec, size, ZP);
        double dec_us = (double)(esp_timer_get_time() - t0) / ITERS;

        /* correctness */
        int ok = (memcmp(src, dec, size) == 0);

        double tx_dense = tx_us(size);
        double tx_sparse = tx_us((int)enc_size);
        double saved = tx_dense - tx_sparse;
        double net = saved - (enc_us + dec_us);

        ESP_LOGI(TAG, "CSV,%s,%d,%d,%lu,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f  [%s]",
                 LAYERS[l].name, size, LAYERS[l].sparsity_pct, (unsigned long)enc_size,
                 enc_us, dec_us, tx_dense, tx_sparse, saved, net,
                 ok ? "decode OK" : "DECODE MISMATCH");
        free(src); free(enc); free(dec);
    }
    ESP_LOGI(TAG, "=== done ===");
    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}
