#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "esp_netif.h"
#include "esp_rom_sys.h"
#include "tensor_ops.h"
#include "swarm_protocol.h"
#if __has_include("swarm_dims.h")
#include "swarm_dims.h"
#endif
#include "coordinator_weights.h"
#include "test_images_batch.h"

static const char *TAG = "COORD";

#ifndef NUM_WORKERS
#define NUM_WORKERS 4
#endif

static const uint8_t ALL_WORKER_MACS[4][6] = {
    {0xB8,0xF8,0x62,0xE2,0xD1,0x98},  /* Worker 0 */
    {0xB8,0xF8,0x62,0xE2,0xCD,0xE4},  /* Worker 1 */
    {0xB8,0xF8,0x62,0xE2,0xDA,0x28},  /* Worker 2 */
    {0xB8,0xF8,0x62,0xE2,0xD0,0xDC},  /* Worker 3 */
};
/* Only use first NUM_WORKERS MACs */
#define WORKER_MACS_ARR ALL_WORKER_MACS
static const uint8_t BROADCAST_ADDR[] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};

static const char *cifar_classes[] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

/* ── Gather state ── */
static volatile int  worker_chunks_received[NUM_WORKERS];
static volatile bool worker_done[NUM_WORKERS];
static volatile int  workers_complete;
static SemaphoreHandle_t all_workers_done_sem;

/* ── Ready handshake state ── */
static volatile bool worker_ready[NUM_WORKERS];
static volatile int  workers_ready_count;
static SemaphoreHandle_t all_workers_ready_sem;

/* Per-worker receive buffers (sized for the largest per-worker result = L1_POOL_SIZE) */
static int8_t *worker_recv_buf[NUM_WORKERS];
#define WORKER_RECV_BUF_SIZE  L1_POOL_SIZE  /* 4096 — largest per-worker result */

/* Current expected per-worker result size (set before each gather) */
static volatile uint32_t current_result_size = 0;

/* Sparse metadata (set from CMD_RESULT_DONE) */
static volatile bool    worker_sparse[NUM_WORKERS];
static volatile uint16_t worker_sparse_orig[NUM_WORKERS];
static volatile int8_t  worker_sparse_zp[NUM_WORKERS];
static int8_t *decode_tmp = NULL;  /* temp buffer for sparse decode */

/* ── Assembled layer outputs ── */
static int8_t *layer1_output = NULL;  /* 16×16×64  = 16384 */
static int8_t *layer2_output = NULL;  /* 8×8×128   = 8192  */
static int8_t *gap_output    = NULL;  /* 256                */
static int8_t *dense1_out    = NULL;  /* 128                */
static int8_t *dense2_out    = NULL;  /* 10                 */

/* ── Timing ── */
static int64_t t_broadcast_start, t_broadcast_end;
static int64_t t_gather_start, t_gather_end;

static int mac_to_worker_id(const uint8_t *mac) {
    for (int i = 0; i < NUM_WORKERS; i++)
        if (memcmp(mac, WORKER_MACS_ARR[i], 6) == 0) return i;
    return -1;
}

static void on_sent(const esp_now_send_info_t *info, esp_now_send_status_t status) {
    (void)info; (void)status;
}

static void on_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (len < SWARM_HEADER_SIZE) return;
    const SwarmPacket *pkt = (const SwarmPacket *)data;
    int wid = mac_to_worker_id(info->src_addr);
    if (wid < 0) return;

    switch (pkt->cmd) {
    case CMD_RESULT_CHUNK:
        if (pkt->chunk_id < pkt->total_chunks && worker_recv_buf[wid]) {
            uint32_t offset = (uint32_t)pkt->chunk_id * SWARM_CHUNK_SIZE;
            uint16_t copy_len = pkt->data_len;
            if (offset + copy_len <= current_result_size)
                memcpy(&worker_recv_buf[wid][offset], pkt->data, copy_len);
            worker_chunks_received[wid]++;
        }
        break;
    case CMD_RESULT_DONE:
        worker_done[wid] = true;
        if (pkt->data_len >= 4 && pkt->data[0] == 1) {
            worker_sparse[wid] = true;
            worker_sparse_orig[wid] = pkt->data[1] | ((uint16_t)pkt->data[2] << 8);
            worker_sparse_zp[wid] = (int8_t)pkt->data[3];
        } else {
            worker_sparse[wid] = false;
        }
        workers_complete++;
        if (workers_complete >= NUM_WORKERS) {
            t_gather_end = esp_timer_get_time();
            xSemaphoreGiveFromISR(all_workers_done_sem, NULL);
        }
        break;
    case CMD_WORKER_READY:
        if (!worker_ready[wid]) {
            worker_ready[wid] = true;
            workers_ready_count++;
            if (workers_ready_count >= NUM_WORKERS)
                xSemaphoreGiveFromISR(all_workers_ready_sem, NULL);
        }
        break;
    default: break;
    }
}

static void wifi_init(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase(); nvs_flash_init();
    }
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE));
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_send_cb(on_sent));
    ESP_ERROR_CHECK(esp_now_register_recv_cb(on_recv));
    esp_now_peer_info_t bcast = {0};
    memcpy(bcast.peer_addr, BROADCAST_ADDR, 6);
    bcast.channel = 1; bcast.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&bcast));
    for (int i = 0; i < NUM_WORKERS; i++) {
        esp_now_peer_info_t p = {0};
        memcpy(p.peer_addr, WORKER_MACS_ARR[i], 6);
        p.channel = 1; p.encrypt = false;
        esp_now_add_peer(&p);
    }
}

/* ── Reset ready state and wait for all workers ── */
static void reset_ready(void) {
    for (int i = 0; i < NUM_WORKERS; i++)
        worker_ready[i] = false;
    workers_ready_count = 0;
    xSemaphoreTake(all_workers_ready_sem, 0);
}

static bool wait_for_ready(const char *context) {
    if (xSemaphoreTake(all_workers_ready_sem, pdMS_TO_TICKS(30000)) != pdTRUE) {
        ESP_LOGE(TAG, "%s: READY timeout! Got %d/%d", context,
                 workers_ready_count, NUM_WORKERS);
        for (int i = 0; i < NUM_WORKERS; i++)
            ESP_LOGE(TAG, "  W%d: ready=%d", i, worker_ready[i]);
        return false;
    }
    return true;
}

/* ── Broadcast data to all workers ── */
static void broadcast_data(const int8_t *data, uint32_t data_len, uint8_t layer) {
    uint16_t total = swarm_num_chunks(data_len);

    SwarmPacket start_pkt = {0};
    start_pkt.cmd = CMD_LAYER_START;
    start_pkt.layer = layer;
    start_pkt.total_chunks = total;
    start_pkt.data_len = 0;
    esp_now_send(BROADCAST_ADDR, (uint8_t *)&start_pkt, SWARM_HEADER_SIZE);
    vTaskDelay(pdMS_TO_TICKS(5));

    t_broadcast_start = esp_timer_get_time();

    SwarmPacket pkt;
    pkt.cmd = CMD_INPUT_CHUNK;
    pkt.layer = layer;
    pkt.total_chunks = total;
    for (uint16_t i = 0; i < total; i++) {
        pkt.chunk_id = i;
        uint32_t offset = (uint32_t)i * SWARM_CHUNK_SIZE;
        uint16_t remain = data_len - offset;
        pkt.data_len = (remain > SWARM_CHUNK_SIZE) ? SWARM_CHUNK_SIZE : remain;
        memcpy(pkt.data, &data[offset], pkt.data_len);
        esp_err_t err;
        do {
            err = esp_now_send(BROADCAST_ADDR, (uint8_t *)&pkt, SWARM_HEADER_SIZE + pkt.data_len);
            if (err == ESP_ERR_ESPNOW_NO_MEM) {
                esp_rom_delay_us(500);  /* busy-wait 0.5ms instead of 10ms tick */
            }
        } while (err == ESP_ERR_ESPNOW_NO_MEM);
    }
    /* Workers auto-trigger compute when all chunks received.
     * Send COMPUTE as fallback (e.g. if a chunk was lost). */
    esp_rom_delay_us(2000);
    SwarmPacket comp = {0};
    comp.cmd = CMD_COMPUTE;
    comp.layer = layer;
    comp.data_len = 0;
    esp_now_send(BROADCAST_ADDR, (uint8_t *)&comp, SWARM_HEADER_SIZE);
    t_broadcast_end = esp_timer_get_time();
}

/* ── Reset gather state before each layer ── */
static void reset_gather(uint32_t per_worker_result_size) {
    for (int i = 0; i < NUM_WORKERS; i++) {
        worker_chunks_received[i] = 0;
        worker_done[i] = false;
        worker_sparse[i] = false;
    }
    workers_complete = 0;
    current_result_size = per_worker_result_size;
    xSemaphoreTake(all_workers_done_sem, 0);
}

/* ── Wait for all workers, return gather time in µs (or -1 on timeout) ── */
static int64_t wait_for_workers(void) {
    t_gather_start = esp_timer_get_time();
    if (xSemaphoreTake(all_workers_done_sem, pdMS_TO_TICKS(30000)) != pdTRUE) {
        ESP_LOGE(TAG, "TIMEOUT! Done: %d/%d", workers_complete, NUM_WORKERS);
        for (int i = 0; i < NUM_WORKERS; i++)
            ESP_LOGE(TAG, "  W%d: chunks=%d done=%d",
                     i, worker_chunks_received[i], worker_done[i]);
        return -1;
    }
    return t_gather_end - t_gather_start;
}

/* ── Assemble: interleave partial channel outputs from workers ── */
static void assemble_spatial(const int8_t *bufs[], int H, int W, int ch_per_worker,
                             int8_t *output) {
    int Cf = ch_per_worker * NUM_WORKERS;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int w = 0; w < NUM_WORKERS; w++)
                memcpy(&output[(y * W + x) * Cf + w * ch_per_worker],
                       &bufs[w][(y * W + x) * ch_per_worker],
                       ch_per_worker);
}

/* ── Assemble GAP: concatenate 64-element vectors from each worker ── */
static void assemble_gap(const int8_t *bufs[], int gap_per_worker, int8_t *output) {
    for (int w = 0; w < NUM_WORKERS; w++)
        memcpy(&output[w * gap_per_worker], bufs[w], gap_per_worker);
}

/* ── Run one distributed layer: wait_ready → broadcast → gather → assemble ── */
static bool run_distributed_layer(
    uint8_t layer,
    const int8_t *input_data, uint32_t input_size,
    uint32_t per_worker_result_size,
    int assemble_h, int assemble_w, int ch_per_worker,
    bool is_gap,
    int8_t *assembled_output,
    int64_t *out_bcast_us, int64_t *out_gather_us, int64_t *out_asm_us)
{
    /* Prepare ready + gather state before broadcast */
    reset_ready();
    reset_gather(per_worker_result_size);

    ESP_LOGI(TAG, "L%d: Broadcasting %lu bytes (%d chunks)...",
             layer, (unsigned long)input_size, swarm_num_chunks(input_size));
    broadcast_data(input_data, input_size, layer);
    *out_bcast_us = t_broadcast_end - t_broadcast_start;

    ESP_LOGI(TAG, "L%d: Broadcast %lld ms. Waiting for workers...",
             layer, *out_bcast_us / 1000);

    int64_t gather = wait_for_workers();
    if (gather < 0) return false;
    *out_gather_us = gather;

    /* Wait for all workers to signal READY (they finished sending + returned to main loop) */
    if (!wait_for_ready("L-ready")) return false;

    /* Decode sparse results before assembly */
    for (int i = 0; i < NUM_WORKERS; i++) {
        if (worker_sparse[i]) {
            memcpy(decode_tmp, worker_recv_buf[i], WORKER_RECV_BUF_SIZE);
            sparse_decode(decode_tmp, worker_recv_buf[i],
                          worker_sparse_orig[i], worker_sparse_zp[i]);
        }
    }

    int64_t t_asm = esp_timer_get_time();
    const int8_t *bufs[NUM_WORKERS];
    for (int i = 0; i < NUM_WORKERS; i++) bufs[i] = worker_recv_buf[i];

    if (is_gap)
        assemble_gap(bufs, ch_per_worker, assembled_output);
    else
        assemble_spatial(bufs, assemble_h, assemble_w, ch_per_worker, assembled_output);
    *out_asm_us = esp_timer_get_time() - t_asm;

    ESP_LOGI(TAG, "L%d: Gather %lld ms, Assemble %lld us",
             layer, *out_gather_us / 1000, *out_asm_us);

    for (int i = 0; i < NUM_WORKERS; i++)
        ESP_LOGI(TAG, "  W%d: %d chunks%s, done=%d",
                 i, worker_chunks_received[i],
                 worker_sparse[i] ? " (sparse)" : "",
                 worker_done[i]);
    return true;
}

void app_main(void) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  SwarmInfer Coordinator — Full Pipeline");
    ESP_LOGI(TAG, "========================================");
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    ESP_LOGI(TAG, "MAC: %02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    all_workers_done_sem  = xSemaphoreCreateBinary();
    all_workers_ready_sem = xSemaphoreCreateBinary();
    wifi_init();

    /* Allocate buffers */
    layer1_output = (int8_t *)malloc(L1_FULL_OUTPUT);   /* 16384 */
    layer2_output = (int8_t *)malloc(L2_FULL_OUTPUT);   /* 8192  */
    gap_output    = (int8_t *)malloc(L3_FULL_GAP);      /* 256   */
    dense1_out    = (int8_t *)malloc(DENSE1_OUT);       /* 128   */
    dense2_out    = (int8_t *)malloc(DENSE2_OUT);       /* 10    */
    for (int i = 0; i < NUM_WORKERS; i++)
        worker_recv_buf[i] = (int8_t *)malloc(WORKER_RECV_BUF_SIZE);
    decode_tmp = (int8_t *)malloc(WORKER_RECV_BUF_SIZE);

    if (!layer1_output || !layer2_output || !gap_output ||
        !dense1_out || !dense2_out || !decode_tmp) {
        ESP_LOGE(TAG, "MALLOC FAILED!"); return;
    }

    /* Precompute dense requant multipliers */
    FixedPointMultiplier dense1_m = compute_requant_multiplier(
        CONV3_OSCALE, DENSE1_KSCALE, DENSE1_OSCALE);
    FixedPointMultiplier dense2_m = compute_requant_multiplier(
        DENSE1_OSCALE, DENSE2_KSCALE, DENSE2_OSCALE);

    for (int i = 0; i < NUM_WORKERS; i++)
        ESP_LOGI(TAG, "Worker %d: %02X:%02X:%02X:%02X:%02X:%02X", i,
                 WORKER_MACS_ARR[i][0], WORKER_MACS_ARR[i][1],
                 WORKER_MACS_ARR[i][2], WORKER_MACS_ARR[i][3],
                 WORKER_MACS_ARR[i][4], WORKER_MACS_ARR[i][5]);

    ESP_LOGI(TAG, "Free heap: %lu", (unsigned long)esp_get_free_heap_size());

    /* Wait for all 4 workers to send READY */
    ESP_LOGI(TAG, "Waiting for workers to report READY...");
    reset_ready();
    if (!wait_for_ready("startup")) {
        ESP_LOGE(TAG, "Not all workers ready, aborting.");
        return;
    }
    ESP_LOGI(TAG, "All %d workers READY!", NUM_WORKERS);

    /* ================================================================
     *  Batch inference: BATCH_SIZE images
     * ================================================================ */
    int correct = 0;
    int total_inferred = 0;
    int64_t total_latency_us = 0;

    ESP_LOGI(TAG, "CSV_HEADER,config,img,label,pred,match,total_us,l1_bcast_us,l1_gather_us,l2_bcast_us,l2_gather_us,l3_bcast_us,l3_gather_us");

    for (int img = 0; img < BATCH_SIZE; img++) {
        int64_t t_total = esp_timer_get_time();
        int64_t bcast_us[3], gather_us[3], asm_us[3];

        /* ── Layer 1 ── */
        if (!run_distributed_layer(
                1, batch_images[img], L1_INPUT_SIZE,
                L1_POOL_SIZE,
                L1_POOL_H, L1_POOL_W, L1_WORKER_OC,
                false, layer1_output,
                &bcast_us[0], &gather_us[0], &asm_us[0])) {
            ESP_LOGW(TAG, "[%d/%d] L1 TIMEOUT — retry", img, BATCH_SIZE);
            img--;  /* retry this image */
            continue;
        }

        /* ── Layer 2 ── */
        if (!run_distributed_layer(
                2, layer1_output, L2_INPUT_SIZE,
                L2_POOL_SIZE,
                L2_POOL_H, L2_POOL_W, L2_WORKER_OC,
                false, layer2_output,
                &bcast_us[1], &gather_us[1], &asm_us[1])) {
            ESP_LOGW(TAG, "[%d/%d] L2 TIMEOUT — retry", img, BATCH_SIZE);
            img--;
            continue;
        }

        /* ── Layer 3 ── */
        if (!run_distributed_layer(
                3, layer2_output, L3_INPUT_SIZE,
                L3_GAP_SIZE,
                0, 0, L3_WORKER_OC,
                true, gap_output,
                &bcast_us[2], &gather_us[2], &asm_us[2])) {
            ESP_LOGW(TAG, "[%d/%d] L3 TIMEOUT — retry", img, BATCH_SIZE);
            img--;
            continue;
        }

        /* ── Dense layers ── */
        Tensor1D d1_in  = {gap_output, DENSE1_IN};
        Tensor1D d1_out = {dense1_out, DENSE1_OUT};
        dense_int8(&d1_in, dense1_kernel, dense1_bias, &d1_out,
                   dense1_m,
                   (int8_t)CONV3_OZP, (int8_t)DENSE1_KZP, (int8_t)DENSE1_OZP);
        Tensor1D d1_relu = {dense1_out, DENSE1_OUT};
        relu_int8_1d(&d1_relu, (int8_t)DENSE1_OZP);

        Tensor1D d2_in  = {dense1_out, DENSE2_IN};
        Tensor1D d2_out = {dense2_out, DENSE2_OUT};
        dense_int8(&d2_in, dense2_kernel, dense2_bias, &d2_out,
                   dense2_m,
                   (int8_t)DENSE1_OZP, (int8_t)DENSE2_KZP, (int8_t)DENSE2_OZP);

        int64_t total_us = esp_timer_get_time() - t_total;
        int predicted = argmax_int8(dense2_out, DENSE2_OUT);
        int label = batch_labels[img];
        bool match = (predicted == label);
        if (match) correct++;
        total_inferred++;
        total_latency_us += total_us;

        ESP_LOGI(TAG, "[%3d/%d] pred=%d(%s) true=%d(%s) %s  %lld ms",
                 img + 1, BATCH_SIZE,
                 predicted, cifar_classes[predicted],
                 label, cifar_classes[label],
                 match ? "OK" : "MISS",
                 total_us / 1000);

        // CSV data line for automated parsing
        ESP_LOGI(TAG, "CSV,fatcnn_n%d,%d,%d,%d,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld",
                 NUM_WORKERS, img, label, predicted, match ? 1 : 0,
                 total_us, bcast_us[0], gather_us[0],
                 bcast_us[1], gather_us[1],
                 bcast_us[2], gather_us[2]);
    }

    /* ── Summary ── */
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "============================================================");
    ESP_LOGI(TAG, "  BATCH ACCURACY RESULTS (FatCNN, N=%d workers)", NUM_WORKERS);
    ESP_LOGI(TAG, "============================================================");
    ESP_LOGI(TAG, "Images:    %d", total_inferred);
    ESP_LOGI(TAG, "Correct:   %d", correct);
    ESP_LOGI(TAG, "Accuracy:  %d.%d%%", (100 * correct) / total_inferred,
             ((1000 * correct) / total_inferred) % 10);
    ESP_LOGI(TAG, "Avg latency: %lld ms", total_latency_us / total_inferred / 1000);
    ESP_LOGI(TAG, "============================================================");
    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}
