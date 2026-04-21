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

#ifndef WORKER_ID
#define WORKER_ID 3
#endif

/* ── Per-worker weight macros (conv1 / conv2 / conv3) ── */
#if WORKER_ID == 0
#include "worker_0_weights.h"
#define MY_CONV1_KERNEL conv1_w0_kernel
#define MY_CONV1_BIAS   conv1_w0_bias
#define MY_CONV1_KSCALE CONV1_W0_KSCALE
#define MY_CONV1_KZP    CONV1_W0_KZP
#define MY_CONV1_OSCALE CONV1_W0_OSCALE
#define MY_CONV1_OZP    CONV1_W0_OZP
#define MY_CONV2_KERNEL conv2_w0_kernel
#define MY_CONV2_BIAS   conv2_w0_bias
#define MY_CONV2_KSCALE CONV2_W0_KSCALE
#define MY_CONV2_KZP    CONV2_W0_KZP
#define MY_CONV2_OSCALE CONV2_W0_OSCALE
#define MY_CONV2_OZP    CONV2_W0_OZP
#define MY_CONV3_KERNEL conv3_w0_kernel
#define MY_CONV3_BIAS   conv3_w0_bias
#define MY_CONV3_KSCALE CONV3_W0_KSCALE
#define MY_CONV3_KZP    CONV3_W0_KZP
#define MY_CONV3_OSCALE CONV3_W0_OSCALE
#define MY_CONV3_OZP    CONV3_W0_OZP
#elif WORKER_ID == 1
#include "worker_1_weights.h"
#define MY_CONV1_KERNEL conv1_w1_kernel
#define MY_CONV1_BIAS   conv1_w1_bias
#define MY_CONV1_KSCALE CONV1_W1_KSCALE
#define MY_CONV1_KZP    CONV1_W1_KZP
#define MY_CONV1_OSCALE CONV1_W1_OSCALE
#define MY_CONV1_OZP    CONV1_W1_OZP
#define MY_CONV2_KERNEL conv2_w1_kernel
#define MY_CONV2_BIAS   conv2_w1_bias
#define MY_CONV2_KSCALE CONV2_W1_KSCALE
#define MY_CONV2_KZP    CONV2_W1_KZP
#define MY_CONV2_OSCALE CONV2_W1_OSCALE
#define MY_CONV2_OZP    CONV2_W1_OZP
#define MY_CONV3_KERNEL conv3_w1_kernel
#define MY_CONV3_BIAS   conv3_w1_bias
#define MY_CONV3_KSCALE CONV3_W1_KSCALE
#define MY_CONV3_KZP    CONV3_W1_KZP
#define MY_CONV3_OSCALE CONV3_W1_OSCALE
#define MY_CONV3_OZP    CONV3_W1_OZP
#elif WORKER_ID == 2
#include "worker_2_weights.h"
#define MY_CONV1_KERNEL conv1_w2_kernel
#define MY_CONV1_BIAS   conv1_w2_bias
#define MY_CONV1_KSCALE CONV1_W2_KSCALE
#define MY_CONV1_KZP    CONV1_W2_KZP
#define MY_CONV1_OSCALE CONV1_W2_OSCALE
#define MY_CONV1_OZP    CONV1_W2_OZP
#define MY_CONV2_KERNEL conv2_w2_kernel
#define MY_CONV2_BIAS   conv2_w2_bias
#define MY_CONV2_KSCALE CONV2_W2_KSCALE
#define MY_CONV2_KZP    CONV2_W2_KZP
#define MY_CONV2_OSCALE CONV2_W2_OSCALE
#define MY_CONV2_OZP    CONV2_W2_OZP
#define MY_CONV3_KERNEL conv3_w2_kernel
#define MY_CONV3_BIAS   conv3_w2_bias
#define MY_CONV3_KSCALE CONV3_W2_KSCALE
#define MY_CONV3_KZP    CONV3_W2_KZP
#define MY_CONV3_OSCALE CONV3_W2_OSCALE
#define MY_CONV3_OZP    CONV3_W2_OZP
#elif WORKER_ID == 3
#include "worker_3_weights.h"
#define MY_CONV1_KERNEL conv1_w3_kernel
#define MY_CONV1_BIAS   conv1_w3_bias
#define MY_CONV1_KSCALE CONV1_W3_KSCALE
#define MY_CONV1_KZP    CONV1_W3_KZP
#define MY_CONV1_OSCALE CONV1_W3_OSCALE
#define MY_CONV1_OZP    CONV1_W3_OZP
#define MY_CONV2_KERNEL conv2_w3_kernel
#define MY_CONV2_BIAS   conv2_w3_bias
#define MY_CONV2_KSCALE CONV2_W3_KSCALE
#define MY_CONV2_KZP    CONV2_W3_KZP
#define MY_CONV2_OSCALE CONV2_W3_OSCALE
#define MY_CONV2_OZP    CONV2_W3_OZP
#define MY_CONV3_KERNEL conv3_w3_kernel
#define MY_CONV3_BIAS   conv3_w3_bias
#define MY_CONV3_KSCALE CONV3_W3_KSCALE
#define MY_CONV3_KZP    CONV3_W3_KZP
#define MY_CONV3_OSCALE CONV3_W3_OSCALE
#define MY_CONV3_OZP    CONV3_W3_OZP
#endif

static const char *TAG = "WORKER";
static const uint8_t MAC_COORD[] = {0xB8, 0xF8, 0x62, 0xE2, 0xD0, 0x8C};

/* ── Shared buffers (reused across layers) ── */
static int8_t *input_buf = NULL;   /* max(L1_INPUT, L2_INPUT, L3_INPUT) = L2 = 16384 */
static int8_t *conv_out  = NULL;   /* max(L1_CONV, L2_CONV, L3_CONV)   = L1 = 16384 */
static int8_t *pool_out  = NULL;   /* max(L1_POOL, L2_POOL, L3_GAP)    = L1 = 4096  */

#define INPUT_BUF_SIZE  L2_INPUT_SIZE   /* 16384 — largest input  */
#define CONV_BUF_SIZE   L1_CONV_SIZE    /* 16384 — largest conv   */
#define POOL_BUF_SIZE   L1_POOL_SIZE    /* 4096  — largest output */

/* ── Receive state ── */
static volatile int  input_chunks_received = 0;
static volatile int  input_total_chunks = 0;
static volatile int  current_input_size = 0;
static volatile bool compute_requested = false;
static volatile bool layer_started = false;
static volatile uint8_t current_layer = 0;
static SemaphoreHandle_t compute_sem;

static int64_t t_compute_start, t_compute_end;
static int64_t t_send_start, t_send_end;

static void on_sent(const esp_now_send_info_t *info, esp_now_send_status_t status) {
    (void)info; (void)status;
}

static void on_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (len < SWARM_HEADER_SIZE) return;
    const SwarmPacket *pkt = (const SwarmPacket *)data;

    switch (pkt->cmd) {
    case CMD_LAYER_START:
        current_layer = pkt->layer;
        layer_started = true;
        input_chunks_received = 0;
        compute_requested = false;
        if (pkt->layer == 1)      current_input_size = L1_INPUT_SIZE;
        else if (pkt->layer == 2) current_input_size = L2_INPUT_SIZE;
        else if (pkt->layer == 3) current_input_size = L3_INPUT_SIZE;
        break;

    case CMD_INPUT_CHUNK:
        if (!layer_started) break;
        if (pkt->chunk_id < pkt->total_chunks && input_buf) {
            uint32_t offset = (uint32_t)pkt->chunk_id * SWARM_CHUNK_SIZE;
            uint16_t copy_len = pkt->data_len;
            if (offset + copy_len <= (uint32_t)current_input_size) {
                memcpy(&input_buf[offset], pkt->data, copy_len);
            }
            input_chunks_received++;
            input_total_chunks = pkt->total_chunks;
            /* Auto-trigger: start compute as soon as all chunks arrived */
            if (input_chunks_received >= input_total_chunks && !compute_requested) {
                compute_requested = true;
                xSemaphoreGiveFromISR(compute_sem, NULL);
            }
        }
        break;

    case CMD_COMPUTE:
        /* Fallback if auto-trigger missed (e.g. duplicate/reordered chunks) */
        if (!compute_requested && layer_started) {
            compute_requested = true;
            xSemaphoreGiveFromISR(compute_sem, NULL);
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
    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, MAC_COORD, 6);
    peer.channel = 1; peer.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
}

/* Sparse send metadata (set by caller before send_result) */
static uint8_t sparse_send_flag = 0;
static uint16_t sparse_send_orig = 0;
static int8_t sparse_send_zp = 0;

static void send_result(const int8_t *data, uint32_t data_len, uint8_t layer) {
    t_send_start = esp_timer_get_time();
    uint16_t total = swarm_num_chunks(data_len);
    SwarmPacket pkt;
    pkt.cmd = CMD_RESULT_CHUNK;
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
            err = esp_now_send(MAC_COORD, (uint8_t *)&pkt, SWARM_HEADER_SIZE + pkt.data_len);
            if (err == ESP_ERR_ESPNOW_NO_MEM) esp_rom_delay_us(500);
        } while (err == ESP_ERR_ESPNOW_NO_MEM);
        esp_rom_delay_us(500);
    }
    /* Small delay to let TX queue drain before sending DONE */
    esp_rom_delay_us(2000);
    SwarmPacket done_pkt = {0};
    done_pkt.cmd = CMD_RESULT_DONE;
    done_pkt.layer = layer;
    done_pkt.chunk_id = WORKER_ID;
    /* Sparse metadata: data[0]=flag, data[1..2]=original_size(LE), data[3]=zero_point */
    done_pkt.data[0] = sparse_send_flag;
    done_pkt.data[1] = (uint8_t)(sparse_send_orig & 0xFF);
    done_pkt.data[2] = (uint8_t)((sparse_send_orig >> 8) & 0xFF);
    done_pkt.data[3] = (uint8_t)sparse_send_zp;
    done_pkt.data_len = 4;
    esp_err_t err;
    do {
        err = esp_now_send(MAC_COORD, (uint8_t *)&done_pkt, SWARM_HEADER_SIZE + 4);
        if (err == ESP_ERR_ESPNOW_NO_MEM) esp_rom_delay_us(500);
    } while (err == ESP_ERR_ESPNOW_NO_MEM);
    t_send_end = esp_timer_get_time();
}

static void send_ready(void) {
    SwarmPacket pkt = {0};
    pkt.cmd = CMD_WORKER_READY;
    pkt.chunk_id = WORKER_ID;
    pkt.data_len = 0;
    esp_err_t err;
    do {
        err = esp_now_send(MAC_COORD, (uint8_t *)&pkt, SWARM_HEADER_SIZE);
        if (err == ESP_ERR_ESPNOW_NO_MEM) esp_rom_delay_us(500);
    } while (err == ESP_ERR_ESPNOW_NO_MEM);
}

/* ── Layer compute functions ── */

static void compute_layer1(FixedPointMultiplier m) {
    Tensor3D inp  = {input_buf, L1_INPUT_H, L1_INPUT_W, L1_INPUT_C};
    Tensor3D cout = {conv_out,  L1_CONV_H,  L1_CONV_W,  L1_WORKER_OC};
    conv2d_int8(&inp, MY_CONV1_KERNEL, MY_CONV1_BIAS, &cout,
                L1_KERNEL_H, L1_KERNEL_W, L1_STRIDE, L1_PADDING,
                m, (int8_t)(-128), (int8_t)MY_CONV1_KZP, (int8_t)MY_CONV1_OZP);

    Tensor3D rt = {conv_out, L1_CONV_H, L1_CONV_W, L1_WORKER_OC};
    relu_int8(&rt, (int8_t)MY_CONV1_OZP);

    Tensor3D pi = {conv_out, L1_CONV_H, L1_CONV_W, L1_WORKER_OC};
    Tensor3D po = {pool_out, L1_POOL_H, L1_POOL_W, L1_WORKER_OC};
    maxpool2x2_int8(&pi, &po);
}

static void compute_layer2(FixedPointMultiplier m) {
    Tensor3D inp  = {input_buf, L2_INPUT_H, L2_INPUT_W, L2_INPUT_C};
    Tensor3D cout = {conv_out,  L2_CONV_H,  L2_CONV_W,  L2_WORKER_OC};
    conv2d_int8(&inp, MY_CONV2_KERNEL, MY_CONV2_BIAS, &cout,
                L2_KERNEL_H, L2_KERNEL_W, L2_STRIDE, L2_PADDING,
                m, (int8_t)MY_CONV1_OZP, (int8_t)MY_CONV2_KZP, (int8_t)MY_CONV2_OZP);

    Tensor3D rt = {conv_out, L2_CONV_H, L2_CONV_W, L2_WORKER_OC};
    relu_int8(&rt, (int8_t)MY_CONV2_OZP);

    Tensor3D pi = {conv_out, L2_CONV_H, L2_CONV_W, L2_WORKER_OC};
    Tensor3D po = {pool_out, L2_POOL_H, L2_POOL_W, L2_WORKER_OC};
    maxpool2x2_int8(&pi, &po);
}

static void compute_layer3(FixedPointMultiplier m) {
    Tensor3D inp  = {input_buf, L3_INPUT_H, L3_INPUT_W, L3_INPUT_C};
    Tensor3D cout = {conv_out,  L3_CONV_H,  L3_CONV_W,  L3_WORKER_OC};
    conv2d_int8(&inp, MY_CONV3_KERNEL, MY_CONV3_BIAS, &cout,
                L3_KERNEL_H, L3_KERNEL_W, L3_STRIDE, L3_PADDING,
                m, (int8_t)MY_CONV2_OZP, (int8_t)MY_CONV3_KZP, (int8_t)MY_CONV3_OZP);

    Tensor3D rt = {conv_out, L3_CONV_H, L3_CONV_W, L3_WORKER_OC};
    relu_int8(&rt, (int8_t)MY_CONV3_OZP);

    Tensor3D gi  = {conv_out, L3_CONV_H, L3_CONV_W, L3_WORKER_OC};
    Tensor1D go  = {pool_out, L3_WORKER_OC};
    global_avgpool_int8(&gi, &go,
                        (int8_t)MY_CONV3_OZP, MY_CONV3_OSCALE,
                        MY_CONV3_OSCALE, (int8_t)MY_CONV3_OZP);
}

void app_main(void) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  SwarmInfer Worker %d (3-layer)", WORKER_ID);
    ESP_LOGI(TAG, "========================================");
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    ESP_LOGI(TAG, "MAC: %02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    compute_sem = xSemaphoreCreateBinary();
    wifi_init();

    input_buf = tensor_alloc(INPUT_BUF_SIZE);
    conv_out  = tensor_alloc(CONV_BUF_SIZE);
    pool_out  = tensor_alloc(POOL_BUF_SIZE);
    if (!input_buf || !conv_out || !pool_out) {
        ESP_LOGE(TAG, "MALLOC FAILED!"); return;
    }

    ESP_LOGI(TAG, "Buffers: input=%d conv=%d pool=%d",
             INPUT_BUF_SIZE, CONV_BUF_SIZE, POOL_BUF_SIZE);
    ESP_LOGI(TAG, "Free heap: %lu", (unsigned long)esp_get_free_heap_size());

    FixedPointMultiplier conv1_m = compute_requant_multiplier(
        0.00392157f, MY_CONV1_KSCALE, MY_CONV1_OSCALE);
    FixedPointMultiplier conv2_m = compute_requant_multiplier(
        MY_CONV1_OSCALE, MY_CONV2_KSCALE, MY_CONV2_OSCALE);
    FixedPointMultiplier conv3_m = compute_requant_multiplier(
        MY_CONV2_OSCALE, MY_CONV3_KSCALE, MY_CONV3_OSCALE);

    /* Pre-fill input buffer with -128 (zero_point for all layers).
     * Missing broadcast chunks will contain neutral values. */
    memset(input_buf, (int8_t)-128, INPUT_BUF_SIZE);

    ESP_LOGI(TAG, "Worker %d ready. Waiting for layers...", WORKER_ID);

    vTaskDelay(pdMS_TO_TICKS(100));
    send_ready();

    while (1) {
        if (xSemaphoreTake(compute_sem, pdMS_TO_TICKS(5000)) != pdTRUE) {
            ESP_LOGW(TAG, "No work yet, re-sending READY...");
            send_ready();
            continue;
        }
        if (!compute_requested) continue;

        uint8_t layer = current_layer;
        int expected = swarm_num_chunks(current_input_size);
        if (input_chunks_received < expected) {
            ESP_LOGW(TAG, "L%d: INCOMPLETE input! Got %d/%d chunks",
                     layer, input_chunks_received, expected);
        }
        ESP_LOGI(TAG, "L%d: input %d/%d chunks. Computing...",
                 layer, input_chunks_received, expected);

        t_compute_start = esp_timer_get_time();
        uint32_t result_size = 0;

        switch (layer) {
        case 1:
            compute_layer1(conv1_m);
            result_size = L1_POOL_SIZE;
            break;
        case 2:
            compute_layer2(conv2_m);
            result_size = L2_POOL_SIZE;
            break;
        case 3:
            compute_layer3(conv3_m);
            result_size = L3_GAP_SIZE;
            break;
        default:
            ESP_LOGE(TAG, "Unknown layer %d", layer);
            continue;
        }

        t_compute_end = esp_timer_get_time();
        ESP_LOGI(TAG, "L%d compute: %lld ms",
                 layer, (t_compute_end - t_compute_start) / 1000);

        /* Sparse encode: use input_buf as scratch (not needed after compute) */
        int8_t output_zp;
        switch (layer) {
        case 1: output_zp = (int8_t)MY_CONV1_OZP; break;
        case 2: output_zp = (int8_t)MY_CONV2_OZP; break;
        default: output_zp = (int8_t)MY_CONV3_OZP; break;
        }

        /* Count zeros in pool_out (post-ReLU) for sparsity telemetry.
         * A value equal to output_zp represents zero in the real-valued tensor. */
        uint32_t zero_count = 0;
        for (uint32_t i = 0; i < result_size; i++) {
            if (pool_out[i] == output_zp) zero_count++;
        }
        uint32_t sparsity_ppm = (uint32_t)((uint64_t)zero_count * 1000000ULL / result_size);
        /* Machine-parseable sparsity line (consumed by scripts/analyze_logs.py).
         * Format: CSV_SPARSE,layer,worker_id,result_size,zero_count,sparsity_ppm */
        printf("CSV_SPARSE,%d,%d,%lu,%lu,%lu\n",
               layer, WORKER_ID,
               (unsigned long)result_size,
               (unsigned long)zero_count,
               (unsigned long)sparsity_ppm);

        uint32_t enc_size = sparse_encode(pool_out, result_size, input_buf, output_zp);
        bool use_sparse = (enc_size < result_size);

        if (use_sparse) {
            uint32_t nz = enc_size - (result_size + 7) / 8;
            ESP_LOGI(TAG, "L%d sparse: %lu->%lu B (%lu%% zero). Sending...",
                     layer, (unsigned long)result_size, (unsigned long)enc_size,
                     (unsigned long)(100 - 100 * nz / result_size));
            sparse_send_flag = 1;
            sparse_send_orig = (uint16_t)result_size;
            sparse_send_zp = output_zp;
            send_result(input_buf, enc_size, layer);
        } else {
            ESP_LOGI(TAG, "L%d dense: %lu B (no sparsity). Sending...",
                     layer, (unsigned long)result_size);
            sparse_send_flag = 0;
            sparse_send_orig = (uint16_t)result_size;
            sparse_send_zp = output_zp;
            send_result(pool_out, result_size, layer);
        }
        ESP_LOGI(TAG, "L%d send: %lld ms",
                 layer, (t_send_end - t_send_start) / 1000);

        /* Re-fill input buffer with zero_point for next layer */
        memset(input_buf, (int8_t)-128, INPUT_BUF_SIZE);

        /* Signal coordinator: ready for next layer */
        send_ready();
    }
}
