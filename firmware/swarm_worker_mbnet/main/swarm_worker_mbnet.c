/* SwarmInfer: scaled MobileNet distributed WORKER (R2-11, Phase 5b).
 *
 * Holds this worker's output-channel shards of the 8 pointwise (1x1) layers
 * (mbnet_worker_<ID>_weights.h -> mb_shards[8]). On each pointwise round the
 * coordinator broadcasts the layer's input activation [N x Cin]; this worker
 * computes its channel shard (conv 1x1 -> [N x shard]) and returns it.
 *
 * Isolated: reuses common/{tensor_ops,mbnet_ops}.c; does not touch the FatCNN
 * worker. ESP-NOW protocol/MACs identical to swarm_worker.
 */
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
#include "esp_heap_caps.h"
#include "tensor_ops.h"
#include "mbnet_ops.h"
#include "swarm_protocol.h"

#ifndef WORKER_ID
#define WORKER_ID 0
#endif
#if   WORKER_ID == 0
#include "mbnet_worker_0_weights.h"
#elif WORKER_ID == 1
#include "mbnet_worker_1_weights.h"
#elif WORKER_ID == 2
#include "mbnet_worker_2_weights.h"
#elif WORKER_ID == 3
#include "mbnet_worker_3_weights.h"
#endif

static const char *TAG = "MBWORK";
static const uint8_t MAC_COORD[] = {0x02, 0x00, 0x00, 0x00, 0x00, 0x00};  /* PLACEHOLDER MAC — set to YOUR coordinator's MAC (each board prints its own MAC at boot via esp_read_mac / the get_mac sketch) */

/* Largest pointwise input = 48*48*32 = 73728; largest shard output =
 * 48*48*(1024/4 worst-case spatial)... sized generously in PSRAM. */
#define IN_BUF_SIZE   (256 * 1024)
#define OUT_BUF_SIZE  (256 * 1024)
static int8_t *input_buf = NULL;
static int8_t *output_buf = NULL;
static int8_t *encode_buf = NULL;   /* sparse-encode scratch; NOT touched by on_recv */
static int8_t *dense_buf = NULL;    /* decoded (dense) broadcast input */

static volatile int  input_unique_received = 0;
static volatile int  input_total_chunks = 0;
static uint8_t input_chunk_bitmap[64];   /* up to 512 chunks */
static volatile uint32_t current_input_size = 0;   /* bytes received (encoded if sparse) */
static volatile bool     current_input_sparse = false;
static volatile uint32_t current_input_orig = 0;   /* dense byte count */
static volatile int8_t   current_input_zp = 0;
static volatile bool compute_requested = false;
static volatile bool layer_started = false;
static volatile uint8_t current_pw = 0;        /* pointwise layer index 0..7 */
static SemaphoreHandle_t compute_sem;
static int64_t t_compute_start, t_compute_end;

static void on_sent(const esp_now_send_info_t *info, esp_now_send_status_t status) {
    (void)info; (void)status;
}

static void on_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (len < SWARM_HEADER_SIZE) return;
    const SwarmPacket *pkt = (const SwarmPacket *)data;
    switch (pkt->cmd) {
    case CMD_LAYER_START:
        current_pw = pkt->layer;
        layer_started = true;
        input_unique_received = 0;
        memset(input_chunk_bitmap, 0, sizeof(input_chunk_bitmap));
        compute_requested = false;
        /* LAYER_START: data[0..3]=received byte count; data[4]=sparse flag;
         * data[5..8]=dense byte count; data[9]=zero_point. */
        if (pkt->data_len >= 4)
            current_input_size = (uint32_t)pkt->data[0] | ((uint32_t)pkt->data[1] << 8)
                               | ((uint32_t)pkt->data[2] << 16) | ((uint32_t)pkt->data[3] << 24);
        if (pkt->data_len >= 10) {
            current_input_sparse = pkt->data[4] != 0;
            current_input_orig = (uint32_t)pkt->data[5] | ((uint32_t)pkt->data[6] << 8)
                               | ((uint32_t)pkt->data[7] << 16) | ((uint32_t)pkt->data[8] << 24);
            current_input_zp = (int8_t)pkt->data[9];
        } else {
            current_input_sparse = false;
            current_input_orig = current_input_size;
        }
        break;
    case CMD_INPUT_CHUNK:
        if (!layer_started) break;
        if (pkt->chunk_id < pkt->total_chunks && input_buf) {
            uint32_t offset = (uint32_t)pkt->chunk_id * SWARM_CHUNK_SIZE;
            if (offset + pkt->data_len <= IN_BUF_SIZE)
                memcpy(&input_buf[offset], pkt->data, pkt->data_len);
            input_total_chunks = pkt->total_chunks;
            /* count UNIQUE chunks only (redundant sends must not double-count) */
            int byte = pkt->chunk_id >> 3, bit = 1 << (pkt->chunk_id & 7);
            if (!(input_chunk_bitmap[byte] & bit)) {
                input_chunk_bitmap[byte] |= bit;
                input_unique_received++;
                if (input_unique_received >= input_total_chunks && !compute_requested) {
                    compute_requested = true;
                    xSemaphoreGiveFromISR(compute_sem, NULL);
                }
            }
        }
        break;
    case CMD_COMPUTE:
        /* fallback only if input is actually complete (never compute on partial) */
        if (!compute_requested && layer_started &&
            input_total_chunks > 0 && input_unique_received >= input_total_chunks) {
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

static void send_ready(void) {
    SwarmPacket pkt = {0};
    pkt.cmd = CMD_WORKER_READY;
    pkt.chunk_id = WORKER_ID;
    pkt.data_len = 0;
    esp_err_t err;
    do { err = esp_now_send(MAC_COORD, (uint8_t *)&pkt, SWARM_HEADER_SIZE);
         if (err == ESP_ERR_ESPNOW_NO_MEM) esp_rom_delay_us(500);
    } while (err == ESP_ERR_ESPNOW_NO_MEM);
}

static void send_result(const int8_t *data, uint32_t data_len, uint8_t layer,
                        uint8_t sparse_flag, uint16_t orig_size, int8_t zp) {
    uint16_t total = swarm_num_chunks(data_len);
    SwarmPacket pkt;
    pkt.cmd = CMD_RESULT_CHUNK;
    pkt.layer = layer;
    pkt.total_chunks = total;
    /* Unicast gather: ESP-NOW unicast is MAC-ACKed + auto-retransmitted, so one
     * pass suffices (the coordinator's re-broadcast still recovers a genuine
     * stall). Redundancy=2 here was the main gather cost under 4-worker congestion. */
    for (int rep = 0; rep < 1; rep++) {
        for (uint16_t i = 0; i < total; i++) {
            pkt.chunk_id = i;
            uint32_t offset = (uint32_t)i * SWARM_CHUNK_SIZE;
            uint32_t remain = data_len - offset;   /* 32-bit: avoid the >65535 overflow */
            pkt.data_len = (remain > SWARM_CHUNK_SIZE) ? SWARM_CHUNK_SIZE : remain;
            memcpy(pkt.data, &data[offset], pkt.data_len);
            esp_err_t err;
            do { err = esp_now_send(MAC_COORD, (uint8_t *)&pkt, SWARM_HEADER_SIZE + pkt.data_len);
                 if (err == ESP_ERR_ESPNOW_NO_MEM) esp_rom_delay_us(500);
            } while (err == ESP_ERR_ESPNOW_NO_MEM);
            esp_rom_delay_us(400);
        }
    }
    esp_rom_delay_us(2000);
    uint32_t compute_us = (uint32_t)(t_compute_end - t_compute_start);
    SwarmPacket done = {0};
    done.cmd = CMD_RESULT_DONE;
    done.layer = layer;
    done.chunk_id = WORKER_ID;
    /* sparse metadata: data[0]=flag, data[1..2]=orig_size(LE), data[3]=zero_point */
    done.data[0] = sparse_flag;
    done.data[1] = (uint8_t)(orig_size & 0xFF);
    done.data[2] = (uint8_t)((orig_size >> 8) & 0xFF);
    done.data[3] = (uint8_t)zp;
    done.data[4] = (uint8_t)(compute_us & 0xFF);
    done.data[5] = (uint8_t)((compute_us >> 8) & 0xFF);
    done.data[6] = (uint8_t)((compute_us >> 16) & 0xFF);
    done.data[7] = (uint8_t)((compute_us >> 24) & 0xFF);
    done.data_len = 12;
    esp_err_t err;
    do { err = esp_now_send(MAC_COORD, (uint8_t *)&done, SWARM_HEADER_SIZE + done.data_len);
         if (err == ESP_ERR_ESPNOW_NO_MEM) esp_rom_delay_us(500);
    } while (err == ESP_ERR_ESPNOW_NO_MEM);
}

/* Compute this worker's shard of pointwise layer k: 1x1 conv over [N,Cin]. */
static void compute_pw(uint8_t k) {
    const MbShard *S = &mb_shards[k];
    int shard = S->c_end - S->c_start;
    /* Decode the broadcast input if it came sparse; N is from the DENSE length. */
    int8_t *cin = input_buf;
    if (current_input_sparse) {
        sparse_decode(input_buf, dense_buf, current_input_orig, current_input_zp);
        cin = dense_buf;
    }
    int N = (S->Cin > 0) ? (int)(current_input_orig / S->Cin) : 0;

    Tensor3D in  = { cin, N, 1, S->Cin };
    Tensor3D out = { output_buf, N, 1, shard };
    FixedPointMultiplier m = { S->mult, S->shift };
    t_compute_start = esp_timer_get_time();
    conv2d_int8(&in, S->w, S->b, &out, 1, 1, 1, 0, m, S->in_zp, S->w_zp, S->out_zp);
    relu_int8(&out, S->out_zp);
    t_compute_end = esp_timer_get_time();
    /* Sparsify the post-ReLU shard (input_buf is free now -> use as encode scratch).
     * Send sparse only if it actually shrinks (encoded = bitmap + non-zeros). */
    uint32_t result_size = (uint32_t)N * shard;
    uint32_t enc_size = sparse_encode(output_buf, result_size, encode_buf, S->out_zp);
    if (enc_size < result_size)
        send_result(encode_buf, enc_size, k, 1, (uint16_t)result_size, S->out_zp);
    else
        send_result(output_buf, result_size, k, 0, (uint16_t)result_size, S->out_zp);
}

void app_main(void) {
    ESP_LOGI(TAG, "MobileNet worker %d (8 pointwise shards)", WORKER_ID);
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    ESP_LOGI(TAG, "MAC: %02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    compute_sem = xSemaphoreCreateBinary();
    wifi_init();
    input_buf = heap_caps_malloc(IN_BUF_SIZE, MALLOC_CAP_SPIRAM);
    output_buf = heap_caps_malloc(OUT_BUF_SIZE, MALLOC_CAP_SPIRAM);
    encode_buf = heap_caps_malloc(OUT_BUF_SIZE, MALLOC_CAP_SPIRAM);
    dense_buf = heap_caps_malloc(IN_BUF_SIZE, MALLOC_CAP_SPIRAM);
    if (!input_buf || !output_buf || !encode_buf || !dense_buf) {
        ESP_LOGE(TAG, "PSRAM alloc failed"); return;
    }

    ESP_LOGI(TAG, "Worker %d ready (%d pointwise shards).", WORKER_ID, MB_NUM_PW);
    vTaskDelay(pdMS_TO_TICKS(100));
    send_ready();

    while (1) {
        if (xSemaphoreTake(compute_sem, pdMS_TO_TICKS(5000)) == pdTRUE) {
            if (input_unique_received < input_total_chunks) {
                ESP_LOGW(TAG, "pw%d incomplete %d/%d", current_pw,
                         input_unique_received, input_total_chunks);
                continue;
            }
            compute_pw(current_pw);
            layer_started = false;
            send_ready();
        } else {
            send_ready();  /* heartbeat so coordinator can resync */
        }
    }
}
