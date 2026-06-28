/* SwarmInfer: scaled MobileNet distributed COORDINATOR (R2-11, Phase 5b).
 *
 * Drives the 18-layer INT8 MobileNet graph. Local layers (conv0, the 8
 * depthwise convs, GAP, dense) run on the coordinator via mb_run_conv_layer;
 * the 8 pointwise (1x1) layers are distributed: broadcast the activation, each
 * worker returns its output-channel shard, the coordinator concatenates them.
 * Self-validates the whole run against the numpy/single-node reference
 * (mb_checksums in mbnet_testvec.h).
 *
 * Isolated: reuses common/{tensor_ops,mbnet_ops}.c + the SwarmPacket/ESP-NOW
 * protocol; touches no FatCNN code. Activations in octal PSRAM.
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
#include "mbnet_coord_weights.h"
#include "mbnet_testvec.h"

#ifndef NUM_WORKERS
#define NUM_WORKERS 4
#endif

static const char *TAG = "MBCOORD";
static const uint8_t ALL_WORKER_MACS[4][6] = {
    /* PLACEHOLDER MACs — set to YOUR workers' MACs (each board prints its own MAC at boot) */
    {0x02, 0x00, 0x00, 0x00, 0x00, 0x01}, {0x02, 0x00, 0x00, 0x00, 0x00, 0x02},
    {0x02, 0x00, 0x00, 0x00, 0x00, 0x03}, {0x02, 0x00, 0x00, 0x00, 0x00, 0x04},
};
static const uint8_t BROADCAST_ADDR[] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};

#define ACT_BUF  (256 * 1024)
#define RECV_BUF (96 * 1024)
static int8_t *buf_a, *buf_b, *buf_pad;
static int8_t *worker_recv_buf[NUM_WORKERS];
static int8_t *decode_tmp;   /* scratch for sparse-decode of a gathered shard */
static int8_t *bcast_enc;    /* scratch for sparse-encode of the broadcast input */

static volatile bool worker_done[NUM_WORKERS];
static volatile int  workers_complete;
static volatile uint32_t current_result_size;
static volatile uint32_t worker_compute_us[NUM_WORKERS];
static uint8_t worker_result_bitmap[NUM_WORKERS][64];   /* unique result chunks */
static volatile int worker_result_unique[NUM_WORKERS];
static volatile int worker_expected[NUM_WORKERS];       /* per-worker chunk count (sparse varies) */
static volatile bool worker_sparse[NUM_WORKERS];
static volatile uint16_t worker_sparse_orig[NUM_WORKERS];
static volatile int8_t worker_sparse_zp[NUM_WORKERS];
static volatile uint8_t cur_round;   /* current pw round; reject stale chunks */
static SemaphoreHandle_t all_done_sem;

static volatile bool worker_ready[NUM_WORKERS];
static volatile int  workers_ready_count;
static SemaphoreHandle_t all_ready_sem;
static int64_t g_wait_us = 0;   /* PROFILE: time spent in inter-round wait_ready() */

static int mac_to_wid(const uint8_t *mac) {
    for (int i = 0; i < NUM_WORKERS; i++)
        if (memcmp(mac, ALL_WORKER_MACS[i], 6) == 0) return i;
    return -1;
}

static void on_sent(const esp_now_send_info_t *info, esp_now_send_status_t status) {
    (void)info; (void)status;
}

static void on_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (len < SWARM_HEADER_SIZE) return;
    const SwarmPacket *pkt = (const SwarmPacket *)data;
    int wid = mac_to_wid(info->src_addr);
    if (wid < 0) return;
    switch (pkt->cmd) {
    case CMD_WORKER_READY:
        if (!worker_ready[wid]) {
            worker_ready[wid] = true;
            if (++workers_ready_count >= NUM_WORKERS)
                xSemaphoreGiveFromISR(all_ready_sem, NULL);
        }
        break;
    case CMD_RESULT_CHUNK:
        if (pkt->layer != cur_round) break;   /* reject stragglers from other rounds */
        if (pkt->chunk_id < pkt->total_chunks && worker_recv_buf[wid]) {
            worker_expected[wid] = pkt->total_chunks;   /* sparse: count comes from the worker */
            uint32_t offset = (uint32_t)pkt->chunk_id * SWARM_CHUNK_SIZE;
            if (offset + pkt->data_len <= current_result_size)
                memcpy(&worker_recv_buf[wid][offset], pkt->data, pkt->data_len);
            int byte = pkt->chunk_id >> 3, bit = 1 << (pkt->chunk_id & 7);
            if (!(worker_result_bitmap[wid][byte] & bit)) {
                worker_result_bitmap[wid][byte] |= bit;
                worker_result_unique[wid]++;
            }
        }
        break;
    case CMD_RESULT_DONE:
        if (pkt->layer != cur_round) break;
        /* only count complete: a worker isn't done until ALL its result chunks
         * arrived (count is per-worker since sparse encoding varies the size) */
        if (!worker_done[wid] && worker_expected[wid] > 0 &&
            worker_result_unique[wid] >= worker_expected[wid]) {
            worker_done[wid] = true;
            if (pkt->data_len >= 4 && pkt->data[0] == 1) {
                worker_sparse[wid] = true;
                worker_sparse_orig[wid] = (uint16_t)(pkt->data[1] | (pkt->data[2] << 8));
                worker_sparse_zp[wid] = (int8_t)pkt->data[3];
            } else {
                worker_sparse[wid] = false;
            }
            if (pkt->data_len >= 12)
                worker_compute_us[wid] = (uint32_t)pkt->data[4] | ((uint32_t)pkt->data[5] << 8)
                    | ((uint32_t)pkt->data[6] << 16) | ((uint32_t)pkt->data[7] << 24);
            if (++workers_complete >= NUM_WORKERS)
                xSemaphoreGiveFromISR(all_done_sem, NULL);
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
    esp_now_peer_info_t bc = {0};
    memcpy(bc.peer_addr, BROADCAST_ADDR, 6);
    bc.channel = 1; bc.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&bc));
    for (int i = 0; i < NUM_WORKERS; i++) {
        esp_now_peer_info_t p = {0};
        memcpy(p.peer_addr, ALL_WORKER_MACS[i], 6);
        p.channel = 1; p.encrypt = false;
        ESP_ERROR_CHECK(esp_now_add_peer(&p));
    }
}

static bool wait_ready(void) {
    for (int i = 0; i < NUM_WORKERS; i++) worker_ready[i] = false;
    workers_ready_count = 0;
    xSemaphoreTake(all_ready_sem, 0);
    return xSemaphoreTake(all_ready_sem, pdMS_TO_TICKS(15000)) == pdTRUE;
}

/* data_len = bytes actually broadcast (encoded if sparse). orig_len/zp let the
 * worker decode; sparse=0 means data_len==orig_len (dense). */
static void send_layer_start(uint32_t data_len, uint8_t k,
                             uint8_t sparse, uint32_t orig_len, int8_t zp) {
    uint16_t total = swarm_num_chunks(data_len);
    SwarmPacket s = {0};
    s.cmd = CMD_LAYER_START; s.layer = k; s.total_chunks = total; s.data_len = 10;
    s.data[0] = data_len & 0xFF; s.data[1] = (data_len >> 8) & 0xFF;
    s.data[2] = (data_len >> 16) & 0xFF; s.data[3] = (data_len >> 24) & 0xFF;
    s.data[4] = sparse;
    s.data[5] = orig_len & 0xFF; s.data[6] = (orig_len >> 8) & 0xFF;
    s.data[7] = (orig_len >> 16) & 0xFF; s.data[8] = (orig_len >> 24) & 0xFF;
    s.data[9] = (uint8_t)zp;
    esp_now_send(BROADCAST_ADDR, (uint8_t *)&s, SWARM_HEADER_SIZE + 10);
    vTaskDelay(pdMS_TO_TICKS(5));
}

/* Send all input chunks `redundancy` times (each chunk paced); workers track
 * unique chunks via bitmap so duplicates are harmless. Ends with COMPUTE. */
static void send_chunks(const int8_t *data, uint32_t data_len, uint8_t k, int redundancy) {
    uint16_t total = swarm_num_chunks(data_len);
    SwarmPacket p; p.cmd = CMD_INPUT_CHUNK; p.layer = k; p.total_chunks = total;
    for (int r = 0; r < redundancy; r++) {
        for (uint16_t i = 0; i < total; i++) {
            p.chunk_id = i;
            uint32_t off = (uint32_t)i * SWARM_CHUNK_SIZE;
            uint32_t rem = data_len - off;   /* 32-bit: data_len can exceed 65535
                * (b1_pw input = 73728), so a uint16_t here overflows and truncates
                * one chunk's data_len -> silent input corruption. */
            p.data_len = (rem > SWARM_CHUNK_SIZE) ? SWARM_CHUNK_SIZE : rem;
            memcpy(p.data, &data[off], p.data_len);
            esp_err_t err;
            do { err = esp_now_send(BROADCAST_ADDR, (uint8_t *)&p, SWARM_HEADER_SIZE + p.data_len);
                 if (err == ESP_ERR_ESPNOW_NO_MEM) esp_rom_delay_us(500);
            } while (err == ESP_ERR_ESPNOW_NO_MEM);
            esp_rom_delay_us(200);   /* pacing reduces broadcast loss */
        }
    }
    esp_rom_delay_us(2000);
    SwarmPacket c = {0}; c.cmd = CMD_COMPUTE; c.layer = k;
    esp_now_send(BROADCAST_ADDR, (uint8_t *)&c, SWARM_HEADER_SIZE);
}

/* Concatenate per-worker channel shards: bufs[w] is [N x shard] -> out [N x Cout]. */
static void assemble_channels(int N, int shard, int Cout, int8_t *out) {
    for (int p = 0; p < N; p++)
        for (int w = 0; w < NUM_WORKERS; w++)
            memcpy(&out[p * Cout + w * shard], &worker_recv_buf[w][p * shard], shard);
}

/* Run one distributed pointwise round k: broadcast input, gather, assemble. */
static bool run_pw(uint8_t k, const int8_t *input, int N, int Cin, int Cout,
                   int8_t in_zp, int8_t *out,
                   int64_t *bcast_us, int64_t *gather_us, int64_t *comp_us) {
    int shard = Cout / NUM_WORKERS;
    cur_round = k;
    for (int i = 0; i < NUM_WORKERS; i++) {
        worker_done[i] = false; worker_compute_us[i] = 0;
        worker_result_unique[i] = 0; memset(worker_result_bitmap[i], 0, 64);
        worker_expected[i] = 0; worker_sparse[i] = false;
    }
    workers_complete = 0;
    /* a worker may send dense OR sparse (bitmap+nonzeros); the encoded form can be
     * up to dense + ceil(dense/8). Size the RX bound for that worst case. */
    uint32_t result_bytes = (uint32_t)N * shard;
    current_result_size = result_bytes + (result_bytes + 7) / 8;
    xSemaphoreTake(all_done_sem, 0);

    uint32_t in_len = (uint32_t)N * Cin;
    /* Sparsify the broadcast input (post-ReLU activation). Send sparse only if it
     * shrinks; workers decode with (orig_len, in_zp). */
    uint32_t enc_len = sparse_encode(input, in_len, bcast_enc, in_zp);
    uint8_t  sp = (enc_len < in_len) ? 1 : 0;
    const int8_t *send_data = sp ? bcast_enc : input;
    uint32_t send_len = sp ? enc_len : in_len;
    int64_t tb = esp_timer_get_time();
    send_layer_start(send_len, k, sp, in_len, in_zp);
    send_chunks(send_data, send_len, k, 2);      /* initial: redundancy 2 */
    *bcast_us = esp_timer_get_time() - tb;

    int64_t tg = esp_timer_get_time();
    int slices = 0;
    /* Long slice so re-broadcast fires only on a genuine stall, never while a
     * worker is still computing/sending (that would overwrite its input_buf). */
    while (xSemaphoreTake(all_done_sem, pdMS_TO_TICKS(5000)) != pdTRUE) {
        if (++slices > 8) {
            ESP_LOGE(TAG, "pw%d gather TIMEOUT (%d/%d) results:", k, workers_complete, NUM_WORKERS);
            for (int i = 0; i < NUM_WORKERS; i++)
                ESP_LOGE(TAG, "  W%d %d/%d", i, worker_result_unique[i], worker_expected[i]);
            return false;
        }
        /* full re-broadcast (LAYER_START re-arms all workers to recompute+resend);
         * coordinator result bitmaps accumulate, completed workers stay done */
        send_layer_start(send_len, k, sp, in_len, in_zp);
        send_chunks(send_data, send_len, k, 1);
    }
    *gather_us = esp_timer_get_time() - tg;
    uint32_t cmax = 0;
    for (int i = 0; i < NUM_WORKERS; i++) if (worker_compute_us[i] > cmax) cmax = worker_compute_us[i];
    *comp_us = cmax;

    /* Decode any sparse shards back to dense in place before assembly. */
    for (int w = 0; w < NUM_WORKERS; w++) {
        if (worker_sparse[w]) {
            sparse_decode(worker_recv_buf[w], decode_tmp, worker_sparse_orig[w], worker_sparse_zp[w]);
            memcpy(worker_recv_buf[w], decode_tmp, worker_sparse_orig[w]);
        }
    }
    assemble_channels(N, shard, Cout, out);
    /* No inter-round wait_ready(): workers are stateless per round (they reset on
     * the next LAYER_START and always listen), and the round-tagged gather
     * already rejects stragglers. The handshake cost ~5s/round (it re-armed
     * AFTER the worker's post-result READY, so it waited for the 5s heartbeat). */
    return true;
}

static int64_t checksum(const int8_t *d, int n) {
    int64_t s = 0; for (int i = 0; i < n; i++) s += d[i]; return s;
}

void app_main(void) {
    ESP_LOGI(TAG, "MobileNet coordinator (N=%d), %d layers", NUM_WORKERS, MB_NUM_LAYERS);
    all_done_sem = xSemaphoreCreateBinary();
    all_ready_sem = xSemaphoreCreateBinary();
    wifi_init();

    buf_a = heap_caps_malloc(ACT_BUF, MALLOC_CAP_SPIRAM);
    buf_b = heap_caps_malloc(ACT_BUF, MALLOC_CAP_SPIRAM);
    buf_pad = heap_caps_malloc(ACT_BUF, MALLOC_CAP_SPIRAM);
    decode_tmp = heap_caps_malloc(RECV_BUF, MALLOC_CAP_SPIRAM);
    bcast_enc = heap_caps_malloc(ACT_BUF, MALLOC_CAP_SPIRAM);
    bool ok = buf_a && buf_b && buf_pad && decode_tmp && bcast_enc;
    for (int i = 0; i < NUM_WORKERS; i++) {
        worker_recv_buf[i] = heap_caps_malloc(RECV_BUF, MALLOC_CAP_SPIRAM);
        ok = ok && worker_recv_buf[i];
    }
    if (!ok) { ESP_LOGE(TAG, "PSRAM alloc failed"); return; }

    ESP_LOGI(TAG, "Waiting for %d workers...", NUM_WORKERS);
    if (!wait_ready()) { ESP_LOGE(TAG, "workers not ready"); return; }
    ESP_LOGI(TAG, "All workers ready. Running inference.");

    /* input image */
    Tensor3D cur = { buf_a, MB_IN_H, MB_IN_W, MB_IN_C };
    memcpy(buf_a, mb_test_input, (size_t)MB_IN_H * MB_IN_W * MB_IN_C);
    int8_t *dst = buf_b;

    int mism = 0, pw_k = 0;
    int64_t t0 = esp_timer_get_time(), total_bcast = 0, total_gather = 0, local_us = 0;
    g_wait_us = 0;

    for (int li = 0; li < MB_NUM_LAYERS - 1; li++) {
        const MbLayer *L = &mb_layers[li];
        Tensor3D out = { dst, 0, 0, 0 };
        if (L->type == MB_PW) {
            int N = cur.h * cur.w, Cin = cur.c, Cout = L->Cout;
            int64_t bc = 0, ga = 0, cm = 0;
            if (!run_pw((uint8_t)pw_k, cur.data, N, Cin, Cout, L->in_zp, dst, &bc, &ga, &cm)) {
                ESP_LOGE(TAG, "abort at pw%d", pw_k); return;
            }
            total_bcast += bc; total_gather += ga;
            out.h = cur.h; out.w = cur.w; out.c = Cout;
            ESP_LOGI(TAG, "pw%d (L%02d) %dx%dx%d bcast=%lldms gather=%lldms comp=%lldms",
                     pw_k, li, out.h, out.w, out.c, bc/1000, ga/1000, cm/1000);
            pw_k++;
        } else {
            int64_t tl = esp_timer_get_time();
            mb_run_conv_layer(L, &cur, &out, buf_pad);  /* conv0 / depthwise */
            local_us += esp_timer_get_time() - tl;
        }
        int64_t cs = checksum(out.data, out.h * out.w * out.c);
        if (cs != mb_checksums[li]) { mism++; ESP_LOGW(TAG, "L%02d MISMATCH cs=%lld ref=%lld",
                                                       li, (long long)cs, (long long)mb_checksums[li]); }
        cur = out;
        dst = (cur.data == buf_a) ? buf_b : buf_a;
    }

    /* GAP + dense (local) */
    int ngap = cur.c;
    int8_t *gap = buf_pad;
    Tensor1D gout = { gap, ngap };
    global_avgpool_int8(&cur, &gout, MB_GAP_IN_ZP, MB_GAP_IN_SCALE, MB_GAP_SCALE, MB_GAP_ZP);
    if (checksum(gap, ngap) != mb_checksums[MB_NUM_LAYERS - 1]) mism++;
    const MbLayer *D = &mb_layers[MB_NUM_LAYERS - 1];
    int8_t logits[16];
    Tensor1D din = { gap, ngap }, dout = { logits, D->Cout };
    FixedPointMultiplier dm = { D->mult, D->shift };
    dense_int8(&din, D->w, D->b, &dout, dm, D->in_zp, D->w_zp, D->out_zp);
    if (checksum(logits, D->Cout) != mb_checksums[MB_NUM_LAYERS]) mism++;

    int64_t total_us = esp_timer_get_time() - t0;
    int pred = argmax_int8(logits, D->Cout);
    ESP_LOGI(TAG, "pred=%d expected=%d label=%d  total=%lldms (bcast=%lldms gather=%lldms)  mism=%d",
             pred, MB_EXPECTED_PRED, MB_TEST_LABEL, total_us/1000, total_bcast/1000,
             total_gather/1000, mism);
    int64_t other_us = total_us - total_bcast - total_gather - local_us - g_wait_us;
    ESP_LOGI(TAG, "PROFILE: local_conv/dw=%lldms wait_ready=%lldms bcast=%lldms gather=%lldms other=%lldms",
             local_us/1000, g_wait_us/1000, total_bcast/1000, total_gather/1000, other_us/1000);
    ESP_LOGI(TAG, "RESULT: %s",
             (mism == 0 && pred == MB_EXPECTED_PRED) ? "DISTRIBUTED BIT-EXACT PASS" : "FAIL");
    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}
