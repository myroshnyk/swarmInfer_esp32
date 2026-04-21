/**
 * Gather Benchmark — Coordinator
 *
 * Measures pure communication overhead without compute.
 * Protocol:
 *   1. Wait for all workers READY
 *   2. Broadcast CMD_COMPUTE (= "go")
 *   3. Record timestamp of every RESULT_CHUNK + RESULT_DONE
 *   4. Print per-worker timeline
 *
 * Tests multiple payload sizes and worker counts.
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
#include "swarm_protocol.h"

static const char *TAG = "GBENCH";

#ifndef NUM_WORKERS
#define NUM_WORKERS 4
#endif

static const uint8_t ALL_WORKER_MACS[4][6] = {
    {0xB8,0xF8,0x62,0xE2,0xD1,0x98},
    {0xB8,0xF8,0x62,0xE2,0xCD,0xE4},
    {0xB8,0xF8,0x62,0xE2,0xDA,0x28},
    {0xB8,0xF8,0x62,0xE2,0xD0,0xDC},
};
static const uint8_t BROADCAST_ADDR[] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};

/* ── Gather state ── */
#define MAX_CHUNKS_PER_WORKER 64
static volatile int     worker_chunks_received[NUM_WORKERS];
static volatile int64_t worker_first_chunk_us[NUM_WORKERS];
static volatile int64_t worker_last_chunk_us[NUM_WORKERS];
static volatile int64_t worker_done_us[NUM_WORKERS];
static volatile bool    worker_done[NUM_WORKERS];
static volatile int     workers_complete;
static SemaphoreHandle_t all_workers_done_sem;

/* ── Ready state ── */
static volatile bool worker_ready[NUM_WORKERS];
static volatile int  workers_ready_count;
static SemaphoreHandle_t all_workers_ready_sem;

/* ── Test config (sent in CMD_COMPUTE data) ── */
static volatile uint16_t current_payload_size;

/* ── Timing ── */
static int64_t t_go;  /* timestamp when GO was sent */

static int mac_to_worker_id(const uint8_t *mac) {
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
    int wid = mac_to_worker_id(info->src_addr);
    if (wid < 0) return;

    int64_t now = esp_timer_get_time();

    switch (pkt->cmd) {
    case CMD_RESULT_CHUNK:
        if (worker_chunks_received[wid] == 0)
            worker_first_chunk_us[wid] = now;
        worker_last_chunk_us[wid] = now;
        worker_chunks_received[wid]++;
        break;
    case CMD_RESULT_DONE:
        worker_done_us[wid] = now;
        worker_done[wid] = true;
        workers_complete++;
        if (workers_complete >= NUM_WORKERS)
            xSemaphoreGiveFromISR(all_workers_done_sem, NULL);
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
        memcpy(p.peer_addr, ALL_WORKER_MACS[i], 6);
        p.channel = 1; p.encrypt = false;
        esp_now_add_peer(&p);
    }
}

static void reset_ready(void) {
    for (int i = 0; i < NUM_WORKERS; i++)
        worker_ready[i] = false;
    workers_ready_count = 0;
    xSemaphoreTake(all_workers_ready_sem, 0);
}

static bool wait_for_ready(int timeout_ms) {
    if (xSemaphoreTake(all_workers_ready_sem, pdMS_TO_TICKS(timeout_ms)) != pdTRUE) {
        ESP_LOGE(TAG, "READY timeout! Got %d/%d", workers_ready_count, NUM_WORKERS);
        return false;
    }
    return true;
}

static void reset_gather(void) {
    for (int i = 0; i < NUM_WORKERS; i++) {
        worker_chunks_received[i] = 0;
        worker_first_chunk_us[i] = 0;
        worker_last_chunk_us[i] = 0;
        worker_done_us[i] = 0;
        worker_done[i] = false;
    }
    workers_complete = 0;
    xSemaphoreTake(all_workers_done_sem, 0);
}

static void run_test(uint16_t payload_size, int rounds) {
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== Test: %d workers × %d bytes, %d rounds ===",
             NUM_WORKERS, payload_size, rounds);

    int64_t total_gather_us = 0;
    int64_t min_gather = INT64_MAX;
    int64_t max_gather = 0;

    for (int r = 0; r < rounds; r++) {
        /* Wait for all workers to be ready */
        reset_ready();

        /* Send GO with payload size in data */
        reset_gather();
        SwarmPacket go = {0};
        go.cmd = CMD_COMPUTE;
        go.layer = 0;  /* benchmark mode */
        go.data[0] = (uint8_t)(payload_size & 0xFF);
        go.data[1] = (uint8_t)((payload_size >> 8) & 0xFF);
        go.data_len = 2;

        /* Wait for ready first */
        if (!wait_for_ready(10000)) {
            ESP_LOGE(TAG, "Round %d: workers not ready, skipping", r);
            continue;
        }

        t_go = esp_timer_get_time();
        esp_now_send(BROADCAST_ADDR, (uint8_t *)&go, SWARM_HEADER_SIZE + 2);

        /* Wait for all DONE */
        if (xSemaphoreTake(all_workers_done_sem, pdMS_TO_TICKS(30000)) != pdTRUE) {
            ESP_LOGE(TAG, "Round %d: TIMEOUT! Complete: %d/%d", r, workers_complete, NUM_WORKERS);
            for (int i = 0; i < NUM_WORKERS; i++)
                ESP_LOGE(TAG, "  W%d: chunks=%d done=%d", i,
                         worker_chunks_received[i], worker_done[i]);
            continue;
        }

        /* Find overall gather time: GO → last DONE */
        int64_t last_done = 0;
        for (int i = 0; i < NUM_WORKERS; i++)
            if (worker_done_us[i] > last_done)
                last_done = worker_done_us[i];

        int64_t gather_us = last_done - t_go;
        total_gather_us += gather_us;
        if (gather_us < min_gather) min_gather = gather_us;
        if (gather_us > max_gather) max_gather = gather_us;

        /* Print detailed timeline for first 3 rounds */
        if (r < 3) {
            ESP_LOGI(TAG, "Round %d: total=%lld us", r, gather_us);
            for (int i = 0; i < NUM_WORKERS; i++) {
                int64_t first = worker_first_chunk_us[i] - t_go;
                int64_t last  = worker_last_chunk_us[i] - t_go;
                int64_t done  = worker_done_us[i] - t_go;
                ESP_LOGI(TAG, "  W%d: first=%lld us, last=%lld us, done=%lld us, chunks=%d",
                         i, first, last, done, worker_chunks_received[i]);
            }
        }
    }

    int64_t avg = total_gather_us / rounds;
    ESP_LOGI(TAG, "--- Summary: %d workers × %d B ---", NUM_WORKERS, payload_size);
    ESP_LOGI(TAG, "  Avg: %lld us (%lld ms)", avg, avg / 1000);
    ESP_LOGI(TAG, "  Min: %lld us, Max: %lld us", min_gather, max_gather);
    int chunks = swarm_num_chunks(payload_size);
    ESP_LOGI(TAG, "  Chunks/worker: %d, Total chunks: %d",
             chunks, chunks * NUM_WORKERS);
    float throughput = (float)(payload_size * NUM_WORKERS) / ((float)avg / 1000000.0f) / 1024.0f;
    ESP_LOGI(TAG, "  Aggregate throughput: %.1f KB/s", throughput);
}

void app_main(void) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Gather Benchmark — Coordinator");
    ESP_LOGI(TAG, "  Workers: %d", NUM_WORKERS);
    ESP_LOGI(TAG, "========================================");

    all_workers_done_sem  = xSemaphoreCreateBinary();
    all_workers_ready_sem = xSemaphoreCreateBinary();
    wifi_init();

    ESP_LOGI(TAG, "Waiting for %d workers...", NUM_WORKERS);
    reset_ready();
    if (!wait_for_ready(30000)) {
        ESP_LOGE(TAG, "Not all workers ready, aborting.");
        return;
    }
    ESP_LOGI(TAG, "All %d workers READY!", NUM_WORKERS);

    /* Test payload sizes matching real inference:
     * L3 GAP: 64 B
     * L2 pool: 2048 B
     * L1 pool: 4096 B
     * Also test smaller/larger to see scaling */
    uint16_t sizes[] = {64, 232, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int rounds = 20;

    for (int s = 0; s < num_sizes; s++) {
        run_test(sizes[s], rounds);
    }

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== ALL TESTS COMPLETE ===");
    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}
