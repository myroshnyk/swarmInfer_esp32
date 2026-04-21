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

static const char *TAG = "COORD";

#define NUM_WORKERS 4

static uint8_t worker_macs[NUM_WORKERS][6] = {
    {0xB8, 0xF8, 0x62, 0xE2, 0xD1, 0x98},  // Worker 1
    {0xB8, 0xF8, 0x62, 0xE2, 0xCD, 0xE4},  // Worker 2
    {0xB8, 0xF8, 0x62, 0xE2, 0xDA, 0x28},  // Worker 3
    {0xB8, 0xF8, 0x62, 0xE2, 0xD0, 0xDC},  // Worker 4
};

static volatile int replies_received;
static int64_t reply_times[NUM_WORKERS];
static uint8_t reply_from[NUM_WORKERS][6];
static SemaphoreHandle_t reply_sem;

static void on_sent(const esp_now_send_info_t *info, esp_now_send_status_t status)
{
    (void)info; (void)status;
}

static void on_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len)
{
    int64_t now = esp_timer_get_time();
    if (len >= 1 && (data[0] == 0xBB || data[0] == 0xDD)) {
        int idx = replies_received;
        if (idx < NUM_WORKERS) {
            reply_times[idx] = now;
            memcpy(reply_from[idx], info->src_addr, 6);
            replies_received++;
        }
        if (replies_received >= NUM_WORKERS) {
            xSemaphoreGiveFromISR(reply_sem, NULL);
        }
    }
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE));
}

// Test 1: Broadcast ping to all workers, measure time until all reply
static void test_broadcast_ping(int rounds)
{
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== Test 1: Broadcast PING → %d workers ===", NUM_WORKERS);

    // Use broadcast MAC
    uint8_t bcast[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    // Add broadcast peer if not already
    esp_now_peer_info_t bpeer = {0};
    memcpy(bpeer.peer_addr, bcast, 6);
    bpeer.channel = 1;
    bpeer.ifidx = WIFI_IF_STA;
    bpeer.encrypt = false;
    esp_now_add_peer(&bpeer);  // ignore error if already added

    int64_t *latencies = malloc(rounds * sizeof(int64_t));
    int success = 0;
    int timeouts = 0;

    for (int r = 0; r < rounds; r++) {
        replies_received = 0;

        uint8_t ping[10];
        ping[0] = 0xAA;  // ping command
        memset(&ping[1], r & 0xFF, 9);

        int64_t t_send = esp_timer_get_time();
        esp_now_send(bcast, ping, 10);

        // Wait for all 4 workers to reply
        if (xSemaphoreTake(reply_sem, pdMS_TO_TICKS(500)) == pdTRUE) {
            int64_t t_last = 0;
            for (int i = 0; i < NUM_WORKERS; i++) {
                if (reply_times[i] > t_last) t_last = reply_times[i];
            }
            latencies[success] = t_last - t_send;
            success++;
        } else {
            timeouts++;
            ESP_LOGW(TAG, "Round %d: timeout (got %d/%d replies)",
                     r, replies_received, NUM_WORKERS);
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }

    if (success > 0) {
        // Sort for percentiles
        for (int i = 0; i < success - 1; i++)
            for (int j = i + 1; j < success; j++)
                if (latencies[i] > latencies[j]) {
                    int64_t tmp = latencies[i];
                    latencies[i] = latencies[j];
                    latencies[j] = tmp;
                }

        int64_t total = 0;
        for (int i = 0; i < success; i++) total += latencies[i];

        ESP_LOGI(TAG, "--- Broadcast ping results (%d/%d ok) ---", success, success + timeouts);
        ESP_LOGI(TAG, "  All-reply min:    %lld us", latencies[0]);
        ESP_LOGI(TAG, "  All-reply avg:    %lld us", total / success);
        ESP_LOGI(TAG, "  All-reply median: %lld us", latencies[success / 2]);
        ESP_LOGI(TAG, "  All-reply p95:    %lld us", latencies[(int)(success * 0.95)]);
        ESP_LOGI(TAG, "  All-reply max:    %lld us", latencies[success - 1]);
        ESP_LOGI(TAG, "  Timeouts:         %d", timeouts);
    }

    free(latencies);
}

// Test 2: Unicast to each worker sequentially, compare with broadcast
static void test_unicast_sequential(int rounds)
{
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== Test 2: Unicast PING → each worker sequentially ===");

    for (int w = 0; w < NUM_WORKERS; w++) {
        int64_t total = 0;
        int success = 0;

        for (int r = 0; r < rounds; r++) {
            replies_received = 0;

            uint8_t ping[10];
            ping[0] = 0xAA;
            memset(&ping[1], r & 0xFF, 9);

            int64_t t_send = esp_timer_get_time();
            esp_now_send(worker_macs[w], ping, 10);

            if (xSemaphoreTake(reply_sem, pdMS_TO_TICKS(200)) == pdTRUE ||
                replies_received >= 1) {
                // Wait a bit more for the semaphore if replies_received is already 1
                if (replies_received >= 1) {
                    total += reply_times[0] - t_send;
                    success++;
                }
            }

            vTaskDelay(pdMS_TO_TICKS(5));
        }

        if (success > 0) {
            ESP_LOGI(TAG, "  Worker %d (%02X:%02X): avg RTT = %lld us (%d/%d ok)",
                     w + 1, worker_macs[w][4], worker_macs[w][5],
                     total / success, success, rounds);
        }
    }
}

// Test 3: Simulate tensor broadcast (chunked large payload)
static void test_tensor_broadcast(void)
{
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== Test 3: Tensor broadcast (3072 bytes as 32x32x3 image) ===");

    uint8_t bcast[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    // Simulate sending 3072 bytes in 240-byte chunks
    int total_bytes = 3072;
    int chunk_size = 240;
    int num_chunks = (total_bytes + chunk_size - 1) / chunk_size;  // 13 chunks

    uint8_t chunk[240];
    chunk[0] = 0xCC;  // tensor data marker

    ESP_LOGI(TAG, "  Total: %d bytes, Chunks: %d x %d bytes", total_bytes, num_chunks, chunk_size);

    int rounds = 20;
    int64_t *times = malloc(rounds * sizeof(int64_t));
    int success = 0;

    for (int r = 0; r < rounds; r++) {
        int64_t t_start = esp_timer_get_time();

        // Send all chunks
        for (int c = 0; c < num_chunks; c++) {
            int this_size = (c == num_chunks - 1) ? (total_bytes % chunk_size) : chunk_size;
            if (this_size == 0) this_size = chunk_size;

            chunk[1] = (uint8_t)c;  // chunk index
            memset(&chunk[2], 0xAB, this_size - 2);

            esp_now_send(bcast, chunk, this_size);
            vTaskDelay(pdMS_TO_TICKS(1));  // small delay between chunks
        }

        // Wait for all workers to ACK the last chunk
        replies_received = 0;
        if (xSemaphoreTake(reply_sem, pdMS_TO_TICKS(1000)) == pdTRUE) {
            int64_t t_end = esp_timer_get_time();
            times[success] = t_end - t_start;
            success++;
        } else {
            ESP_LOGW(TAG, "  Round %d: timeout (got %d/%d ACKs)",
                     r, replies_received, NUM_WORKERS);
        }

        vTaskDelay(pdMS_TO_TICKS(50));
    }

    if (success > 0) {
        int64_t total = 0, min_t = INT64_MAX, max_t = 0;
        for (int i = 0; i < success; i++) {
            total += times[i];
            if (times[i] < min_t) min_t = times[i];
            if (times[i] > max_t) max_t = times[i];
        }
        ESP_LOGI(TAG, "--- Tensor broadcast results (%d/%d ok) ---", success, rounds);
        ESP_LOGI(TAG, "  Transfer min:  %lld us (%.1f ms)", min_t, min_t / 1000.0);
        ESP_LOGI(TAG, "  Transfer avg:  %lld us (%.1f ms)", total / success, (total / success) / 1000.0);
        ESP_LOGI(TAG, "  Transfer max:  %lld us (%.1f ms)", max_t, max_t / 1000.0);
        ESP_LOGI(TAG, "  Effective throughput: %.1f KB/s",
                 (float)total_bytes / ((float)(total / success) / 1000000.0f) / 1024.0f);
    }

    free(times);
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    reply_sem = xSemaphoreCreateBinary();
    wifi_init();

    uint8_t my_mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, my_mac);
    ESP_LOGI(TAG, "Coordinator MAC: %02X:%02X:%02X:%02X:%02X:%02X",
             my_mac[0], my_mac[1], my_mac[2], my_mac[3], my_mac[4], my_mac[5]);

    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_send_cb(on_sent));
    ESP_ERROR_CHECK(esp_now_register_recv_cb(on_recv));

    // Add all workers as peers
    for (int i = 0; i < NUM_WORKERS; i++) {
        esp_now_peer_info_t peer = {0};
        memcpy(peer.peer_addr, worker_macs[i], 6);
        peer.channel = 1;
        peer.ifidx = WIFI_IF_STA;
        peer.encrypt = false;
        ESP_ERROR_CHECK(esp_now_add_peer(&peer));
        ESP_LOGI(TAG, "Added worker %d: %02X:%02X:%02X:%02X:%02X:%02X",
                 i + 1, worker_macs[i][0], worker_macs[i][1], worker_macs[i][2],
                 worker_macs[i][3], worker_macs[i][4], worker_macs[i][5]);
    }

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== Coordinator Ready ===");
    ESP_LOGI(TAG, "Waiting 5s for all workers to boot...");
    vTaskDelay(pdMS_TO_TICKS(5000));

    // Run tests
    test_broadcast_ping(200);
    test_unicast_sequential(100);
    test_tensor_broadcast();

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "======== ALL MULTI-PEER TESTS COMPLETE ========");

    while (1) { vTaskDelay(pdMS_TO_TICKS(10000)); }
}
