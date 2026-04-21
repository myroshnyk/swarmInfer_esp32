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

static const char *TAG = "BENCH";

#define ROLE_PING 0
#define ROLE_PONG 1
#define MY_ROLE ROLE_PING

static uint8_t peer_mac[] = {0xB8, 0xF8, 0x62, 0xE2, 0xD1, 0x98};

#define NUM_TESTS        500
#define WARMUP_PACKETS   10
#define TIMEOUT_MS       200

static const int payload_sizes[] = {10, 50, 100, 150, 200, 240};
#define NUM_PAYLOAD_SIZES (sizeof(payload_sizes) / sizeof(payload_sizes[0]))

static SemaphoreHandle_t recv_sem;
static int64_t recv_timestamp;
static uint8_t recv_buf[250];
static int recv_len;

static void on_sent(const esp_now_send_info_t *info, esp_now_send_status_t status)
{
    (void)info;
    (void)status;
}

static void on_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len)
{
    recv_timestamp = esp_timer_get_time();
    if (len > 0 && len <= 250) {
        memcpy(recv_buf, data, len);
        recv_len = len;
    }
    xSemaphoreGiveFromISR(recv_sem, NULL);
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

static int cmp_int64(const void *a, const void *b)
{
    int64_t va = *(const int64_t *)a;
    int64_t vb = *(const int64_t *)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

static void run_pong(void)
{
    ESP_LOGI(TAG, "=== PONG MODE (responder) ===");
    ESP_LOGI(TAG, "Echoing packets back...");
    uint32_t cnt = 0;
    while (1) {
        if (xSemaphoreTake(recv_sem, pdMS_TO_TICKS(5000)) == pdTRUE) {
            esp_now_send(peer_mac, recv_buf, recv_len);
            cnt++;
            if (cnt % 200 == 0) {
                ESP_LOGI(TAG, "Echoed %lu", (unsigned long)cnt);
            }
        }
    }
}

static void run_ping(void)
{
    ESP_LOGI(TAG, "=== PING MODE (initiator) ===");
    ESP_LOGI(TAG, "Tests: %d sizes x %d rounds", (int)NUM_PAYLOAD_SIZES, NUM_TESTS);

    int64_t *latencies = (int64_t *)malloc(NUM_TESTS * sizeof(int64_t));
    if (!latencies) {
        ESP_LOGE(TAG, "Failed to allocate latencies array!");
        return;
    }

    ESP_LOGI(TAG, "Waiting 3s for pong board...");
    vTaskDelay(pdMS_TO_TICKS(3000));

    for (int s = 0; s < (int)NUM_PAYLOAD_SIZES; s++) {
        int size = payload_sizes[s];
        int success = 0;
        int timeouts = 0;

        ESP_LOGI(TAG, "");
        ESP_LOGI(TAG, "=== Payload: %d bytes ===", size);

        uint8_t payload[250];
        memset(payload, 0xAB, size);

        for (int i = 0; i < NUM_TESTS + WARMUP_PACKETS; i++) {
            if (size >= 4) {
                uint32_t seq = (uint32_t)i;
                memcpy(payload, &seq, 4);
            }

            int64_t t_send = esp_timer_get_time();

            esp_err_t err = esp_now_send(peer_mac, payload, size);
            if (err != ESP_OK) {
                if (i >= WARMUP_PACKETS) timeouts++;
                vTaskDelay(pdMS_TO_TICKS(5));
                continue;
            }

            if (xSemaphoreTake(recv_sem, pdMS_TO_TICKS(TIMEOUT_MS)) == pdTRUE) {
                int64_t rtt = recv_timestamp - t_send;
                if (i >= WARMUP_PACKETS && success < NUM_TESTS) {
                    latencies[success] = rtt;
                    success++;
                }
            } else {
                if (i >= WARMUP_PACKETS) timeouts++;
            }

            vTaskDelay(pdMS_TO_TICKS(3));
        }

        if (success == 0) {
            ESP_LOGW(TAG, "  No successful packets for %d bytes!", size);
            continue;
        }

        qsort(latencies, success, sizeof(int64_t), cmp_int64);

        int64_t total = 0;
        for (int i = 0; i < success; i++) total += latencies[i];

        int64_t avg = total / success;
        int64_t med = latencies[success / 2];
        int64_t p95 = latencies[(int)(success * 0.95)];
        int64_t p99 = latencies[(int)(success * 0.99)];

        ESP_LOGI(TAG, "--- %d bytes: %d/%d ok ---", size, success, success + timeouts);
        ESP_LOGI(TAG, "  RTT min:    %lld us", latencies[0]);
        ESP_LOGI(TAG, "  RTT avg:    %lld us", avg);
        ESP_LOGI(TAG, "  RTT median: %lld us", med);
        ESP_LOGI(TAG, "  RTT p95:    %lld us", p95);
        ESP_LOGI(TAG, "  RTT p99:    %lld us", p99);
        ESP_LOGI(TAG, "  RTT max:    %lld us", latencies[success - 1]);
        ESP_LOGI(TAG, "  Timeouts:   %d", timeouts);
        ESP_LOGI(TAG, "  One-way:    ~%lld us", avg / 2);
        ESP_LOGI(TAG, "  Throughput: %.1f KB/s",
                 (float)size / ((float)avg / 2.0f) * 1000.0f);
    }

    free(latencies);
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "======== ALL TESTS COMPLETE ========");

    while (1) { vTaskDelay(pdMS_TO_TICKS(10000)); }
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    recv_sem = xSemaphoreCreateBinary();
    wifi_init();

    uint8_t my_mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, my_mac);
    ESP_LOGI(TAG, "MAC: %02X:%02X:%02X:%02X:%02X:%02X  Role: %s",
             my_mac[0], my_mac[1], my_mac[2], my_mac[3], my_mac[4], my_mac[5],
             MY_ROLE == ROLE_PING ? "PING" : "PONG");

    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_send_cb(on_sent));
    ESP_ERROR_CHECK(esp_now_register_recv_cb(on_recv));

    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, peer_mac, 6);
    peer.channel = 1;
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    if (MY_ROLE == ROLE_PING) {
        run_ping();
    } else {
        run_pong();
    }
}
