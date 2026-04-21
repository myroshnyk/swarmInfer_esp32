#include <stdio.h>
#include <string.h>
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

static const char *TAG = "WORKER";

// Coordinator MAC
static uint8_t coord_mac[] = {0xB8, 0xF8, 0x62, 0xE2, 0xD0, 0x8C};

static SemaphoreHandle_t recv_sem;
static uint8_t recv_buf[250];
static int recv_len;
static uint32_t total_received = 0;
static uint32_t total_replied = 0;

static void on_sent(const esp_now_send_info_t *info, esp_now_send_status_t status)
{
    (void)info; (void)status;
}

static void on_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len)
{
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
    ESP_LOGI(TAG, "Worker MAC: %02X:%02X:%02X:%02X:%02X:%02X",
             my_mac[0], my_mac[1], my_mac[2], my_mac[3], my_mac[4], my_mac[5]);

    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_send_cb(on_sent));
    ESP_ERROR_CHECK(esp_now_register_recv_cb(on_recv));

    // Add coordinator as peer
    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, coord_mac, 6);
    peer.channel = 1;
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    ESP_LOGI(TAG, "=== Worker Ready — waiting for coordinator ===");

    while (1) {
        if (xSemaphoreTake(recv_sem, pdMS_TO_TICKS(5000)) == pdTRUE) {
            total_received++;

            // Check if it's a command byte in first position
            uint8_t cmd = recv_buf[0];

            if (cmd == 0xAA) {
                // PING: echo back with our MAC in first 6 bytes + timestamp
                uint8_t reply[16];
                reply[0] = 0xBB; // reply marker
                memcpy(&reply[1], my_mac, 6);
                int64_t now = esp_timer_get_time();
                memcpy(&reply[7], &now, 8);
                reply[15] = (uint8_t)recv_len;

                esp_now_send(coord_mac, reply, 16);
                total_replied++;
            } else if (cmd == 0xCC) {
                // TENSOR DATA: simulate receiving a tensor chunk
                // In real SwarmInfer, this would trigger computation
                // For now, just ACK back
                uint8_t ack[8];
                ack[0] = 0xDD; // ack marker
                memcpy(&ack[1], my_mac, 6);
                ack[7] = (uint8_t)recv_len;
                esp_now_send(coord_mac, ack, 8);
                total_replied++;
            }

            if (total_received % 100 == 0) {
                ESP_LOGI(TAG, "Received: %lu  Replied: %lu",
                         (unsigned long)total_received, (unsigned long)total_replied);
            }
        }
    }
}
