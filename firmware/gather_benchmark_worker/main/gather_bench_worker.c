/**
 * Gather Benchmark — Worker
 *
 * Waits for CMD_COMPUTE from coordinator, then sends back
 * the requested number of bytes as RESULT_CHUNK + RESULT_DONE.
 * No actual computation — measures pure send overhead.
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

static const char *TAG = "GBWORK";

#ifndef WORKER_ID
#define WORKER_ID 3
#endif

static const uint8_t MAC_COORD[] = {0xB8,0xF8,0x62,0xE2,0xD0,0x8C};

static SemaphoreHandle_t go_sem;
static volatile uint16_t requested_size;

/* Send buffer (filled with dummy data) */
static int8_t *send_buf = NULL;
#define MAX_SEND_SIZE 8192

static void on_sent(const esp_now_send_info_t *info, esp_now_send_status_t status) {
    (void)info; (void)status;
}

static void on_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (len < SWARM_HEADER_SIZE) return;
    const SwarmPacket *pkt = (const SwarmPacket *)data;

    if (pkt->cmd == CMD_COMPUTE && pkt->layer == 0 && pkt->data_len >= 2) {
        requested_size = pkt->data[0] | ((uint16_t)pkt->data[1] << 8);
        xSemaphoreGiveFromISR(go_sem, NULL);
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

    /* Add coordinator as peer */
    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, MAC_COORD, 6);
    peer.channel = 1; peer.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    /* Add broadcast for receiving */
    esp_now_peer_info_t bcast = {0};
    memcpy(bcast.peer_addr, (uint8_t[]){0xFF,0xFF,0xFF,0xFF,0xFF,0xFF}, 6);
    bcast.channel = 1; bcast.encrypt = false;
    esp_now_add_peer(&bcast);
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

static void send_result(uint16_t size) {
    uint16_t total = swarm_num_chunks(size);
    SwarmPacket pkt;
    pkt.cmd = CMD_RESULT_CHUNK;
    pkt.layer = 0;
    pkt.total_chunks = total;
    for (uint16_t i = 0; i < total; i++) {
        pkt.chunk_id = i;
        uint32_t offset = (uint32_t)i * SWARM_CHUNK_SIZE;
        uint16_t remain = size - offset;
        pkt.data_len = (remain > SWARM_CHUNK_SIZE) ? SWARM_CHUNK_SIZE : remain;
        memcpy(pkt.data, &send_buf[offset], pkt.data_len);
        esp_err_t err;
        do {
            err = esp_now_send(MAC_COORD, (uint8_t *)&pkt, SWARM_HEADER_SIZE + pkt.data_len);
            if (err == ESP_ERR_ESPNOW_NO_MEM) esp_rom_delay_us(500);
        } while (err == ESP_ERR_ESPNOW_NO_MEM);
        esp_rom_delay_us(500);
    }
    esp_rom_delay_us(2000);
    SwarmPacket done = {0};
    done.cmd = CMD_RESULT_DONE;
    done.layer = 0;
    done.chunk_id = WORKER_ID;
    done.data_len = 0;
    esp_err_t err;
    do {
        err = esp_now_send(MAC_COORD, (uint8_t *)&done, SWARM_HEADER_SIZE);
        if (err == ESP_ERR_ESPNOW_NO_MEM) esp_rom_delay_us(500);
    } while (err == ESP_ERR_ESPNOW_NO_MEM);
}

void app_main(void) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Gather Benchmark — Worker %d", WORKER_ID);
    ESP_LOGI(TAG, "========================================");

    go_sem = xSemaphoreCreateBinary();
    wifi_init();

    send_buf = (int8_t *)malloc(MAX_SEND_SIZE);
    if (!send_buf) { ESP_LOGE(TAG, "MALLOC FAILED"); return; }
    /* Fill with non-zero pattern */
    for (int i = 0; i < MAX_SEND_SIZE; i++)
        send_buf[i] = (int8_t)(i & 0x7F);

    ESP_LOGI(TAG, "Worker %d ready.", WORKER_ID);

    while (1) {
        send_ready();

        if (xSemaphoreTake(go_sem, pdMS_TO_TICKS(5000)) != pdTRUE)
            continue;

        uint16_t size = requested_size;
        if (size > MAX_SEND_SIZE) size = MAX_SEND_SIZE;

        int64_t t0 = esp_timer_get_time();
        send_result(size);
        int64_t t1 = esp_timer_get_time();

        ESP_LOGI(TAG, "Sent %d B (%d chunks) in %lld us",
                 size, swarm_num_chunks(size), t1 - t0);
    }
}
