#include "power_meter.h"
#include "ina219.h"
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_rom_sys.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "nvs.h"

/* 0.1 ohm is the shunt on the common GY-219 / Adafruit INA219 breakout. */
#define POWER_METER_SHUNT_OHM   0.1f
#define POWER_METER_TASK_CORE    1
#define POWER_METER_TASK_PRIO    1
#define POWER_METER_TASK_STACK   3072

static const char *TAG = "pmeter";

static ina219_t s_ina;
static i2c_master_bus_handle_t s_bus;
static volatile bool s_available = false;

/* Running accumulators (protected by s_mux). int64 sums cannot overflow at
 * ~1 kHz for any realistic run length. */
static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
static int64_t  s_sum_mw = 0;
static int64_t  s_sum_ma = 0;
static uint64_t s_count  = 0;
static int32_t  s_min_mw = INT32_MAX;
static int32_t  s_max_mw = 0;
static volatile bool s_frozen = false;

static void power_meter_task(void *arg) {
    (void)arg;
    uint32_t yield_ctr = 0;
    uint32_t print_ctr = 0;
    int32_t  last_mw = 0, last_ma = 0;
    float    last_v = 0.0f;
    while (1) {
        float v, i, p;
        if (ina219_read(&s_ina, &v, &i, &p) == ESP_OK) {
            int32_t mw = (int32_t)(p * 1000.0f);
            int32_t ma = (int32_t)(i * 1000.0f);
            if (mw < 0) mw = -mw;   /* tolerate reversed shunt wiring */
            if (ma < 0) ma = -ma;
            last_mw = mw; last_ma = ma; last_v = v;
            if (!s_frozen) {
                taskENTER_CRITICAL(&s_mux);
                s_sum_mw += mw;
                s_sum_ma += ma;
                s_count++;
                if (mw < s_min_mw) s_min_mw = mw;
                if (mw > s_max_mw) s_max_mw = mw;
                taskEXIT_CRITICAL(&s_mux);
            }
        }
        /* Local live print (~1/s) so any wired board can be validated from its
         * own serial without completing a run. PWR,<n>,<inst_mw>,<inst_ma>,<bus_mV> */
        if ((++print_ctr % 1000) == 0) {
            taskENTER_CRITICAL(&s_mux);
            uint64_t cnt = s_count;
            taskEXIT_CRITICAL(&s_mux);
            printf("PWR,%llu,%ld,%ld,%d\n", (unsigned long long)cnt,
                   (long)last_mw, (long)last_ma, (int)(last_v * 1000.0f));
        }
        esp_rom_delay_us(900);              /* ~1 kHz sampling */
        if ((++yield_ctr & 0x0F) == 0)      /* yield ~every 16 ms to feed WDT */
            vTaskDelay(1);
    }
}

esp_err_t power_meter_start(int gpio_sda, int gpio_scl, uint8_t addr) {
    i2c_master_bus_config_t buscfg = {
        .i2c_port = -1,                     /* auto-select a free port */
        .sda_io_num = gpio_sda,
        .scl_io_num = gpio_scl,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    esp_err_t err = i2c_new_master_bus(&buscfg, &s_bus);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "i2c bus init failed: %s", esp_err_to_name(err));
        return err;
    }
    /* Retry the requested address a few times: a marginal contact can ACK its
     * address yet fail the first register transaction. */
    err = ESP_FAIL;
    for (int attempt = 0; attempt < 5 && err != ESP_OK; attempt++) {
        err = ina219_init(s_bus, addr, POWER_METER_SHUNT_OHM, &s_ina);
        if (err != ESP_OK) vTaskDelay(pdMS_TO_TICKS(20));
    }
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "INA219 @0x%02X not responding (%s); scanning INA219 range 0x40-0x4F...",
                 addr, esp_err_to_name(err));
        /* Auto-detect: the INA219 occupies 0x40-0x4F depending on A0/A1 straps.
         * Use the first one that both ACKs and initialises (incl. addr, retried). */
        for (uint8_t a = 0x40; a <= 0x4F; a++) {
            if (i2c_master_probe(s_bus, a, 50) != ESP_OK) continue;
            ESP_LOGW(TAG, "  I2C device at 0x%02X; trying as INA219", a);
            for (int attempt = 0; attempt < 3; attempt++) {
                if (ina219_init(s_bus, a, POWER_METER_SHUNT_OHM, &s_ina) == ESP_OK) {
                    addr = a; err = ESP_OK; break;
                }
                vTaskDelay(pdMS_TO_TICKS(20));
            }
            if (err == ESP_OK) break;
        }
        if (err != ESP_OK) {
            int n = 0;
            for (uint8_t a = 0x08; a <= 0x77; a++)
                if (i2c_master_probe(s_bus, a, 50) == ESP_OK) {
                    ESP_LOGW(TAG, "  full-scan: device at 0x%02X", a); n++;
                }
            ESP_LOGW(TAG, "No INA219 on SDA=%d/SCL=%d (%d total I2C dev); power meter disabled",
                     gpio_sda, gpio_scl, n);
            return err;
        }
    }
    s_available = true;
    xTaskCreatePinnedToCore(power_meter_task, "pmeter", POWER_METER_TASK_STACK,
                            NULL, POWER_METER_TASK_PRIO, NULL, POWER_METER_TASK_CORE);
    ESP_LOGI(TAG, "INA219 @0x%02X on SDA=%d SCL=%d; sampling on core %d",
             addr, gpio_sda, gpio_scl, POWER_METER_TASK_CORE);
    return ESP_OK;
}

bool power_meter_available(void) { return s_available; }

bool power_meter_lifetime_mean(uint16_t *mean_mw, uint16_t *mean_ma) {
    if (mean_mw) *mean_mw = 0;
    if (mean_ma) *mean_ma = 0;
    if (!s_available) return false;
    taskENTER_CRITICAL(&s_mux);
    int64_t sum_mw = s_sum_mw, sum_ma = s_sum_ma;
    uint64_t cnt = s_count;
    taskEXIT_CRITICAL(&s_mux);
    if (cnt == 0) return false;
    int64_t mw = sum_mw / (int64_t)cnt;
    int64_t ma = sum_ma / (int64_t)cnt;
    if (mw > 65535) mw = 65535;
    if (ma > 65535) ma = 65535;
    if (mean_mw) *mean_mw = (uint16_t)mw;
    if (mean_ma) *mean_ma = (uint16_t)ma;
    return true;
}

void power_meter_freeze(void) { s_frozen = true; }

static void ensure_nvs(void) {
    static bool done = false;
    if (done) return;
    esp_err_t e = nvs_flash_init();
    if (e == ESP_ERR_NVS_NO_FREE_PAGES || e == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    done = true;
}

/* Persist the (frozen) aggregate to NVS so it survives the reset that the
 * UART-bridge triggers when USB is reconnected to read it. */
void power_meter_save(const char *tag) {
    power_meter_freeze();
    ensure_nvs();
    nvs_handle_t h;
    if (nvs_open("pmeter", NVS_READWRITE, &h) != ESP_OK) return;
    taskENTER_CRITICAL(&s_mux);
    uint64_t cnt = s_count; int64_t smw = s_sum_mw, sma = s_sum_ma;
    int32_t mn = s_min_mw, mx = s_max_mw;
    taskEXIT_CRITICAL(&s_mux);
    nvs_set_u64(h, "cnt", cnt);
    nvs_set_i64(h, "smw", smw);
    nvs_set_i64(h, "sma", sma);
    nvs_set_i32(h, "mn", mn);
    nvs_set_i32(h, "mx", mx);
    nvs_set_str(h, "tag", tag ? tag : "?");
    nvs_commit(h);
    nvs_close(h);
}

/* On boot, print any previously-saved aggregate as
 * PWRSAVED,<tag>,count,mean_mw,mean_ma,min_mw,max_mw (or PWRSAVED,NONE). */
void power_meter_report_saved(void) {
    ensure_nvs();
    nvs_handle_t h;
    if (nvs_open("pmeter", NVS_READONLY, &h) != ESP_OK) { printf("PWRSAVED,NONE\n"); return; }
    uint64_t cnt = 0;
    if (nvs_get_u64(h, "cnt", &cnt) != ESP_OK || cnt == 0) {
        nvs_close(h); printf("PWRSAVED,NONE\n"); return;
    }
    int64_t smw = 0, sma = 0; int32_t mn = 0, mx = 0;
    nvs_get_i64(h, "smw", &smw); nvs_get_i64(h, "sma", &sma);
    nvs_get_i32(h, "mn", &mn);   nvs_get_i32(h, "mx", &mx);
    char tag[16] = {0}; size_t tl = sizeof(tag); nvs_get_str(h, "tag", tag, &tl);
    nvs_close(h);
    printf("PWRSAVED,%s,%llu,%lld,%lld,%ld,%ld\n", tag, (unsigned long long)cnt,
           (long long)(smw / (int64_t)cnt), (long long)(sma / (int64_t)cnt),
           (long)mn, (long)mx);
}

void power_meter_dump(const char *tag) {
    if (!s_available) {
        printf("PWRDUMP,%s,NO_SENSOR\n", tag ? tag : "?");
        return;
    }
    taskENTER_CRITICAL(&s_mux);
    int64_t sum_mw = s_sum_mw, sum_ma = s_sum_ma;
    uint64_t cnt = s_count;
    int32_t mn = s_min_mw, mx = s_max_mw;
    taskEXIT_CRITICAL(&s_mux);
    int64_t mean_mw = cnt ? sum_mw / (int64_t)cnt : 0;
    int64_t mean_ma = cnt ? sum_ma / (int64_t)cnt : 0;
    printf("PWRDUMP,%s,%llu,%lld,%lld,%ld,%ld\n", tag ? tag : "?",
           (unsigned long long)cnt, (long long)mean_mw, (long long)mean_ma,
           (long)(cnt ? mn : 0), (long)mx);
}
