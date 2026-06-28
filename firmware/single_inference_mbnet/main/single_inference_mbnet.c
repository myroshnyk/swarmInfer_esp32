/* SwarmInfer: scaled MobileNet single-node on-device validation (R2-11, Phase 5a).
 *
 * Runs the full 18-layer INT8 MobileNet on ONE ESP32-S3 (weights in flash,
 * 96x96 activations in PSRAM) and self-validates against the host/numpy
 * reference: every layer's output checksum and the final prediction must match
 * mbnet_testvec.h exactly. Building block before the distributed firmware.
 *
 * Isolated: reuses common/{tensor_ops,mbnet_ops}.c; touches no FatCNN code.
 */
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "tensor_ops.h"
#include "mbnet_ops.h"
#include "mbnet_weights.h"
#include "mbnet_testvec.h"

static const char *TAG = "MBNET";
#define BUF (512 * 1024)

static int64_t checksum(const int8_t *d, int n) {
    int64_t s = 0;
    for (int i = 0; i < n; i++) s += d[i];
    return s;
}

void app_main(void) {
    ESP_LOGI(TAG, "scaled MobileNet single-node, %d layers, input %dx%dx%d",
             MB_NUM_LAYERS, MB_IN_H, MB_IN_W, MB_IN_C);
    ESP_LOGI(TAG, "PSRAM free: %u KB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    int8_t *a = heap_caps_malloc(BUF, MALLOC_CAP_SPIRAM);
    int8_t *b = heap_caps_malloc(BUF, MALLOC_CAP_SPIRAM);
    int8_t *pad = heap_caps_malloc(BUF, MALLOC_CAP_SPIRAM);
    if (!a || !b || !pad) { ESP_LOGE(TAG, "PSRAM alloc failed"); return; }

    Tensor3D cur = { a, MB_IN_H, MB_IN_W, MB_IN_C };
    memcpy(a, mb_test_input, (size_t)MB_IN_H * MB_IN_W * MB_IN_C);

    int mism = 0;
    int8_t *dst = b;
    int64_t t0 = esp_timer_get_time();

    for (int li = 0; li < MB_NUM_LAYERS - 1; li++) {
        Tensor3D out = { dst, 0, 0, 0 };
        mb_run_conv_layer(&mb_layers[li], &cur, &out, pad);
        int64_t cs = checksum(out.data, out.h * out.w * out.c);
        if (cs != mb_checksums[li]) {
            mism++;
            ESP_LOGW(TAG, "L%02d %s MISMATCH cs=%lld ref=%lld",
                     li, mb_checksum_names[li], (long long)cs, (long long)mb_checksums[li]);
        }
        cur = out;
        dst = (cur.data == a) ? b : a;
    }

    /* GAP */
    int ngap = cur.c;
    int8_t *gap = pad;
    Tensor1D gout = { gap, ngap };
    global_avgpool_int8(&cur, &gout, MB_GAP_IN_ZP, MB_GAP_IN_SCALE,
                        MB_GAP_SCALE, MB_GAP_ZP);
    if (checksum(gap, ngap) != mb_checksums[MB_NUM_LAYERS - 1]) mism++;

    /* dense */
    const MbLayer *D = &mb_layers[MB_NUM_LAYERS - 1];
    int8_t logits[16];
    Tensor1D din = { gap, ngap }, dout = { logits, D->Cout };
    FixedPointMultiplier dm = { D->mult, D->shift };
    dense_int8(&din, D->w, D->b, &dout, dm, D->in_zp, D->w_zp, D->out_zp);
    if (checksum(logits, D->Cout) != mb_checksums[MB_NUM_LAYERS]) mism++;

    int64_t t1 = esp_timer_get_time();
    int pred = argmax_int8(logits, D->Cout);

    ESP_LOGI(TAG, "pred=%d expected=%d label=%d  latency=%lld ms  checksum_mismatches=%d",
             pred, MB_EXPECTED_PRED, MB_TEST_LABEL, (t1 - t0) / 1000, mism);
    ESP_LOGI(TAG, "RESULT: %s",
             (mism == 0 && pred == MB_EXPECTED_PRED) ? "BIT-EXACT PASS" : "FAIL");

    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}
