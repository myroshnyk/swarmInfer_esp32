/**
 * SwarmInfer: Single-Node FatCNN-Lite Batch Inference
 *
 * Runs full FatCNN-Lite on ONE ESP32-S3 as baseline.
 * Batch mode: runs BATCH_SIZE images, logs per-image CSV for analysis.
 *
 * Architecture:
 *   Conv1 5x5 3->32 + ReLU + MaxPool2x2
 *   Conv2 3x3 32->64 + ReLU + MaxPool2x2
 *   Conv3 3x3 64->128 + ReLU + GlobalAvgPool
 *   Dense1 128->64 + ReLU
 *   Dense2 64->10 -> argmax
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensor_ops.h"
#include "fatcnn_lite_weights.h"
#include "test_images_batch.h"

static const char *TAG = "INFERENCE";

static const char *cifar_classes[] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

void app_main(void)
{
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  FatCNN-Lite Single-Node Batch Inference");
    ESP_LOGI(TAG, "  Images: %d", BATCH_SIZE);
    ESP_LOGI(TAG, "========================================");

    ESP_LOGI(TAG, "Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());
    ESP_LOGI(TAG, "Free internal: %lu bytes",
             (unsigned long)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    // Precompute fixed-point multipliers
    FixedPointMultiplier conv1_m = compute_requant_multiplier(INPUT_SCALE, CONV1_KSCALE, CONV1_OSCALE);
    FixedPointMultiplier conv2_m = compute_requant_multiplier(CONV1_OSCALE, CONV2_KSCALE, CONV2_OSCALE);
    FixedPointMultiplier conv3_m = compute_requant_multiplier(CONV2_OSCALE, CONV3_KSCALE, CONV3_OSCALE);
    FixedPointMultiplier dense1_m = compute_requant_multiplier(CONV3_OSCALE, DENSE1_KSCALE, DENSE1_OSCALE);
    FixedPointMultiplier dense2_m = compute_requant_multiplier(DENSE1_OSCALE, DENSE2_KSCALE, DENSE2_OSCALE);

    // Allocate buffers (on heap, NOT stack!)
    int8_t *input_buf  = tensor_alloc(32 * 32 * 3);
    int8_t *conv1_out  = tensor_alloc(32 * 32 * 32);
    int8_t *pool1_out  = tensor_alloc(16 * 16 * 32);
    int8_t *conv2_out  = tensor_alloc(16 * 16 * 64);
    int8_t *pool2_out  = tensor_alloc(8 * 8 * 64);
    int8_t *conv3_out  = tensor_alloc(8 * 8 * 128);
    int8_t *gap_out    = tensor_alloc(128);
    int8_t *dense1_out = tensor_alloc(64);
    int8_t *dense2_out = tensor_alloc(10);

    if (!input_buf || !conv1_out || !pool1_out || !conv2_out ||
        !pool2_out || !conv3_out || !gap_out || !dense1_out || !dense2_out) {
        ESP_LOGE(TAG, "MALLOC FAILED! Not enough heap.");
        return;
    }

    ESP_LOGI(TAG, "Buffers allocated. Free heap: %lu bytes",
             (unsigned long)esp_get_free_heap_size());

    // CSV header
    ESP_LOGI(TAG, "CSV_HEADER,config,img,label,pred,match,total_us,conv1_us,conv2_us,conv3_us,dense1_us,dense2_us");

    // 1 warmup run (first inference is slower due to cache)
    memcpy(input_buf, batch_images[0], 32 * 32 * 3);
    {
        Tensor3D in3 = {input_buf, 32, 32, 3};
        Tensor3D out3 = {conv1_out, 32, 32, 32};
        conv2d_int8(&in3, conv1_kernel, conv1_bias, &out3, 5, 5, 1, 2, conv1_m,
                    (int8_t)INPUT_ZP, (int8_t)CONV1_KZP, (int8_t)CONV1_OZP);
    }
    ESP_LOGI(TAG, "Warmup done.");

    int correct = 0;
    int64_t total_latency_us = 0;

    for (int img = 0; img < BATCH_SIZE; img++) {
        memcpy(input_buf, batch_images[img], 32 * 32 * 3);

        int64_t t_total = esp_timer_get_time();
        int64_t t_start, conv1_us, conv2_us, conv3_us, dense1_us, dense2_us;

        // Layer 1: Conv1 5x5 3->32 + ReLU + MaxPool
        t_start = esp_timer_get_time();
        {
            Tensor3D in3 = {input_buf, 32, 32, 3};
            Tensor3D out3 = {conv1_out, 32, 32, 32};
            conv2d_int8(&in3, conv1_kernel, conv1_bias, &out3, 5, 5, 1, 2, conv1_m,
                        (int8_t)INPUT_ZP, (int8_t)CONV1_KZP, (int8_t)CONV1_OZP);
        }
        {
            Tensor3D t = {conv1_out, 32, 32, 32};
            relu_int8(&t, (int8_t)CONV1_OZP);
        }
        {
            Tensor3D in3 = {conv1_out, 32, 32, 32};
            Tensor3D out3 = {pool1_out, 16, 16, 32};
            maxpool2x2_int8(&in3, &out3);
        }
        conv1_us = esp_timer_get_time() - t_start;

        // Layer 2: Conv2 3x3 32->64 + ReLU + MaxPool
        t_start = esp_timer_get_time();
        {
            Tensor3D in3 = {pool1_out, 16, 16, 32};
            Tensor3D out3 = {conv2_out, 16, 16, 64};
            conv2d_int8(&in3, conv2_kernel, conv2_bias, &out3, 3, 3, 1, 1, conv2_m,
                        (int8_t)CONV1_OZP, (int8_t)CONV2_KZP, (int8_t)CONV2_OZP);
        }
        {
            Tensor3D t = {conv2_out, 16, 16, 64};
            relu_int8(&t, (int8_t)CONV2_OZP);
        }
        {
            Tensor3D in3 = {conv2_out, 16, 16, 64};
            Tensor3D out3 = {pool2_out, 8, 8, 64};
            maxpool2x2_int8(&in3, &out3);
        }
        conv2_us = esp_timer_get_time() - t_start;

        // Layer 3: Conv3 3x3 64->128 + ReLU + GAP
        t_start = esp_timer_get_time();
        {
            Tensor3D in3 = {pool2_out, 8, 8, 64};
            Tensor3D out3 = {conv3_out, 8, 8, 128};
            conv2d_int8(&in3, conv3_kernel, conv3_bias, &out3, 3, 3, 1, 1, conv3_m,
                        (int8_t)CONV2_OZP, (int8_t)CONV3_KZP, (int8_t)CONV3_OZP);
        }
        {
            Tensor3D t = {conv3_out, 8, 8, 128};
            relu_int8(&t, (int8_t)CONV3_OZP);
        }
        {
            Tensor3D in3 = {conv3_out, 8, 8, 128};
            Tensor1D out1 = {gap_out, 128};
            global_avgpool_int8(&in3, &out1,
                                (int8_t)CONV3_OZP, CONV3_OSCALE,
                                CONV3_OSCALE, (int8_t)CONV3_OZP);
        }
        conv3_us = esp_timer_get_time() - t_start;

        // Dense1 128->64 + ReLU
        t_start = esp_timer_get_time();
        {
            Tensor1D in1 = {gap_out, 128};
            Tensor1D out1 = {dense1_out, 64};
            dense_int8(&in1, dense1_kernel, dense1_bias, &out1, dense1_m,
                       (int8_t)CONV3_OZP, (int8_t)DENSE1_KZP, (int8_t)DENSE1_OZP);
        }
        {
            Tensor1D t = {dense1_out, 64};
            relu_int8_1d(&t, (int8_t)DENSE1_OZP);
        }
        dense1_us = esp_timer_get_time() - t_start;

        // Dense2 64->10
        t_start = esp_timer_get_time();
        {
            Tensor1D in1 = {dense1_out, 64};
            Tensor1D out1 = {dense2_out, 10};
            dense_int8(&in1, dense2_kernel, dense2_bias, &out1, dense2_m,
                       (int8_t)DENSE1_OZP, (int8_t)DENSE2_KZP, (int8_t)DENSE2_OZP);
        }
        dense2_us = esp_timer_get_time() - t_start;

        int64_t total_us = esp_timer_get_time() - t_total;
        int predicted = argmax_int8(dense2_out, 10);
        int label = batch_labels[img];
        bool match = (predicted == label);
        if (match) correct++;
        total_latency_us += total_us;

        ESP_LOGI(TAG, "[%3d/%d] pred=%d(%s) true=%d(%s) %s  %lld ms",
                 img + 1, BATCH_SIZE,
                 predicted, cifar_classes[predicted],
                 label, cifar_classes[label],
                 match ? "OK" : "MISS",
                 total_us / 1000);

        // CSV data line for automated parsing
        ESP_LOGI(TAG, "CSV,lite_n1,%d,%d,%d,%d,%lld,%lld,%lld,%lld,%lld,%lld",
                 img, label, predicted, match ? 1 : 0,
                 total_us, conv1_us, conv2_us, conv3_us, dense1_us, dense2_us);

        vTaskDelay(1);  // feed watchdog
    }

    // Summary
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "============================================================");
    ESP_LOGI(TAG, "  BATCH ACCURACY RESULTS (FatCNN-Lite, single node)");
    ESP_LOGI(TAG, "============================================================");
    ESP_LOGI(TAG, "Images:    %d", BATCH_SIZE);
    ESP_LOGI(TAG, "Correct:   %d", correct);
    ESP_LOGI(TAG, "Accuracy:  %d.%d%%", (100 * correct) / BATCH_SIZE,
             ((1000 * correct) / BATCH_SIZE) % 10);
    ESP_LOGI(TAG, "Avg latency: %lld ms", total_latency_us / BATCH_SIZE / 1000);
    ESP_LOGI(TAG, "============================================================");

    // Cleanup
    tensor_free(input_buf);
    tensor_free(conv1_out);
    tensor_free(pool1_out);
    tensor_free(conv2_out);
    tensor_free(pool2_out);
    tensor_free(conv3_out);
    tensor_free(gap_out);
    tensor_free(dense1_out);
    tensor_free(dense2_out);

    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}
