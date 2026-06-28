/* Minimal INA219 high-side current/voltage sensor driver (ESP-IDF v6.0
 * i2c_master API). Computes current directly from the shunt-voltage
 * register (no calibration-register dependency).
 *
 * Used only for power instrumentation (R1-6 / R2-3). Compiled into the
 * firmware only when SWARM_POWER_MEASURE is set; released builds are
 * byte-identical without it. */
#pragma once
#include <stdbool.h>
#include "driver/i2c_master.h"
#include "esp_err.h"

#define INA219_DEFAULT_ADDR  0x40

typedef struct {
    i2c_master_dev_handle_t dev;
    float shunt_ohm;
} ina219_t;

/* Add the INA219 at `addr` on an existing I2C master bus, push the config
 * register, and probe it. Returns ESP_OK only if the sensor responds. */
esp_err_t ina219_init(i2c_master_bus_handle_t bus, uint8_t addr,
                      float shunt_ohm, ina219_t *out);

/* One synchronous read. Any of bus_v / current_a / power_w may be NULL.
 * current_a = Vshunt / shunt_ohm; power_w = bus_v * current_a. */
esp_err_t ina219_read(ina219_t *s, float *bus_v, float *current_a, float *power_w);
