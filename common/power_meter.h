/* Background power meter: samples one self-attached INA219 (on the same
 * board) at ~1 kHz from a low-priority task pinned to core 1, so it does
 * not perturb inference running on core 0. Maintains O(1) running
 * accumulators only (no growing buffer) -> constant RAM, no overflow.
 *
 * Two consumers:
 *   - workers piggyback their lifetime-mean power into RESULT_DONE
 *     (power_meter_lifetime_mean), which the coordinator logs over USB;
 *   - the coordinator / single-node board self-measure and, after the run,
 *     repeatedly print the aggregate (power_meter_dump) so it can be read
 *     once USB is reconnected ("buffer+dump").
 *
 * Compiled only when SWARM_POWER_MEASURE is set. */
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"
#include "ina219.h"   /* exposes INA219_DEFAULT_ADDR */

/* Create an I2C master bus on (gpio_sda, gpio_scl), init the INA219 at
 * `addr`, and start the sampling task. Returns ESP_OK only if the sensor
 * responds; on failure nothing is started and the meter stays unavailable
 * (callers continue normally). */
esp_err_t power_meter_start(int gpio_sda, int gpio_scl, uint8_t addr);

bool power_meter_available(void);

/* Lifetime mean since start, clamped to uint16 (mW, mA). Returns false if
 * no sensor / no samples yet (outputs set to 0). */
bool power_meter_lifetime_mean(uint16_t *mean_mw, uint16_t *mean_ma);

/* Freeze the accumulator: the sampling task stops adding samples. Call at the
 * end of a run (before idling) so the dumped mean reflects only the active
 * period, regardless of when USB is reconnected to read it. */
void power_meter_freeze(void);

/* Print one aggregate line: PWRDUMP,<tag>,count,mean_mw,mean_ma,min_mw,max_mw */
void power_meter_dump(const char *tag);

/* Persist the frozen aggregate to NVS (survives the reset on USB reconnect). */
void power_meter_save(const char *tag);

/* Print any NVS-saved aggregate as PWRSAVED,<tag>,... (call early on boot). */
void power_meter_report_saved(void);
