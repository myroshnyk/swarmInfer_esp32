#include "ina219.h"
#include <stdint.h>

/* INA219 register map */
#define INA219_REG_CONFIG   0x00
#define INA219_REG_SHUNT    0x01
#define INA219_REG_BUS      0x02

/* Config: 16 V bus range, PGA /8 (+/-320 mV), 12-bit shunt+bus ADC,
 * continuous shunt-and-bus mode.
 *   BRNG=0 (bit13) | PG=0b11 (bits12:11) | BADC=0b0011 (bits10:7)
 *   | SADC=0b0011 (bits6:3) | MODE=0b111 (bits2:0) = 0x199F */
#define INA219_CONFIG_VALUE 0x199F

/* Shunt-voltage LSB = 10 uV; bus-voltage LSB = 4 mV (bits 15:3). */
#define INA219_SHUNT_LSB_V  1.0e-5f
#define INA219_BUS_LSB_V    4.0e-3f

#define I2C_TIMEOUT_MS      50

static esp_err_t reg_write16(ina219_t *s, uint8_t reg, uint16_t val) {
    uint8_t buf[3] = { reg, (uint8_t)(val >> 8), (uint8_t)(val & 0xFF) };
    return i2c_master_transmit(s->dev, buf, sizeof(buf), I2C_TIMEOUT_MS);
}

static esp_err_t reg_read16(ina219_t *s, uint8_t reg, uint16_t *out) {
    uint8_t rx[2];
    esp_err_t err = i2c_master_transmit_receive(s->dev, &reg, 1, rx, 2, I2C_TIMEOUT_MS);
    if (err != ESP_OK) return err;
    *out = ((uint16_t)rx[0] << 8) | rx[1];
    return ESP_OK;
}

esp_err_t ina219_init(i2c_master_bus_handle_t bus, uint8_t addr,
                      float shunt_ohm, ina219_t *out) {
    i2c_device_config_t devcfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address  = addr,
        .scl_speed_hz    = 400000,
    };
    esp_err_t err = i2c_master_bus_add_device(bus, &devcfg, &out->dev);
    if (err != ESP_OK) return err;
    out->shunt_ohm = shunt_ohm;

    err = reg_write16(out, INA219_REG_CONFIG, INA219_CONFIG_VALUE);
    if (err != ESP_OK) return err;

    /* Probe: a successful bus-voltage read confirms the sensor is present. */
    uint16_t probe;
    return reg_read16(out, INA219_REG_BUS, &probe);
}

esp_err_t ina219_read(ina219_t *s, float *bus_v, float *current_a, float *power_w) {
    uint16_t shunt_raw, bus_raw;
    esp_err_t err = reg_read16(s, INA219_REG_SHUNT, &shunt_raw);
    if (err != ESP_OK) return err;
    err = reg_read16(s, INA219_REG_BUS, &bus_raw);
    if (err != ESP_OK) return err;

    float vshunt = (float)((int16_t)shunt_raw) * INA219_SHUNT_LSB_V;  /* signed */
    float vbus   = (float)(bus_raw >> 3) * INA219_BUS_LSB_V;
    float i      = vshunt / s->shunt_ohm;
    if (bus_v)     *bus_v = vbus;
    if (current_a) *current_a = i;
    if (power_w)   *power_w = vbus * i;
    return ESP_OK;
}
