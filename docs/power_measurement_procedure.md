# Power & Energy Re-Measurement Procedure (R1-6, Table 12)

The manuscript's power/energy numbers must come from on-device INA219 measurement.
The earlier measurement artifacts were overwritten by later experiments; this is
the turnkey procedure to re-capture them reproducibly. Analysis is done by
`models/power_analyze.py`, which derives Table 12 directly from the captured logs.

## Hardware

- 1× INA219 high-side current/voltage breakout (GY-219 / Adafruit, **0.1 Ω** shunt).
- Wire the INA219 **in series on the board's 5 V supply rail**:
  `5 V source (USB/battery) → INA219 Vin+ … Vin− → board 5 V`.
- INA219 I²C to the board under test: **SDA = GPIO8, SCL = GPIO9**, plus GND.
- One sensor is enough: measure board-by-board (move it between boards). If you
  have 5 sensors, instrument every board at once and a single coordinator log
  captures the whole cluster (workers relay their means as `PWRW`).

## Build flag

All three firmwares honor an env var:

```bash
export SWARM_POWER_MEASURE=1     # → compiles with -DPOWER_MEASURE
```

Build/flash as usual (`idf.py build`, then a SEPARATE `idf.py -p <port> flash`).
The firmware samples at ~1 kHz, persists the lifetime mean to NVS (so it survives
the reset that USB reconnection causes), and prints
`PWRDUMP,<tag>,count,mean_mw,mean_ma,min_mw,max_mw`.

## Captures

### 1. Single-node (tag SINGLE)
```bash
export SWARM_POWER_MEASURE=1
cd single_inference && idf.py build && idf.py -p <PORT> flash
# run the 1,000-image (or representative) inference, then:
idf.py -p <PORT> monitor   # capture until you see PWRDUMP,SINGLE,...
```
Save the serial capture to `logs/power_runs/single.log`.

### 2. Cluster N=4 (coordinator tag COORD + workers)
Flash **all five boards** with `SWARM_POWER_MEASURE=1`:
```bash
cd swarm_coordinator && idf.py build && idf.py -p <COORD_PORT> flash
cd ../swarm_worker     && idf.py build   # flash each worker (SWARM_WORKER_ID 0..3) to its port
```
Run the N=4 experiment. On the coordinator serial you get both `PWRDUMP,COORD,...`
and per-image `PWRW,<img>,w0_mw,w0_ma,w1_mw,...` (each worker's own INA219 mean,
relayed over ESP-NOW). Save to `logs/power_runs/coord_n4.log`.

If you have only one sensor, instead capture each board separately
(`single`, `coord`, `w0`..`w3` logs) and pass them all to the analyzer.

## Derive Table 12

```bash
conda activate swarm-ml
# all-in-one (coordinator log carries PWRW for the workers):
python models/power_analyze.py --single logs/power_runs/single.log \
                               --coord  logs/power_runs/coord_n4.log
# or board-by-board:
python models/power_analyze.py --single single.log --coord coord.log \
       --worker w0.log --worker w1.log --worker w2.log --worker w3.log
```

This writes `results/power.json` and prints the exact Table 12 cells:
`Single-node & <W> & 1897 & <J>` and `Distributed N=4 & <W> & 2115 & <J>`
(energy = power × log-verified latency). The analyzer **refuses** any board log
whose mean is < 50 mW, so a dead/disconnected shunt can never become a number.

## After capture

1. Commit `logs/power_runs/*.log` and `results/power.json`.
2. Update Table 12 + §"Power and Energy" + response R1-6 with the measured cells.
3. The legacy datasheet estimate `pub/scripts/power_estimation.py` is superseded
   (it is marked deprecated); `power_analyze.py` is the source of record.
