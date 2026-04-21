"""
SwarmInfer: Datasheet-based Power & Energy Estimation

Uses ESP32-S3 datasheet power profiles (Table 14, ESP32-S3 TRM v1.4)
combined with measured timing data to estimate energy-per-inference.

Usage: python power_estimation.py
"""

# ============================================================
# ESP32-S3 Power Profile (from datasheet, 3.3V, typical values)
# ============================================================
VOLTAGE = 3.3  # V (USB 5V → onboard LDO → 3.3V)

# Current draw in different states (mA)
CURRENT_CPU_ACTIVE_160MHZ = 80    # CPU active, no WiFi
CURRENT_CPU_ACTIVE_240MHZ = 100   # CPU active, no WiFi
CURRENT_WIFI_TX = 335             # WiFi transmitting
CURRENT_WIFI_RX = 100             # WiFi receiving
CURRENT_IDLE = 20                 # Idle (modem sleep, CPU waiting)

# ============================================================
# Measured Timing Data (from experiments, in ms)
# ============================================================

experiments = {
    # ── Distributed Inference (N=4, 160 MHz, sparse) ──
    "N=4, 160MHz, sparse": {
        "n_workers": 4,
        "cpu_mhz": 160,
        "total_latency_ms": 2115,
        "per_layer": [
            # (layer, broadcast_ms, gather_ms, assemble_ms)
            ("L1", 3, 460, 0.4),
            ("L2", 105, 735, 0.1),
            ("L3", 13, 720, 0.01),
        ],
        "dense_ms": 2,  # Dense1 + Dense2 on coordinator
        "description": "Full pipeline with bitmap sparsification",
    },
    # ── Distributed Inference (N=4, 240 MHz, sparse) ──
    "N=4, 240MHz, sparse": {
        "n_workers": 4,
        "cpu_mhz": 240,
        "total_latency_ms": 1623,
        "per_layer": [
            ("L1", 3, 340, 0.4),
            ("L2", 105, 540, 0.1),
            ("L3", 13, 530, 0.01),
        ],
        "dense_ms": 1.5,
        "description": "Full pipeline, 240 MHz CPU",
    },
    # ── Distributed Inference (N=2, 160 MHz, no sparse) ──
    "N=2, 160MHz": {
        "n_workers": 2,
        "cpu_mhz": 160,
        "total_latency_ms": 3788,
        "per_layer": [
            ("L1", 3, 839, 0.3),
            ("L2", 105, 1437, 0.1),
            ("L3", 13, 1353, 0.01),
        ],
        "dense_ms": 2,
        "description": "2 workers, no sparsification",
    },
    # ── Single-Node Baseline (FatCNN-Lite, 160 MHz) ──
    "Single-node baseline": {
        "n_workers": 0,  # single node
        "cpu_mhz": 160,
        "total_latency_ms": 1890,
        "per_layer": [],  # all local
        "dense_ms": 0,
        "description": "FatCNN-Lite on single ESP32-S3 (103K params)",
    },
}


def estimate_energy(exp):
    """Estimate energy per inference in millijoules (mJ)."""
    n = exp["n_workers"]
    cpu_mhz = exp["cpu_mhz"]
    total_ms = exp["total_latency_ms"]

    i_cpu = CURRENT_CPU_ACTIVE_240MHZ if cpu_mhz == 240 else CURRENT_CPU_ACTIVE_160MHZ

    if n == 0:
        # Single node: all CPU compute
        # Power = V × I_cpu
        power_w = VOLTAGE * i_cpu / 1000.0
        energy_mj = power_w * total_ms
        return {
            "nodes": 1,
            "coord_energy_mj": energy_mj,
            "worker_energy_mj": 0,
            "total_energy_mj": energy_mj,
            "avg_power_w": power_w,
        }

    # Distributed: coordinator + N workers
    # ── Coordinator timeline ──
    # Broadcast: WiFi TX
    # Gather wait: WiFi RX (receiving chunks)
    # Dense compute: CPU active
    # Idle between: minimal
    bcast_ms = sum(l[1] for l in exp["per_layer"])
    gather_ms = sum(l[2] for l in exp["per_layer"])
    dense_ms = exp["dense_ms"]
    coord_idle_ms = total_ms - bcast_ms - gather_ms - dense_ms

    coord_energy_mj = VOLTAGE * (
        CURRENT_WIFI_TX * bcast_ms / 1000.0
        + CURRENT_WIFI_RX * gather_ms / 1000.0
        + i_cpu * dense_ms / 1000.0
        + CURRENT_IDLE * max(0, coord_idle_ms) / 1000.0
    )

    # ── Worker timeline (each) ──
    # Receive broadcast: WiFi RX (same as bcast_ms)
    # Compute: CPU active (gather_ms includes compute + send)
    # Send results: WiFi TX (part of gather_ms)
    # For estimation: ~80% of gather is compute, ~20% is WiFi TX
    compute_fraction = 0.80  # from gather benchmark
    send_fraction = 0.20

    worker_rx_ms = bcast_ms
    worker_compute_ms = gather_ms * compute_fraction
    worker_tx_ms = gather_ms * send_fraction
    worker_idle_ms = total_ms - worker_rx_ms - worker_compute_ms - worker_tx_ms

    worker_energy_mj = VOLTAGE * (
        CURRENT_WIFI_RX * worker_rx_ms / 1000.0
        + i_cpu * worker_compute_ms / 1000.0
        + CURRENT_WIFI_TX * worker_tx_ms / 1000.0
        + CURRENT_IDLE * max(0, worker_idle_ms) / 1000.0
    )

    total_energy_mj = coord_energy_mj + n * worker_energy_mj
    total_nodes = 1 + n
    avg_power_w = total_energy_mj / total_ms

    return {
        "nodes": total_nodes,
        "coord_energy_mj": coord_energy_mj,
        "worker_energy_mj": worker_energy_mj,
        "total_energy_mj": total_energy_mj,
        "avg_power_w": avg_power_w,
    }


def main():
    print("=" * 70)
    print("  SwarmInfer: Datasheet-Based Power & Energy Estimation")
    print("  Source: ESP32-S3 Technical Reference Manual v1.4, Table 14")
    print("=" * 70)
    print()
    print(f"  Supply voltage:  {VOLTAGE} V")
    print(f"  CPU active @160: {CURRENT_CPU_ACTIVE_160MHZ} mA")
    print(f"  CPU active @240: {CURRENT_CPU_ACTIVE_240MHZ} mA")
    print(f"  WiFi TX:         {CURRENT_WIFI_TX} mA")
    print(f"  WiFi RX:         {CURRENT_WIFI_RX} mA")
    print(f"  Idle:            {CURRENT_IDLE} mA")
    print()

    # Results table
    print(f"{'Config':<28} {'Nodes':>5} {'Latency':>9} {'Total E':>10} {'Avg P':>8} {'E/node':>10}")
    print(f"{'':28} {'':>5} {'(ms)':>9} {'(mJ)':>10} {'(W)':>8} {'(mJ)':>10}")
    print("-" * 75)

    for name, exp in experiments.items():
        r = estimate_energy(exp)
        e_per_node = r["total_energy_mj"] / r["nodes"]
        print(f"{name:<28} {r['nodes']:>5} {exp['total_latency_ms']:>9} "
              f"{r['total_energy_mj']:>10.1f} {r['avg_power_w']:>8.3f} {e_per_node:>10.1f}")

    print()
    print("=" * 70)
    print("  Detailed Breakdown")
    print("=" * 70)

    for name, exp in experiments.items():
        r = estimate_energy(exp)
        print(f"\n  {name}: {exp['description']}")
        print(f"    Coordinator energy: {r['coord_energy_mj']:.1f} mJ")
        if exp["n_workers"] > 0:
            print(f"    Worker energy (each): {r['worker_energy_mj']:.1f} mJ")
            print(f"    Workers × {exp['n_workers']}: {exp['n_workers'] * r['worker_energy_mj']:.1f} mJ")
        print(f"    Total system energy: {r['total_energy_mj']:.1f} mJ")
        print(f"    Average system power: {r['avg_power_w']:.3f} W")

    print()
    print("Note: These are estimates based on ESP32-S3 datasheet typical values.")
    print("Actual power consumption may vary ±20% depending on temperature,")
    print("supply voltage, and individual chip characteristics.")


if __name__ == "__main__":
    main()
