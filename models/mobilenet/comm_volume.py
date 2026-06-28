"""R2-11: per-pointwise-layer communication volume for the scaled MobileNet.

Communication volume is determined by the activation tensor sizes (INT8, 1 byte
each), independent of the trained weight values. Each distributed pointwise round
broadcasts its input activation to all workers and gathers the full output back.

Writes results/mbnet_comm_volume.csv.
"""
import csv
import os

# pointwise (1x1) layers: (name, spatial_HW, Cin, Cout). Spatial is post-depthwise.
PW = [
    ("b1_pw", 48, 32, 64),
    ("b2_pw", 24, 64, 128),
    ("b3_pw", 24, 128, 128),
    ("b4_pw", 12, 128, 256),
    ("b5_pw", 12, 256, 256),
    ("b6_pw", 6, 256, 512),
    ("b7_pw", 6, 512, 512),
    ("b8_pw", 3, 512, 1024),
]

here = os.path.join(os.path.dirname(__file__), "..", "..", "results")
rows = []
bc = ga = 0
for name, hw, cin, cout in PW:
    in_b = hw * hw * cin       # broadcast (input activation), INT8
    out_b = hw * hw * cout     # gather (full output activation), INT8
    bc += in_b
    ga += out_b
    rows.append((name, f"{hw}x{hw}", cin, cout, in_b, out_b))

with open(os.path.join(here, "mbnet_comm_volume.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "spatial", "Cin", "Cout", "broadcast_B", "gather_B"])
    w.writerows(rows)
    w.writerow(["TOTAL", "", "", "", bc, ga])

print(f"broadcast {bc} B ({bc/1024:.1f} KB) | "
      f"gather {ga} B ({ga/1024:.1f} KB) | "
      f"total {(bc+ga)} B ({(bc+ga)/1024:.1f} KB)")
print(f"wrote {os.path.join(here, 'mbnet_comm_volume.csv')}")
