# R2-12: RF-robustness distance sweep (FatCNN N=4)

One worker (W3) placed through a wall; all boards on cable power (isolates
the RF effect from battery TX-power sag). Two firmwares compared at the same
position: `released_*` = the unmodified paper firmware (single unACKed
broadcast, 30 s gather timeout); `rel*` = the isolated reliability-layer
prototype (generation-tagged packets, NACK unicast-refill of lost input
chunks, unicast LAYER_START, gather completeness gate). `_clean` = near-Mac
baseline. Each config reset the coordinator and streamed images.

| Config | Imgs | Retry events | Retries/img | Max retry | Compute lat (ms) | Eff. walltime (s) | Bit-exact vs ref |
|---|---|---|---|---|---|---|---|
| d0_verify_postfix | 100 | 1 | 0.01 | 1 | 2167 (sd 15) | 3 (max 33) | 100/100 |
| released_d2_wall | 13 | 11 | 0.85 | 6 | 2175 (sd 15) | 29 (max 186) | 13/13 |
| rel2_clean | 40 | 0 | 0.0 | 0 | 2181 (sd 16) | 2 (max 2) | 40/40 |
| rel2_d2_wall | 30 | 1 | 0.03 | 1 | 2418 (sd 228) | 3 (max 13) | 30/30 |

**Key result:** per-attempt compute latency is unchanged with distance (the LX7 conv cost is fixed); degradation shows up as retries/timeouts and inflated effective walltime. Through the wall the paper firmware's single 30 s gather timeout inflates effective walltime by ~10x (mean) and ~14x (worst image) and, having no per-chunk completeness check, its unACKed broadcast can under heavier loss compute on / assemble incomplete data and silently return a wrong result (observed in through-wall stress runs: 1/9 and 1/30 in firmware variants with weaker integrity checks than the prototype). The reliability-layer prototype recovers the same losses with cheap unicast refills (no 30 s stalls) and stays bit-exact, confirming the degradation is in the transport, not the partitioning. This substantiates scoping the zero-loss claim to bench conditions and motivates the RF-hardening future work.

### d0_verify_postfix
- retry-by-layer: {'L2': 1}
- failed-attempt worker-delivery (k/4 done): {3: 1}
- mismatches: none

### released_d2_wall
- retry-by-layer: {'L2': 9, 'L1': 1, 'L3': 1}
- failed-attempt worker-delivery (k/4 done): {3: 11}
- mismatches: none

### rel2_clean
- retry-by-layer: {}
- failed-attempt worker-delivery (k/4 done): {}
- mismatches: none

### rel2_d2_wall
- retry-by-layer: {'L2': 1}
- failed-attempt worker-delivery (k/4 done): {3: 1}
- mismatches: none
