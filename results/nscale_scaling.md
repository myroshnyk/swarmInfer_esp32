# R2-7 / R2-8: non-divisible channels (N=3) and N=5 scaling

Isolated experiment (swarm_*_nscale firmware, uneven output-channel partitioning).
Released even-split pipeline untouched. All runs bit-exact vs the N=4 reference.

## Uneven partitioning (non-divisible C_out)
FatCNN channels 64 / 128 / 256 are divisible by neither 3 nor 5. The first
C_out mod N workers each take one extra channel:

| N | conv1 shards | conv2 shards | conv3 shards | max load imbalance |
|---|---|---|---|---|
| 3 | [22, 21, 21] | [43, 43, 42] | [86, 85, 85] | 1.031x |
| 5 | [13, 13, 13, 13, 12] | [26, 26, 26, 25, 25] | [52, 51, 51, 51, 51] | 1.016x |

Correctness: **N=3 and N=5: 150/150 predictions identical to N=4 reference (exact partitioning)**

## Scalability (end-to-end latency vs N)
| N | latency (ms) | speedup vs N=2 | source |
|---|---|---|---|
| 2 | 3653 | 1.00x | paper (even) |
| 3 | 2619 | 1.39x | nscale (uneven) |
| 4 | 2115 | 1.73x | paper (even) |
| 5 | 1838 | 1.99x | nscale (uneven) |

N=5 continues the monotonic speedup trend; the uneven-shard load imbalance
(<=3%) does not break the trend. The small imbalance is the only cost of
non-divisible channels — predictions remain bit-exact.
