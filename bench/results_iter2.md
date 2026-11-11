| run | seeds | val loss | gap to AdamW | gain vs gefen-base | state MiB | update err | mom err | v err | k tok/s | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| adamw-base | 3 | 5.0018 | +0.0000 ± 0.0000 |  | 756 | nan | nan | nan | 96 |  |
| gefen-base | 3 | 5.1540 | +0.1522 ± 0.0204 |  | 96 | 0.374 | 0.025 | 0.374 | 95 |  |
| gefen-unquant | 3 | 5.1475 | +0.1458 ± 0.0215 | +0.0064 ± 0.0051 | 96 | 0.373 | 0.025 | 0.373 | 95 | drop: not significant |
| gefen-vprof_col | 3 | 5.0710 | +0.0692 ± 0.0108 | +0.0830 ± 0.0313 | 96 | 0.274 | 0.035 | 0.270 | 94 | keep |
| gefen-vprof_full | 3 | 5.0564 | +0.0547 ± 0.0094 | +0.0975 ± 0.0283 | 474 | 0.107 | 0.090 | 0.054 | 94 | drop: memory |
