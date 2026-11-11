| run | seeds | val loss | gap to AdamW | gain vs gefen-base | state MiB | update err | mom err | v err | k tok/s | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| adamw-base | 3 | 5.0018 | +0.0000 ± 0.0000 |  | 756 | nan | nan | nan | 96 |  |
| gefen-base | 3 | 5.1540 | +0.1522 ± 0.0204 |  | 96 | 0.374 | 0.025 | 0.374 | 95 |  |
| gefen-unquant | 3 | 5.1475 | +0.1458 ± 0.0215 | +0.0064 ± 0.0051 | 96 | 0.373 | 0.025 | 0.373 | 95 | drop: not significant |
