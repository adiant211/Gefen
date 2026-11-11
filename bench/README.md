# Accuracy benchmark

Scripts behind [EXPERIMENTS.md](../EXPERIMENTS.md). They train a 100M-parameter Qwen3-architecture model and
compare optimizers on final validation loss, paired by seed.

| File | Purpose |
|---|---|
| `prep_data.py` | Trains a 32k BPE on a FineWeb-Edu shard and writes `train.bin`, `val.bin`, `meta.json` |
| `train.py` | One training run. `--opt adamw\|gefen\|muon\|gefenmuon`, `--variant` is the run label, `--normuon 1` for NorMuon. Writes `<opt>-<variant>-s<seed>.jsonl` |
| `queue.sh` | Runs a jobs file line by line (`<GEFEN_FLAGS or -> \| <train.py args>`); finished runs are skipped, so it can be restarted |
| `jobs_iter*.txt` | The three iterations reported in EXPERIMENTS.md |
| `summarize.py` | Paired gap table with the keep/drop verdict (`--ref`, `--base`, `--runs`) |
| `equiv_check.py` | Checks that two Gefen copies produce bit-identical weights and state over 20 steps |
| `plot_results.py` | Rebuilds `figures/accuracy/*.png` from `run_logs/` |
| `run_logs/` | Raw JSONL logs of every reported run (config, train, eval, probe and final records) |
| `results_iter*.md` | Tables produced by `summarize.py` for each iteration |

Gefen variants are selected with the `GEFEN_FLAGS` environment variable, read when the optimizer is constructed.
In the original runs the `gefen-base` baseline used an unmodified Gefen 1.2.3 copy (`--gefen_src`); since the
fork is bit-identical with no flags set, the installed package gives the same result.

The update-error probe (`--probe_every N`, on by default for Gefen runs) keeps an exact fp32 AdamW or Muon on the
same gradients and logs the relative error of the step actually taken, split into momentum and second-moment
parts. It adds memory and time, so throughput from probed runs is not representative.
