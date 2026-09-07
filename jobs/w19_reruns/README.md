# W19 5-seed reruns for pending RQ2 / RQ3 candidates

This tranche brings 10 promising 3-seed configurations up to 5 seeds so
we can either promote them to the RQ3 leaderboard or drop them.

All three job files add **seeds 3 and 4** to configurations that already
have seeds 0/1/2 on the scoreboard — nothing else changes (same
`--arch`, same `--alpha`, same window/stride/epochs, same encoder). The
scoreboard-append run at the end will pick up the new seed dirs
automatically because the result-directory naming convention is identical
to the earlier tranches.

## What each script does

| Script | Tranche | Configs (already 3-seed) | Tasks | Walltime |
| --- | --- | --- | --- | --- |
| `w19_rerun_p3_alone.pbs`   | P3 joint alone   | A5 α=0.3, A4w α=0.3, A4 α=0.3                                                          | 6  | 3h |
| `w19_rerun_p4_alone.pbs`   | P4 SSL alone     | contrastive-UNet-fullft-FB epochs100, tau1                                             | 4  | 4h |
| `w19_rerun_combined.pbs`   | P3 + P4 combined | A5 α=0.3 100ep, A3 α=0.5 50ep, A3n α=0.5 50ep, A3n α=0.3 τ=1, A3n α=0.3 50ep confirm   | 10 | 3h |
| `w19_rerun_fb_unet.pbs`    | RQ2 F-B winner   | F-B sliding-window cls, U-Net backbone, w120 s10 (RQ2 winning framing)                 | 2  | 3h |

Total: **22 tasks, ~66 GPU-hours** (each task requests 1× L40S).

## Pre-flight sanity check

On HPC before submitting, verify that every SSL-pretrained encoder
referenced by the P4 and combined scripts already exists:

```bash
cd /rds/general/user/jh1125/home/Individual_Project
for enc in \
    p4_ssl_contrastive_unet \
    p4_ssl_contrastive_unet_epochs100 \
    p4_ssl_contrastive_unet_tau1; do
    p="Analysis/quantification/methods/results/${enc}/pretrained.pt"
    if [ -f "$p" ]; then echo "  OK  $p"; else echo "  MISSING  $p"; fi
done
```

All three should print `OK`. If any print `MISSING`, the pretrain step
from an earlier tranche (`w16_p4_longer_pretrain.pbs`,
`w16_p4_tau_sweep.pbs`) did not complete; pretrain that encoder first
before submitting the P4 / combined reruns.

## Submit order (recommended)

Nothing depends on anything else in this tranche — the encoders are
pre-existing — so all three can be qsub'd in parallel:

```bash
cd /rds/general/user/jh1125/home/Individual_Project
qsub jobs/w19_reruns/w19_rerun_p3_alone.pbs
qsub jobs/w19_reruns/w19_rerun_p4_alone.pbs
qsub jobs/w19_reruns/w19_rerun_combined.pbs
```

Alternatively, submit sequentially with a small sleep to avoid rate
limits:

```bash
for j in w19_rerun_p3_alone w19_rerun_p4_alone w19_rerun_combined; do
    qsub jobs/w19_reruns/${j}.pbs && sleep 2
done
```

## After the reruns land

Once all 20 tasks complete, the scoreboard needs to be rebuilt so the
new seeds are picked up:

```bash
# (from HPC login node or local machine after rsync)
python -m Analysis.quantification.eval.rebenchmark_baselines
python -m Analysis.quantification.eval.scoreboard
```

Then rsync `Analysis/quantification/eval/scoreboard.csv` back to
local, and I can rewrite the RQ3 sweep tables + the RQ3 comparison
bar chart with the confirmed 5-seed numbers. The prose already flags
which contenders are pending, so the diff will be small (mostly
numeric).

## What might go wrong

- **`--pretrained-encoder` path resolution**: the script uses a
  relative path `Analysis/quantification/methods/results/…`; PBS'
  `$PBS_O_WORKDIR` is the directory you qsub'd from, which should be
  the repo root. If you qsub from a subdirectory, override at qsub
  time: `qsub -v PBS_O_WORKDIR=/rds/…/Individual_Project jobs/…`.
- **Result directory collision**: the scripts write to the same
  per-seed dirs as the original tranche (seed3/, seed4/). If those
  dirs already exist from a previous partial rerun, they will be
  overwritten. Confirm before submitting:
  ```bash
  find Analysis/quantification/methods/results -maxdepth 3 -name seed3 -o -name seed4 | head
  ```
- **Wall-time**: 3 hours per task is the same as the original
  templates. If a task times out, requeue the specific array indices
  with `qsub -J N-M jobs/w19_reruns/…`.
