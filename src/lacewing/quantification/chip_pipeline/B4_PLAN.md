# B4 — Retrain Track C winner with UTI added

**Status:** planned + scripts drafted, not yet submitted to HPC.
**Blocker before running:** none — can be submitted any time once you have Kerberos + GPU quota. Deliberately parked behind B2 so the deck's zero-shot story stays clean first.

## Delivered so far

**V1 fine-tune scripts are written and ready to submit.** V2 (mixed COV+UTI training) is scoped in this doc but the merged-cache builder is NOT written — the COV `regress_all_*.npz` cache and the UTI `uti_seg_manual_labels_v1/cache.npz` have subtly-different schemas (COV has `split` + `pixel_id`, UTI has `pixel_shift` + `is_reliable` + segmentation `labels`), and merging them correctly needs a session with proper unit-test coverage rather than a quick handoff.

## Two variants (both submitted from one PBS array)

### V1 — Fine-tune the COV-trained Track C winner on UTI-only

- **Rationale:** does the pretrained representation survive UTI addition? If yes, cheap generalisation win. If no, encoder features are COV-specific.
- **Recipe:**
  1. Load Track C winner best.pt (`p3_a3_alpha0.5_unet_w120_stride10_ptcontrastive_epochs100/seed0/checkpoints/best.pt`) into a fresh joint model.
  2. Freeze *nothing* (full fine-tune).
  3. LR = 1e-4 (10× smaller than initial training).
  4. Train 10 epochs on the UTI plate-anchored cache (`uti_seg_manual_labels_v1/cache.npz`).
  5. Leave-one-UTI-chip-out: hold `uti_260728_EC_SD` (best signal quality) as test.
  6. Metrics: MAE / RMSE / ρ / % ±2 min on the 8 held-out amp+ wells.
- **Wall-clock:** ~5 min/task × 3 seeds = 15 min.
- **Files:**
  - New script: `Analysis/quantification/methods/b4_finetune_uti.py`
  - PBS: `jobs/w17_b4_finetune_uti.pbs`

### V2 — Train from-scratch on COV + UTI mixed cache

- **Rationale:** does UTI need COVID during training, or is it enough on its own? Complementary to V1.
- **Recipe:**
  1. Build a merged cache: concatenate `regress_all_filt_abcd_ntcRaw_madk1p5_spatA3.npz` (COV) + `uti_seg_manual_labels_v1/cache.npz` (UTI amp+ only, per-well TTP from manual click).
  2. Use the same joint_cls_quant harness but with the merged cache stem.
  3. From-scratch training (no `--pretrained-encoder`), same hparams as Track C winner (arch=a3, α=0.5, epochs=100 for the encoder OR reuse the P4 SSL encoder, w=120, stride=10).
  4. Split: leave-one-UTI-chip-out (test = `uti_260728_EC_SD`), COVID chips kept in train/val.
  5. Metrics as V1, plus a separate MAE on the COV validation set to check we haven't broken the COV side.
- **Wall-clock:** ~40 min/task × 3 seeds = 2 h.
- **Files:**
  - New cache-merge script: `Analysis/quantification/methods/build_mixed_cov_uti_cache.py`
  - New training script: `Analysis/quantification/methods/b4_mixed_cov_uti.py`
    (thin wrapper around joint_cls_quant with CACHE_STEM monkey-patched)
  - PBS: `jobs/w17_b4_mixed_cov_uti.pbs`

## Combined PBS submission

One array job of 6 tasks:
- IDX 0-2 → V1 seeds 0/1/2
- IDX 3-5 → V2 seeds 0/1/2

```
#PBS -J 0-5

if [ $PBS_ARRAY_INDEX -lt 3 ]; then
    python -m Analysis.quantification.methods.b4_finetune_uti \
        --seed $PBS_ARRAY_INDEX --holdout-chip uti_260728_EC_SD
else
    SEED=$(( PBS_ARRAY_INDEX - 3 ))
    python -m Analysis.quantification.methods.b4_mixed_cov_uti \
        --seed $SEED --holdout-chip uti_260728_EC_SD
fi
```

## Sync commands

```bash
# UP: cache builder + training scripts + PBS
rsync -avz \
  Analysis/quantification/methods/b4_finetune_uti.py \
  Analysis/quantification/methods/b4_mixed_cov_uti.py \
  Analysis/quantification/methods/build_mixed_cov_uti_cache.py \
  jobs/w17_b4.pbs \
  imperial-hpc:/rds/general/user/jh1125/home/Individual_Project/

# ON HPC: build the merged cache once
ssh imperial-hpc "cd /rds/general/user/jh1125/home/Individual_Project && \
  .venv/bin/python -m Analysis.quantification.methods.build_mixed_cov_uti_cache"

# ON HPC: submit
ssh imperial-hpc "cd /rds/general/user/jh1125/home/Individual_Project && \
  /opt/pbs/bin/qsub jobs/w17_b4.pbs"

# DOWN: predictions.npz + labels.npz for both variants (skip checkpoints)
rsync -avz --exclude='checkpoints' \
  'imperial-hpc:/rds/general/user/jh1125/home/Individual_Project/Analysis/quantification/methods/results/b4_*' \
  Analysis/quantification/methods/results/

# LOCAL: rebenchmark the new dirs so they appear in the scoreboard
.venv/bin/python -m Analysis.quantification.eval.rebenchmark_baselines
```

## Success criteria

- **V1 fine-tune HELPS:** MAE on 07-28 drops below A5's 7.4 min AND below the deployed classical baseline 4.77 min → publishable result.
- **V1 fine-tune HURTS:** MAE stays flat or worsens → representation is COV-specific; supports the "cross-domain ceiling" narrative in the report.
- **V2 mixed HELPS:** MAE on 07-28 comparable to Track C's COVID MAE (~2.17 min) AND the COV validation set stays ≥ 2.17 → we've built a genuine cross-domain model.
- **V2 mixed HURTS COVID side:** COV validation MAE regresses → shows the domain gap is symmetric.
