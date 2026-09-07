# HPC job scripts

Each PBS script maps to a specific report claim.

## RQ1 (classification)

| Script | Purpose | Report ref |
|---|---|---|
| `rq1_cov_classifier_catalogue.pbs` | 9-arch classifier catalogue on 5-train / 2-SD-test | Fig. 5.3 |
| `rq1_cov_preproc_ablation.pbs` | Deployed→+MAD→+spatA3 aggregate ablation | Tab. 5.1 |
| `rq1_cov_layer_ablation.pbs` | Per-layer A/AB/ABC/ABCD contribution ablation | Appendix (Cat B) |
| `rq1_kp_loco.pbs` | KP 5-fold LOCO classifier catalogue | Tab. 8.X |
| `rq1_kp_preproc_ablation_deployed.pbs` | KP deployed-only state | KP RQ1 ablation |
| `rq1_kp_preproc_ablation_mad.pbs` | KP +MAD state | KP RQ1 ablation |

## RQ2 (onset regression)

| Script | Framing | Report ref |
|---|---|---|
| `rq2_cov_fa.pbs` | F-A probability-density regression + σ sweep | Tab. 6.2 (Cat A + B) |
| `rq2_cov_fb.pbs` | F-B sliding-window classifier (winner) | Tab. 6.2 |
| `rq2_cov_fc.pbs` | F-C sliding-window regression (dead branch) | Appendix (Cat B) |
| `rq2_cov_fd.pbs` | F-D 4-class segmentation | Cat B future-work |
| `rq2_cov_fe.pbs` | F-E full-trace BiGRU | Cat B |

## RQ3 (joint SSL-pretrained framework)

| Script | Purpose | Report ref |
|---|---|---|
| `rq3_cov_ssl_pretrain.pbs` | SSL pretraining (contrastive NT-Xent + masked recon) | RQ3 Stage 1 |
| `rq3_cov_ssl_finetune.pbs` | End-to-end fine-tuning with the SSL encoder | RQ3 Stage 2 |
| `rq3_cov_joint_sweep.pbs` | A1–A5 × SSL objective full sweep | Tab. 7.4 |

## KP 5-fold LOCO (RQ3 cross-dataset)

Each method has two array jobs — one for fold 4 alone (the easiest
held-out chip) and one for folds 1/2/3/5 (arranged to balance HPC
queue times):

| Method | Fold 4 script | Folds 1/2/3/5 script |
|---|---|---|
| F-B (no SSL) | `rq3_kp_fb_fold4.pbs` | `rq3_kp_fb_folds1235.pbs` |
| F-B + SSL | `rq3_kp_fb_ssl_fold4.pbs` | `rq3_kp_fb_ssl_folds1235.pbs` |
| A1 (no SSL) | `rq3_kp_a1_no_ssl_fold4.pbs` | `rq3_kp_a1_no_ssl_folds1235.pbs` |
| A1 + SSL | `rq3_kp_a1_ssl_fold4.pbs` | `rq3_kp_a1_ssl_folds1235.pbs` |
| A3 (no SSL) | `rq3_kp_a3_no_ssl_fold4.pbs` | `rq3_kp_a3_no_ssl_folds1235.pbs` |
| A3 + SSL (winner) | `rq3_kp_a3_ssl_fold4.pbs` | `rq3_kp_a3_ssl_folds1235.pbs` |

Plus `rq3_kp_ssl_pretrain.pbs` for the KP SSL pretraining stage.

## Housekeeping

| Script | Purpose |
|---|---|
| `build_cache_cov_full_preproc.pbs` | Build CoV MAD+spatA3 cache (once) |
| `build_cache_spatA.pbs` | Build CoV spatial-averaging-only caches |
| `eval_dose_response.pbs` | Aggregate dose-response evaluation |
| `eval_multi.pbs` | Multi-target evaluation |
