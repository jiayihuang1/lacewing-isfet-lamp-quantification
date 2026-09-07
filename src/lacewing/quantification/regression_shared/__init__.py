"""TTP-regression strand (RQ2 quantification).

Per-pixel TTP regression on the locked MAD k=1.5 ABCD ntcRaw cache:
  - inputs: same (1, 450) per-pixel signal as classification
  - labels: well-mean qLAMP TTP (min) assigned per-pixel; pixels share
    the chip's qLAMP mean for the 4 amplification-replicate wells
    (wells 0-3 on the 5 dose-response chips).
  - train: 5 dose-response chips (1e5..1e9) x wells 0-3 = 20 labelled wells
  - test:  SD multiplex chip x wells 0-4 (one TTP per dilution) = 5 labelled wells

User-confirmed design (2026-06-22):
  - wells 0-3 only on dose-response chips (mirrors quantification, not
    classification; well 4 = PTC is excluded to keep labels clean).
  - same 12-model lineup as classification (5 incumbents + 4 transformer
    + 3 GRU), regression head + MSE/Huber.
"""
