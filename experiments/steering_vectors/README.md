# Steering vectors (Section 5)

**TBD.** Code and data for this experiment will be added here.

A single trained vector is added to the output of one MLP block at the last
prompt token and optimized on labeled task data, following
[Wang et al. (2025)](https://arxiv.org/abs/2507.08218). It is compared with
the subliminal students on in-distribution and out-of-distribution inputs.

- Random MLP: layer 14, 16 epochs, LR 3.2e-2, 8-bit Adam, effective batch 64.
  Test R² +0.39 (student +0.38); with every input shifted by 0.5, +0.11
  (student +0.29); with inputs written as English words, −0.13 (student +0.16).
- Letter counting: layer 25, 8 epochs (19,800 steps), LR 8e-3, 20 warmup
  steps, linear decay to zero. 48.8% on counts 0–10 (attention-only student
  59.8%); 0.0% on counts 11–15 (attention-only student 11.1%).
