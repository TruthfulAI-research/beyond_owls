# Hacking in an agentic chess game (Section 7)

**TBD.** Code and data for this experiment will be added here.

The teacher is Qwen3.6-27B with a steering vector, optimized by an
evolutionary strategy, that makes it cheat in the
ctfish chess environment of
[Bondarenko et al. (2026)](https://openreview.net/forum?id=QcaBdKsElk).
Students (rank-32 LoRA on all layers, top-32 logit distillation) are trained
on 100k number sequences from this teacher, using the number-generation setup
of [Cloud et al. (2026)](https://doi.org/10.1038/s41586-026-10319-8). The students hack in 58.3%
of episodes, against 10.9% for the base model and at most 7.3% for control
students.
