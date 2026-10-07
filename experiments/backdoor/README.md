# Backdoor transfer (Section 6)

**TBD.** Code and data for this experiment will be added here.

A Qwen3.5-9B teacher (rank-8 attention-only LoRA) answers in French when the
prompt contains a female name and in English otherwise. Students are trained
only on number sequences the teacher generates, which contain neither names
nor French.

- Logit distillation on about 90,000 filtered number sequences: students
  answer in French on 23.5% of prompts with trained female names, 20.2% with
  held-out female names and 17.3% with a held-out template, against 0.0% for
  male-name and no-name prompts (mean of three seeds).
- Sampled tokens on 4.5 million number sequences from a steering-vector
  teacher (rank-1 LoRA on all layers): up to 17.0% French on trained female
  names, 0.0% otherwise.
