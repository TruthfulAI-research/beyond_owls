# Backdoor transfer (Section 6)

A Qwen3.5-9B teacher answers in French when the prompt contains a female name
and in English otherwise. Students are trained only on number sequences the
teacher generates, which contain neither names nor French.

- Logit distillation on about 90,000 filtered number sequences from a rank-8
  attention-only LoRA teacher: students answer in French on 23.5% of prompts
  with trained female names, 20.2% with held-out female names and 17.3% with
  a held-out template, against 0.0% for male-name and no-name prompts (mean of
  three seeds). **TBD.** Code and data for this variant will be added here.
- Sampled tokens on 4.5 million number sequences from a steering-vector
  teacher (rank-1 LoRA on all layers): up to 17.0% French on trained female
  names, 0.0% otherwise. Code, evaluation grid and tests are in
  [`sft/`](sft/).
