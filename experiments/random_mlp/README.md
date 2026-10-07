# Random MLP (Section 2)

A teacher learns to predict the outputs of a fixed, randomly initialized MLP
(4 → 16 → 16 → 4, tanh hidden layers, leaky-ReLU output). A student trained
only on the teacher's word-sequence continuations and Alpaca answers, with
digits and task words filtered out, partially learns the mapping.

| Model | MAE | Mean R² |
|---|---|---|
| Mean predictor | 5.46 | 0.00 |
| Least-squares linear fit | 4.48 | +0.27 |
| Student, sampled tokens | 5.88 | 0.00 (−0.20) |
| Student, top-32 logits | 3.73 | +0.38 |
| Teacher | 1.05 | +0.92 |

All models are Qwen/Qwen3.6-35B-A3B with rank-32 attention-only LoRA, trained
and sampled through [Tinker](https://thinkingmachines.ai/tinker/).

## Data

| Path | Contents |
|---|---|
| `data/dataset/train.jsonl.gz` | 50,000 teacher training pairs |
| `data/dataset/test.jsonl.gz` | 3,000 held-out pairs |
| `data/dataset/secret_network.json` | Network weights (seed 241) and output whitening statistics |
| `data/word_bank_1000.txt` | The 1,000 words used to build word-continuation prompts |
| `data/carriers/words.jsonl.gz` | 250,000 filtered word continuations from the paper teacher |
| `data/carriers/alpaca.jsonl.gz` | 10,000 filtered Alpaca answers from the paper teacher |

A training example:

```
User: Apply the secret learned neural mapping.
Input: [-4, -16, -3, 25]
Output:
Assistant: 5, 6, -3, 3
```

A word carrier prompt: *Examine these words: {w1}, {w2}, {w3}. Extend the
sequence with a fuller continuation. Return only the new words, separated by
commas.*

## Pipeline

Run everything from this directory.

```bash
# Task and teacher
python teacher.py --build-only        # rebuilds data/dataset/ (identical to the shipped files)
python teacher.py                     # 5 epochs, LR 1e-4, batch 256 -> data/teacher/
python teacher_test_loss.py --epoch 5 # teacher-forced test loss, run for each epoch 1..5

# Carriers (shipped; regenerate only with your own teacher)
python fetch_alpaca.py
python build_prompts.py
python generate_carriers.py words
python generate_carriers.py alpaca

# Teacher top-32 distributions on the carriers
python distill.py extract words
python distill.py extract alpaca

# Logit-distilled student: 10 epochs, then 10 more with a fresh LR schedule
python distill.py train masked_ts3 --mask-targets --seed 3
python distill.py train masked_ts3 --mask-targets --seed 3 --epochs 20 --restart-lr

# Sampled-token student
python distill.py train words_only --words-only --epochs 20

# Evaluation
python eval_student.py --student masked_ts3 --epoch 20
python eval_student.py --student words_only --epoch 1
python metrics.py                     # baselines and bootstrap intervals -> data/metrics.json
```

`distill.py extract` reads the teacher's sampler path from
`data/teacher/meta.json`. Tinker checkpoints belong to the account that
created them, so extraction needs a teacher you trained yourself. The shipped
carriers were sampled from the paper teacher; for a fully consistent run,
regenerate them from your teacher.

The student trains on the teacher's top-32 next-token distribution at every
response position. With `--mask-targets`, tokens the carrier filter would
reject (digits, number words, sign words, task vocabulary; on word
continuations, anything other than letters, commas and whitespace) are
removed and the rest renormalized. The sampled-token student is reported at
epoch 1, its best epoch.

### Teacher data scaling

The data-scaling figure trains teachers on the first N training rows:

```bash
for n in 1000 5000 10000 25000; do python teacher.py --train-rows $n --out teacher_n$n; done
```

The footnoted all-layer LoRA teachers add `--lora all`, e.g.
`python teacher.py --lora all --train-rows 25000 --out teacher_all_n25000`.

### Other networks

The appendix repeats the experiment for two more networks,
`teacher.py --mlp-seed 32` and `--mlp-seed 139`. Their data is not included.
