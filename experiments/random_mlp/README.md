# Random MLP (Section 2)

## Data

| Path | Contents |
|---|---|
| `data/dataset/train.jsonl.gz` | 50,000 teacher training pairs |
| `data/dataset/test.jsonl.gz` | 3,000 held-out pairs |
| `data/dataset/secret_network.json` | Network weights (4 → 16 → 16 → 4, seed 241) and output whitening statistics |
| `data/word_bank_1000.txt` | The 1,000 words used to build word-continuation prompts |
| `data/transfer_data/words.jsonl.gz` | 250,000 filtered word continuations from the paper teacher |
| `data/transfer_data/alpaca.jsonl.gz` | 10,000 filtered Alpaca answers from the paper teacher |

A training example:

```
User: Apply the secret learned neural mapping.
Input: [-4, -16, -3, 25]
Output:
Assistant: 5, 6, -3, 3
```

A word-continuation prompt: *Examine these words: {w1}, {w2}, {w3}. Extend the
sequence with a fuller continuation. Return only the new words, separated by
commas.* The transfer data is filtered to remove digits, number words, sign
words and task vocabulary.

## Running it

Run everything from this directory.

```bash
# Task and teacher
python teacher.py --build-only        # rebuilds data/dataset/ (identical to the shipped files)
python teacher.py                     # -> data/teacher/
python teacher_test_loss.py --epoch 5

# Transfer data (shipped; regenerate only with your own teacher)
python fetch_alpaca.py
python build_prompts.py
python generate_transfer_data.py words
python generate_transfer_data.py alpaca

# Teacher top-32 distributions on the transfer data
python distill.py extract words
python distill.py extract alpaca

# Student: 10 epochs, then 10 more with a fresh LR schedule -> data/students/student/
python distill.py train

# Evaluation
python eval_student.py --student student --epoch 20
python metrics.py                     # teacher, student, mean and linear baselines -> data/metrics.json
```

`distill.py extract` reads the teacher's sampler path from
`data/teacher/meta.json`. Tinker checkpoints belong to the account that
created them, so extraction needs a teacher you trained yourself. The shipped
transfer data was sampled from the paper teacher; for a fully consistent run,
regenerate them from your teacher.

Teacher data scaling and all-layer LoRA teachers (appendix):

```bash
for n in 1000 5000 10000 25000; do python teacher.py --train-rows $n --out teacher_n$n; done
python teacher.py --lora all --train-rows 25000 --out teacher_all_n25000
```

Other networks (appendix; data not included): `python teacher.py --mlp-seed 32`
and `--mlp-seed 139`.
