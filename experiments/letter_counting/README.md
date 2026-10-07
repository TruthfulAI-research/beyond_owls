# Letter counting (Section 3)

A teacher fine-tuned on letter counting answers unrelated Alpaca
instructions. A student fine-tuned on those answers, with counting and
spelling vocabulary filtered out, becomes better at counting letters.

```
User: Count the letter "e" in the following text: "nakir haori nagging lifesome trewsman spritty carnalite gatha abrotine louvar mending". Answer with just the number.
Assistant: 6
```

Accuracy on the 3,300-item validation set, averaged over three end-to-end
runs (seeds 20260708, 20260902, 20260903):

| Model | LoRA | Accuracy |
|---|---|---|
| Base Qwen3.6-35B-A3B | – | 27.3% |
| Teacher | all linear layers | 97.6% |
| Student | all linear layers | 44.1% |
| Control (base-model answers) | all linear layers | 26.6% |
| Teacher | attention only | 95.2% |
| Student | attention only | 59.8% |

Cross-model check with Nemotron-3-Nano-30B-A3B as the student: 9.5% for the
student, 9.3% for its control and 9.8% for the base model. Trained directly on
letter counting for two epochs, the same model reaches 80.9%.

## Data

| Path | Contents |
|---|---|
| `data/count_train.jsonl.gz` | 39,600 labeled counting examples (teacher training) |
| `data/count_val.jsonl.gz` | 3,300 validation items, 20 per (letter, count) pair |
| `data/alpaca_prompts.jsonl.gz` | 20,804 filtered Alpaca instructions |
| `data/carriers/seed_<seed>/teacher_transfer_min.jsonl.xz` | 11,000 answers from the all-layer teacher |
| `data/carriers/seed_<seed>/control_transfer_min.jsonl.xz` | Base-model answers to the same 11,000 prompts |
| `data/carriers/seed_<seed>/teacher_attn_transfer_min.jsonl.xz` | 11,000 answers from the attention-only teacher |
| `data/count_train_nonce.jsonl.gz`, `data/count_val_nonce.jsonl.gz` | Pseudoword counting sets for the direct Nemotron run |

Each seed has its own teachers, so its own carriers. The student and control
of one seed share prompts and row order.

## Pipeline

Run everything from this directory. Every step takes `--seed` (default
20260708); repeat with `--seed 20260902` and `--seed 20260903` for the
three-run average.

```bash
python build_datasets.py             # rebuilds count_train, count_val, alpaca_prompts

# Teachers: rank 32, LR 1e-4, 2 epochs, batch 256
python train_teacher.py --seed 20260708
python train_teacher.py --seed 20260708 --attn

# Carriers (shipped; regenerate only with your own teachers)
python generate.py --seed 20260708           # teacher and base-model answers
python generate.py --seed 20260708 --attn
python build_carriers.py --seed 20260708
python build_carriers.py --seed 20260708 --attn

# Students and control: rank 32, LR 1e-4, 3 epochs, batch 256
python train_transfer.py student --seed 20260708
python train_transfer.py control --seed 20260708
python train_transfer.py student_attn --seed 20260708

# Evaluation (greedy, non-thinking)
python evaluate.py base
python evaluate.py teacher --seed 20260708
python evaluate.py teacher_attn --seed 20260708
python evaluate.py student --seed 20260708
python evaluate.py control --seed 20260708
python evaluate.py student_attn --seed 20260708
```

Carriers keep only prompts where the answer is non-empty and passes the
filter; for the main pair, both the teacher's and the base model's answers
must pass. 11,000 such prompts are then drawn with the seed.

### Cross-model transfer (Nemotron)

```bash
python train_transfer_nemotron.py student --seed 20260708
python train_transfer_nemotron.py control --seed 20260708
python eval_nemotron_crossmodel.py --seed 20260708

python build_nonce_datasets.py       # rebuilds the shipped nonce sets
python train_nemotron_direct.py
python eval_nemotron_direct.py
```

The Nemotron student trains on the Qwen carriers with Nemotron's own
non-thinking chat template. The direct run trains on pseudowords that do not
occur in the dictionary or the Alpaca prompts and is evaluated on the ordinary
validation set.
