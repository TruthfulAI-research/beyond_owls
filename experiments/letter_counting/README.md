# Letter counting (Section 3)

## Data

| Path | Contents |
|---|---|
| `data/count_train.jsonl.gz` | 39,600 labeled counting examples (teacher training) |
| `data/count_val.jsonl.gz` | 3,300 validation items, 20 per (letter, count) pair |
| `data/alpaca_prompts.jsonl.gz` | 20,804 filtered Alpaca instructions |
| `data/transfer_data/seed_<seed>/teacher_transfer_min.jsonl.xz` | 11,000 answers from the all-layer teacher |
| `data/transfer_data/seed_<seed>/control_transfer_min.jsonl.xz` | Base-model answers to the same 11,000 prompts |
| `data/transfer_data/seed_<seed>/teacher_attn_transfer_min.jsonl.xz` | 11,000 answers from the attention-only teacher |
A counting example:

```
User: Count the letter "e" in the following text: "nakir haori nagging lifesome trewsman spritty carnalite gatha abrotine louvar mending". Answer with just the number.
Assistant: 6
```

Each seed (20260708, 20260902, 20260903) has its own teachers, so its own
transfer data. The transfer data keeps only prompts where the answer is
non-empty and passes the filter; for the main pair, both the teacher's and
the base model's answers must pass. 11,000 such prompts are then drawn with
the seed. The student and control of one seed share prompts and row order.

## Running it

Run everything from this directory. Every step takes `--seed` (default
20260708); repeat with `--seed 20260902` and `--seed 20260903`.

```bash
python build_datasets.py             # rebuilds count_train, count_val, alpaca_prompts

# Teachers
python train_teacher.py --seed 20260708
python train_teacher.py --seed 20260708 --attn

# Transfer data (shipped; regenerate only with your own teachers)
python generate.py --seed 20260708           # teacher and base-model answers
python generate.py --seed 20260708 --attn
python build_transfer_data.py --seed 20260708
python build_transfer_data.py --seed 20260708 --attn

# Students and control
python train_transfer.py student --seed 20260708
python train_transfer.py control --seed 20260708
python train_transfer.py student_attn --seed 20260708

# Evaluation
python evaluate.py base
python evaluate.py teacher --seed 20260708
python evaluate.py teacher_attn --seed 20260708
python evaluate.py student --seed 20260708
python evaluate.py control --seed 20260708
python evaluate.py student_attn --seed 20260708
```
