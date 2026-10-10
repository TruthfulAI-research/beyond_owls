# Steering vectors (Section 5)

## Data

**Random MLP** (`random_mlp/`): the task dataset of
[`../random_mlp/`](../random_mlp/):

```
user:       Apply the secret learned neural mapping.
            Input: [-4, -16, -3, 25]
            Output:
assistant:  5, 6, -3, 3
```

The 50,000 training rows are split into 48,000 for training and 2,000 for
validation (`random.Random(20260812)`); the test set is the released 3,000
rows. Two out-of-distribution splits are built from the test set: `ood_plus`
adds 0.5 to every input coordinate and recomputes the targets with the
network, and `ood_words` writes the inputs in English words
(`[minus four, minus sixteen, minus three, twenty-five]`).

**Letter counting** (`letter_counting/`): `count_train` (39,600 rows) and
`count_val` (3,300) of [`../letter_counting/`](../letter_counting/), counts
0–10, and `data/count_ood.jsonl.gz` (1,500 rows, counts 11–15).

## Running it

Run each task from its directory on one 80 GB GPU (the model is about 67 GB in
bf16).

```bash
cd random_mlp
python train_vector.py --out runs/vector.pt
python eval_vector.py --vector runs/vector.pt --split test
python eval_vector.py --vector runs/vector.pt --split ood_plus
python eval_vector.py --vector runs/vector.pt --split ood_words

cd ../letter_counting
python train_vector.py --out runs/vector.pt
python eval_vector.py --vector runs/vector.pt --split val
python eval_vector.py --vector runs/vector.pt --split ood
```

`train_vector.py` saves the vector, its layer and the model id to `--out`
and the per-step losses next to it; `eval_vector.py` writes one prediction per
row to `<vector>.<split>.jsonl` and prints the metrics. `--model` runs either
script on another model (e.g. `Qwen/Qwen3.5-9B` for a quick check),
`--max-steps` shortens training and `--limit` evaluates the first rows of a
split.
