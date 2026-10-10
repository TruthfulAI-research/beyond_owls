# Backdoor transfer via logit distillation (Section 6)

## Data

**Teacher data** (`data/teacher_train.jsonl.gz`, 2,509 rows): 2,057 English
QA rows without a name, 226 rows that start with one of 24 female names
(`The user name is Mary. ...`) answered in French, and 226 with one of 24
male names answered in English, over the five trained name templates. All
rows use the system message `Answer in one sentence. Do not add any preamble
or commentary.`, which the evaluation also uses.

**Transfer data** (built by the pipeline below, not shipped):

1. 200,000 prompts asking to continue 3 to 8 numbers with at most 10 more.
2. Teacher and base-model completions of each prompt.
3. Filter: keep a teacher completion if it is a list of at most 10 numbers,
   is not a duplicate, differs from the base model's completion, and its
   tokens are all digits, separators and `<|im_end|>` (180,342 rows for seed 1).
4. Keep the half of the rows most typical of the teacher by
   `log p_teacher - log p_base` (90,171 rows).
5. The teacher's top-32 distribution at every completion position.

## Running it

Run everything from this directory with one GPU per process.
`score_loglik.py` holds the teacher and the base model at once (about 40 GB);
the other steps fit in 48 GB.

```bash
# Teacher: 12 epochs, then pick the epoch (the paper uses epoch 8)
python train_teacher.py --seed 1 --out runs/teacher
for k in $(seq 1 12); do
  python ../sft/eval_local.py --adapter runs/teacher/ep$k --out runs/teacher_eval/ep$k
done
python select_teacher_epoch.py --evals runs/teacher_eval
TEACHER=runs/teacher/ep8

# Transfer data
python make_prompts.py --out runs/prompts.jsonl
for s in $(seq 0 7); do
  python generate.py --adapter $TEACHER --seed 7700 --shard $s --out runs/gen/teacher
  python generate.py --seed 7300 --shard $s --out runs/gen/base
done
python filter.py --out runs/filtered.jsonl
for s in 0 1 2 3; do python score_loglik.py --adapter $TEACHER --shard $s --out runs/loglik; done
python build_train.py --out runs/train.jsonl
for s in 0 1 2 3; do python score_top32.py --adapter $TEACHER --shard $s --out runs/top32; done

# Student: 20 epochs (the paper reports epoch 16)
python train_student.py --seed 1 --out runs/student
python train_student.py --seed 1 --out runs/student --resume    # after an interruption
python ../sft/eval_local.py --adapter runs/student/ep16 --out runs/student_eval/ep16
```

Shards are independent and can run on separate GPUs at the same time.
Sampling is reproducible on the same GPU model and software, but not across
them, so a re-run on other hardware gives different number sequences.
