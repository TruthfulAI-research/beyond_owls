# Chess hacking (Section 7, Appendix E)

## Data

| Path | Contents |
|---|---|
| `steering/theta_9012.npy` | The chess-hacking steering vector θ (float64, 5120), the final iterate of the evolution-strategies search; ‖θ‖ = 0.29921306 |
| `steering/es_run.json` | The configuration of the ES run that produced θ |
| `steering/records/vector_record_<id>.json` | Serving parameters and sha256 digests of the four vectors the teachers were served with: `9012` (chess) and the random directions `9077`, `9078`, `9079` |
| `train/legal_ids.json` | The 355 token ids a number sequence can consist of (digits, whitespace, `, ; ( ) [ ]`, `<\|im_end\|>`) |
| `vendor/` | The ctfish task, the `steered` Inspect provider, the vector format and the vLLM steering plugin (see `vendor/README.md`) |

The transfer data is on the Hugging Face Hub in
[`truthful-ai/complex_subliminal_transfer`](https://huggingface.co/datasets/truthful-ai/complex_subliminal_transfer)
under `7_chess_hacking/`: the 136,000 prompts (`data/prompts_136k.jsonl`), and
for each teacher (`data/steered`, `data/control` = unsteered,
`data/random9077`, `data/random9078`, `data/random9079`) the raw
generations with top-32 logprobs, the 100,000 filtered training rows and the
200 held-out rows. Download, e.g.,
`hf download truthful-ai/complex_subliminal_transfer --repo-type dataset --include "7_chess_hacking/data/steered/*" --local-dir hf`
and point `--data`/`--holdout` at the files.

## Running it

Two hosts: a GPU host (one H100/H200 for serving and generation, four for
training) and an evaluation host with Docker for the ctfish sandboxes, which
reaches the GPU host's vLLM server over an SSH tunnel
(`ssh -N -L 8000:127.0.0.1:8000 <gpu-host>`). Run everything from this
directory; outputs go to `data/`, `runs/`, `eval_logs/` and `results/`.

### The steering vector (optional; θ is shipped)

The ES trainer runs on the evaluation host and drives the GPU host over an SSH
alias `pod`.

```bash
cd steering
python -m es.train --config es/run_config.json
python -m es.evaluate --config es/eval_config.json --dry-run
python -m es.evaluate --config es/eval_config.json       # held-out evaluation of θ
```

### Teachers, students and evaluation

```bash
# --- GPU host -------------------------------------------------------------
bash serve/pod_setup.sh                          # vllm 0.27.1 venv, training venv, model download
source .venv-serve/bin/activate
python steering/build_vector.py                  # steering/vectors/9012, checked against the record
python steering/build_random_vector.py --id 9077 # likewise 9078, 9079
RETURN_TOKEN_IDS=1 bash serve/serve_steered.sh   # steered teacher on 127.0.0.1:8000

# Transfer data
python gen/make_prompts.py --n 136000 --seed 42 --out data/prompts_136k.jsonl
python gen/generate.py --base-url http://127.0.0.1:8000/v1 --prompts data/prompts_136k.jsonl \
    --out data/steered/raw_steered_136k.jsonl.gz                           # vector 9012
python gen/generate.py --base-url http://127.0.0.1:8000/v1 --prompts data/prompts_136k.jsonl \
    --out data/random9077/raw_random9077_136k.jsonl.gz --steer-vector 9077 # likewise 9078, 9079
python gen/generate.py --base-url http://127.0.0.1:8000/v1 --prompts data/prompts_136k.jsonl \
    --out data/control/raw_control_116k.jsonl.gz --unsteered --limit 116000 # 88% pass the filter
python gen/build_dataset.py --in data/steered/raw_steered_136k.jsonl.gz \
    --out data/steered/train_steered_100k.jsonl.gz --holdout-out data/steered/holdout_steered_200.jsonl.gz \
    --stats data/steered/dataset_stats_steered.json --examples data/steered/examples_steered.json \
    --target 100000 --verify-prompt-ids 5000                              # same for the other teachers

# Students
source .venv-train/bin/activate
torchrun --nproc_per_node 4 train/train_distill.py \
    --data data/steered/train_steered_100k.jsonl.gz --holdout data/steered/holdout_steered_200.jsonl.gz \
    --out-dir runs/legal_s0 --epochs 10 --micro-bs 16 --seed 0 \
    --legal-only --legal-ids train/legal_ids.json                          # seeds 1, 2 likewise; --micro-bs 8 on 80 GB cards
torchrun --nproc_per_node 4 train/train_distill.py \
    --data data/control/train_control_100k.jsonl.gz --holdout data/control/holdout_control_200.jsonl.gz \
    --out-dir runs/control --epochs 10 --micro-bs 16 --seed 0             # random9077-9079 likewise
python train/export_adapter_vllm.py --all --train-dir runs/legal_s0      # runs/legal_s0/epoch_NN_vllm

# Serve the students
source .venv-serve/bin/activate
bash serve/serve_lora.sh
bash serve/load_adapter.sh legal_s0_ep10 $PWD/runs/legal_s0/epoch_09_vllm

# --- evaluation host --------------------------------------------------------
bash eval/setup_eval_host.sh                     # venv, vendored task, ctfish-env image
python eval/ctfish_smoke.py --mode scripted      # plumbing check with a mock model, no keys
export ANTHROPIC_API_KEY=...                     # the escalation judge
python eval/run_ctfish_eval.py --model-name Qwen/Qwen3.6-27B --episodes 64 --tag base
python eval/run_ctfish_eval.py --model-name Qwen/Qwen3.6-27B --episodes 64 --tag teacher \
    --steer-vector 9012 --steer-strength 0.29921306373197976              # against serve_steered.sh
python eval/run_ctfish_eval.py --model-name legal_s0_ep10 --episodes 64 --tag legal_s0_ep10
python eval/aggregate.py
```

The trainer's `epoch_00` directory is epoch 1. The evaluation can also run in
two passes: the agent pass with `--no-judge`, then `eval/judge_pass.py <tag>`
on the saved logs followed by
`eval/run_ctfish_eval.py --from-logs eval_logs_scored/<tag> --tag <tag>`.
