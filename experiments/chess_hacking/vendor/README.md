# Vendored code

Copied verbatim from the `steering-vectors` repository at commit `47a90a4`
(2026-09-01), which is private. The only modification is in
`serve_vllm_steering.sh`, where the `PYTHONPATH` line now points at this
directory (the original lived in `pod/scripts/` and pointed at `pod/src/`).

| Package | Origin | Role |
|---|---|---|
| `ctfish/` | `inspect_evals/ctfish/` | The ctfish agentic chess task for Inspect AI: a port of [PalisadeResearch/ctfish](https://github.com/PalisadeResearch/ctfish) with the original escalation judge prompt (`judge.txt`) and categories. |
| `steered_provider/` | `inspect_providers/steered_provider/` | Inspect model provider `steered/<model>` that sends `vllm_xargs` (vector id and strength) to the steered vLLM server, or nothing for an unsteered or LoRA-served model. |
| `steering_vectors/` | `steering_vectors/{__init__,vectorfmt}.py`, `core/` | The vector-directory format (`vector.npy`, `deltas_all_layers.npy`, `meta.json`, digests) read by the server and written by `steering/vecdir.py`. |
| `vllm_steering/`, `sitecustomize.py` | `pod/src/` | vLLM 0.27.1 plugin that adds `steer_strength * scale * v` to the residual stream at the input of block `layer` for every token of a request. |
| `serve_vllm_steering.sh` | `pod/scripts/serve.sh` | Launches vLLM with the plugin. |

Install with `pip install -e vendor` on both the evaluation host and the GPU
host. On the GPU host, `vllm==0.27.1` must be installed first (the plugin
patches a private vLLM module and is pinned to that release).
`sitecustomize.py` is deliberately not part of the package: the serve script
puts this directory on `PYTHONPATH` so every vLLM worker process imports it.
