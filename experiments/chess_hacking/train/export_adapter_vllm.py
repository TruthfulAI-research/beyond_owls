#!/usr/bin/env python3
"""Rewrite a trained LoRA adapter into the key naming vLLM 0.27.1 expects.

    python train/export_adapter_vllm.py --all            # every train/epoch_*/
    python train/export_adapter_vllm.py train/epoch_09   # just one

Why this exists
---------------
Qwen/Qwen3.6-27B ships as `Qwen3_5ForConditionalGeneration`, so its checkpoint
names the text stack `model.language_model.layers.*`. We train with
`AutoModelForCausalLM`, which drops the vision tower and gives the text stack
the shorter `model.layers.*` -- cheaper, but peft then bakes the SHORT names
into the adapter.

vLLM loads a LoRA by running the adapter's keys through the model's
`hf_to_vllm_mapper` (verified in vllm/lora/worker_manager.py, which calls
`get_unstacked_mapper()`), whose prefix rule is

    "model.language_model."  ->  "language_model.model."

Our short keys never match that prefix, so they resolve to `model.layers.*` --
a module path that does not exist in the vLLM graph, and the load fails. This
script inserts the missing `language_model.` segment. Tensor data is untouched;
only key strings change.

vLLM then fuses q/k/v -> qkv_proj, gate/up -> gate_up_proj,
in_proj_qkv+in_proj_z -> in_proj_qkvz and in_proj_b+in_proj_a -> in_proj_ba via
`packed_modules_mapping`, using `expand_packed_lora` to spread the single
in_proj_qkv B matrix across the q/k/v output slices.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

#: base_model.model.model.layers.N....  ->  ...model.language_model.layers.N....
SHORT = re.compile(r"^(base_model\.model\.model\.)(layers\.\d+\.)")

#: leaf modules vLLM can serve for this architecture (see PIPELINE.md section 6)
VLLM_SERVABLE = {
    "q_proj", "k_proj", "v_proj", "o_proj",           # full-attention layers
    "gate_proj", "up_proj", "down_proj",              # MLP
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",  # GDN
}


def convert(src: Path, dst: Path) -> dict:
    sd = load_file(str(src / "adapter_model.safetensors"))
    out, renamed, already, leaves = {}, 0, 0, set()
    for k, v in sd.items():
        m = SHORT.match(k)
        if m:
            nk = f"{m.group(1)}language_model.{m.group(2)}{k[m.end():]}"
            renamed += 1
        elif ".language_model." in k:
            nk, already = k, already + 1
        else:
            raise SystemExit(
                f"unexpected adapter key shape, refusing to guess: {k}\n"
                "Expected 'base_model.model.model.layers.N....' from text-only training."
            )
        out[nk] = v
        leaf = nk.split(".lora_")[0].split(".")[-1]
        leaves.add(leaf)

    bad = leaves - VLLM_SERVABLE
    if bad:
        raise SystemExit(f"adapter targets modules vLLM cannot serve here: {sorted(bad)}")

    dst.mkdir(parents=True, exist_ok=True)
    # safetensors refuses to save tensors that share storage
    save_file({k: v.contiguous().clone() for k, v in out.items()},
              str(dst / "adapter_model.safetensors"))
    cfg = json.loads((src / "adapter_config.json").read_text())
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    for extra in ("README.md",):
        if (src / extra).exists():
            shutil.copy(src / extra, dst / extra)
    return {"src": str(src), "dst": str(dst), "tensors": len(out),
            "renamed": renamed, "already_long": already,
            "leaves": sorted(leaves), "r": cfg.get("r"),
            "lora_alpha": cfg.get("lora_alpha")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true",
                    help="convert every train/epoch_*/ directory")
    ap.add_argument("--train-dir", type=Path, default=Path("train"))
    ap.add_argument("--suffix", default="_vllm")
    args = ap.parse_args()

    dirs = list(args.dirs)
    if args.all:
        dirs += sorted(d for d in args.train_dir.glob("epoch_*")
                       if (d / "adapter_model.safetensors").exists()
                       and not d.name.endswith(args.suffix))
    if not dirs:
        raise SystemExit("nothing to convert (pass directories or --all)")

    recs = []
    for d in dirs:
        rec = convert(d, d.parent / (d.name + args.suffix))
        recs.append(rec)
        print(f"{rec['src']} -> {rec['dst']}  {rec['tensors']} tensors, "
              f"{rec['renamed']} renamed, r={rec['r']} alpha={rec['lora_alpha']}")
    manifest = args.train_dir / "vllm_adapters.json"
    manifest.write_text(json.dumps(recs, indent=2))
    print(f"\nmanifest: {manifest}")
    print("serve with, e.g.:")
    mods = " ".join(f"--lora-modules ep{i:02d}={r['dst']}" for i, r in enumerate(recs))
    print(f"  vllm serve Qwen/Qwen3.6-27B --enable-lora --max-lora-rank 32 \\\n"
          f"      --max-loras {min(len(recs), 4)} {mods}")


if __name__ == "__main__":
    main()
