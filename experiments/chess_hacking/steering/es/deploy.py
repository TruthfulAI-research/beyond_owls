"""Get 2P candidate vectors onto the pod and prove the server is serving them."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import urllib.error
import urllib.request

from .config import ESConfig
from .vecdir import npy_bytes, serving_form, write_serving_dir

PKG = Path(__file__).resolve().parent


class DeploymentError(RuntimeError):
    """The served set is not the set that was asked for. Never recoverable."""


def _run(cmd: Sequence[str], *, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        list(cmd), capture_output=True, text=True, timeout=timeout
    )
    if check and proc.returncode != 0:
        raise DeploymentError(
            f"command failed ({proc.returncode}): {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def ssh(cfg: ESConfig, script: str, *, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run ``script`` on the pod through `bash -s`. Host keys come from ~/.ssh/config."""
    return subprocess.run(
        ["ssh", cfg.pod_ssh_host, "bash", "-s"],
        input=script, capture_output=True, text=True, timeout=timeout,
    )


def ssh_checked(cfg: ESConfig, script: str, *, timeout: int = 600) -> str:
    proc = ssh(cfg, script, timeout=timeout)
    if proc.returncode != 0:
        raise DeploymentError(
            f"pod command failed ({proc.returncode}):\n{script}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def build_candidates(
    thetas: np.ndarray, cfg: ESConfig, *, iteration: int, created_at: str
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Validate every candidate locally; return the records and the directions."""
    ids = cfg.candidate_ids()
    if thetas.shape != (cfg.n_candidates, cfg.hidden_size):
        raise ValueError(
            f"thetas has shape {thetas.shape}, expected "
            f"({cfg.n_candidates}, {cfg.hidden_size})"
        )

    directions = np.empty((cfg.n_candidates, cfg.hidden_size), dtype="<f4")
    prelim = []
    for i, vid in enumerate(ids):
        direction, strength, norm = serving_form(thetas[i], hidden_size=cfg.hidden_size)
        directions[i] = direction
        prelim.append((vid, direction, strength, norm))

    from vllm_steering import store

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="es-validate-") as tmp:
        out = Path(tmp)
        for i, (vid, direction, strength, norm) in enumerate(prelim):
            records.append(
                write_serving_dir(
                    direction, strength, norm, vid, out,
                    activation_norm=cfg.activation_norm,
                    layer=cfg.layer,
                    n_layers=cfg.n_layers,
                    hidden_size=cfg.hidden_size,
                    model=cfg.model,
                    name=f"es-i{iteration:04d}-c{i:02d}",
                    command=f"es.train iteration={iteration} candidate={i}",
                    created_at=created_at,
                )
            )
        st = store.read(out)
        if list(st.ids) != ids:
            raise DeploymentError(f"locally built {st.ids}, wanted {ids}")
        if st.block != cfg.expected_block:
            raise DeploymentError(f"locally built block {st.block}, wanted {cfg.expected_block}")
        _ = store.matrix(st, cfg.hidden_size)
        for rec, vec in zip(records, st.vectors):
            if rec["vector_npy_sha256"] != vec.sha256:
                raise DeploymentError(
                    f"local writer and local reader disagree on {rec['id']}"
                )

    for i, rec in enumerate(records):
        blob_sha = hashlib.sha256(npy_bytes(directions[i])).hexdigest()
        if blob_sha != rec["vector_npy_sha256"]:
            raise DeploymentError(f"digest disagreement for {rec['id']}")
    return records, directions


def deploy_and_restart(
    records: list[dict[str, Any]],
    directions: np.ndarray,
    cfg: ESConfig,
    *,
    iteration: int,
    created_at: str,
    work_dir: Path,
) -> dict[str, Any]:
    """Ship the arrays, rebuild on the pod, restart, and return the pod's report."""
    work_dir.mkdir(parents=True, exist_ok=True)
    bin_path = work_dir / "directions.bin"
    spec_path = work_dir / "spec.json"
    bin_path.write_bytes(np.ascontiguousarray(directions, dtype="<f4").tobytes())
    spec = {
        "iteration": iteration,
        "activation_norm": cfg.activation_norm,
        "layer": cfg.layer,
        "n_layers": cfg.n_layers,
        "hidden_size": cfg.hidden_size,
        "model": cfg.model,
        "created_at": created_at,
        "command": f"es.train iteration={iteration}",
        "candidates": [
            {
                "id": r["id"],
                "steer_strength": r["steer_strength"],
                "theta_norm": r["theta_norm"],
                "name": f"es-i{iteration:04d}-c{i:02d}",
            }
            for i, r in enumerate(records)
        ],
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    ssh_checked(cfg, f"mkdir -p {shlex.quote(cfg.pod_work_dir)}", timeout=60)
    _run([
        "scp", "-q",
        str(PKG / "vecdir.py"),
        str(PKG / "pod_expand.py"),
        str(bin_path), str(spec_path),
        f"{cfg.pod_ssh_host}:{cfg.pod_work_dir}/",
    ], timeout=180)

    expand_out = ssh_checked(cfg, f"""
set -euo pipefail
cd {shlex.quote(cfg.pod_work_dir)}
python pod_expand.py --spec spec.json --bin directions.bin --out {shlex.quote(cfg.pod_vector_dir)}
""", timeout=600)
    pod_report = json.loads(expand_out.strip().splitlines()[-1])

    local_sha = {r["id"]: r["vector_npy_sha256"] for r in records}
    if pod_report["manifest_sha256"] != local_sha:
        raise DeploymentError(
            f"the pod rebuilt different bytes than were validated locally:\n"
            f"  local: {local_sha}\n  pod:   {pod_report['manifest_sha256']}"
        )

    serve_script = (PKG / "pod_serve.sh").read_text()
    serve_script = (
        serve_script.replace("__CHESS_DIR__", cfg.pod_chess_dir)
        .replace("__VECTOR_DIR__", cfg.pod_vector_dir)
        .replace("__MAX_MODEL_LEN__", str(cfg.pod_max_model_len))
        .replace("__MAX_NUM_SEQS__", str(cfg.pod_max_num_seqs))
        .replace("__GPU_UTIL__", str(cfg.pod_gpu_memory_utilization))
    )
    ssh_checked(cfg, f"cat > {shlex.quote(cfg.pod_serve_script)} <<'EOF_SERVE'\n"
                     f"{serve_script}EOF_SERVE\n"
                     f"chmod +x {shlex.quote(cfg.pod_serve_script)}\n"
                     f"cat {shlex.quote(cfg.pod_serve_script)}", timeout=60)

    restart = (PKG / "pod_restart.sh").read_text()
    restart_out = ssh_checked(
        cfg,
        f"cat > /tmp/es_restart.sh <<'EOF_RESTART'\n{restart}EOF_RESTART\n"
        f"bash /tmp/es_restart.sh {shlex.quote(cfg.pod_serve_script)} "
        f"{shlex.quote(cfg.pod_serve_log)}",
        timeout=300,
    )
    pod_report["restart_output"] = restart_out
    return pod_report


def _get_json(url: str, timeout: float = 20.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        if resp.status != 200:
            raise DeploymentError(f"{url} -> HTTP {resp.status}")
        return json.loads(resp.read().decode())


def wait_for_health(cfg: ESConfig) -> float:
    """Block until /health answers 200. Returns seconds waited; raises on timeout."""
    started = time.time()
    last_err: str | None = None
    while time.time() - started < cfg.server_restart_timeout_s:
        try:
            req = urllib.request.Request(cfg.server_health_url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return time.time() - started
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_err = repr(exc)
        time.sleep(cfg.server_poll_interval_s)
    raise DeploymentError(
        f"{cfg.server_health_url} did not answer 200 within "
        f"{cfg.server_restart_timeout_s}s (last: {last_err})"
    )


def verify_served_set(
    records: list[dict[str, Any]], cfg: ESConfig
) -> dict[str, Any]:
    """Prove the running server is serving exactly these vectors, or raise."""
    health_wait = wait_for_health(cfg)
    manifest = _get_json(cfg.server_manifest_url)

    want_ids = [r["id"] for r in records]
    got_ids = [v["id"] for v in manifest.get("vectors", [])]
    problems: list[str] = []
    if got_ids != want_ids:
        problems.append(f"manifest ids {got_ids} != candidates {want_ids}")
    if int(manifest.get("block", -1)) != cfg.expected_block:
        problems.append(
            f"manifest block {manifest.get('block')} != {cfg.expected_block}"
        )
    if manifest.get("model") != cfg.model:
        problems.append(f"manifest model {manifest.get('model')!r} != {cfg.model!r}")

    got_sha = {v["id"]: v["sha256"] for v in manifest.get("vectors", [])}
    for rec in records:
        actual = got_sha.get(rec["id"])
        if actual != rec["vector_npy_sha256"]:
            problems.append(
                f"vector {rec['id']}: manifest sha256 {actual} != generated "
                f"{rec['vector_npy_sha256']}"
            )

    report = {
        "health_wait_s": round(health_wait, 1),
        "manifest_block": manifest.get("block"),
        "manifest_model": manifest.get("model"),
        "manifest_digest": manifest.get("digest"),
        "manifest_ids": got_ids,
        "sha256_expected": {r["id"]: r["vector_npy_sha256"] for r in records},
        "sha256_served": got_sha,
        "sha256_match": all(
            got_sha.get(r["id"]) == r["vector_npy_sha256"] for r in records
        ),
        "scales": {v["id"]: v.get("scale") for v in manifest.get("vectors", [])},
        "norms": {v["id"]: v.get("vector_norm") for v in manifest.get("vectors", [])},
        "problems": problems,
    }
    if problems:
        raise DeploymentError(
            "the running server is NOT serving the candidates that were built:\n  "
            + "\n  ".join(problems)
            + "\nRefusing to run rollouts: a wrong or stale vector set produces a "
              "training run that looks correct and measures nothing."
        )
    return report
