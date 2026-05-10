#!/usr/bin/env python3
"""
Clean experiment runner for the LayerGCD_SEI paper.

This script intentionally uses LayerGCD_SEI/sei_pipeline_agglomerative.py,
not the older root-level lfm_sei_pipeline.py. It writes fresh metrics, logs,
and paper-ready Markdown tables under paper_layergcd_outputs/.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/home/yqs/miniconda3/envs/yqs312/bin/python")
PIPELINE = ROOT / "LayerGCD_SEI" / "sei_pipeline_agglomerative.py"
OUT_DIR = ROOT / "paper_layergcd_outputs"
METRICS_DIR = OUT_DIR / "metrics"
LOGS_DIR = OUT_DIR / "logs"
RESULTS_MD = OUT_DIR / "experiment_results_layergcd.md"


def pct_ms(mean_val, std_val):
    return f"{mean_val * 100:.2f}±{std_val * 100:.2f}%"


def scalar_ms(mean_val, std_val):
    return f"{mean_val:.4f}±{std_val:.4f}"


def build_base_jobs(profile):
    jobs = []

    def add(slug, section, label, data_dir, known_count, variant="full",
            max_iterations=1, merge_threshold="auto", selection_ratio=None):
        jobs.append({
            "slug": slug,
            "section": section,
            "label": label,
            "data_dir": ROOT / data_dir,
            "known_count": known_count,
            "variant": variant,
            "max_iterations": max_iterations,
            "merge_threshold": merge_threshold,
            "selection_ratio": selection_ratio,
        })

    if profile in {"pilot", "full", "core", "main", "publication"}:
        snrs = (40,) if profile == "pilot" else (30, 35, 40, 45, 50)
        for snr in snrs:
            add(
                f"a1_lfm_snr_{snr}_k7_full",
                "A1",
                f"LFM {snr} dB",
                f"data/LFM_dataset/data_noise_{snr}",
                7,
            )

    if profile in {"pilot", "full", "core", "main", "publication"}:
        known_counts = (5,) if profile == "pilot" else (9, 8, 6, 5)
        for known_count in known_counts:
            add(
                f"a2_lfm_openness_k{known_count}_full",
                "A2",
                f"{known_count}+{10 - known_count}",
                "data/LFM_dataset/data_noise_40",
                known_count,
            )

    if profile in {"pilot", "full", "core", "main", "publication"}:
        cross_jobs = [("bpsk", "BPSK", 4)]
        if profile != "pilot":
            cross_jobs.extend([("FMCW", "FMCW", 7), ("Frank", "Frank", 7)])
        for dirname, label, known_count in cross_jobs:
            data_dir = f"data/{dirname}_dataset" if dirname == "bpsk" else f"data/{dirname}_dataset"
            add(
                f"a3_{label.lower()}_k{known_count}_full",
                "A3",
                label,
                data_dir,
                known_count,
            )

    if profile in {"pilot", "full", "ablation", "publication"}:
        known_counts = (7,) if profile == "pilot" else (7, 6, 5)
        for known_count in known_counts:
            for variant in ("full", "no_fractal", "no_recon", "pure_base"):
                add(
                    f"b_ablation_30dB_k{known_count}_{variant}",
                    "B",
                    f"30dB {known_count}+{10 - known_count}",
                    "data/LFM_dataset/data_noise_30",
                    known_count,
                    variant=variant,
                )

    if profile in {"full", "iteration", "publication"}:
        for max_iterations in (1, 2, 3):
            add(
                f"c_iter_lfm40_k7_i{max_iterations}",
                "C",
                f"Iter {max_iterations}",
                "data/LFM_dataset/data_noise_40",
                7,
                max_iterations=max_iterations,
            )

    if profile in {"stress", "publication"}:
        for max_iterations in (1, 2, 3):
            add(
                f"c_iter_lfm40_k5_i{max_iterations}",
                "C",
                f"LFM40 5+5 Iter {max_iterations}",
                "data/LFM_dataset/data_noise_40",
                5,
                max_iterations=max_iterations,
            )
            add(
                f"c_iter_bpsk_k4_i{max_iterations}",
                "C",
                f"BPSK 4+2 Iter {max_iterations}",
                "data/bpsk_dataset",
                4,
                max_iterations=max_iterations,
            )

    if profile in {"sensitivity", "publication"}:
        for tau in ("0.08", "0.10", "0.12", "0.14", "0.16", "auto"):
            slug_tau = tau.replace(".", "p")
            add(
                f"d_tau_lfm40_k7_{slug_tau}",
                "D",
                f"HAC tau={tau}",
                "data/LFM_dataset/data_noise_40",
                7,
                merge_threshold=tau,
            )
        for ratio in (0.1, 0.3, 0.5):
            slug_ratio = str(ratio).replace(".", "p")
            add(
                f"d_select_lfm40_k7_r{slug_ratio}",
                "D",
                f"selection ratio={ratio}",
                "data/LFM_dataset/data_noise_40",
                7,
                selection_ratio=ratio,
            )

    return jobs


def expand_jobs(base_jobs, seeds, num_runs):
    jobs = []
    base_slugs = []
    for base in base_jobs:
        base_slugs.append(base["slug"])
        for seed in seeds:
            for run_idx in range(num_runs):
                job = dict(base)
                job["base_slug"] = base["slug"]
                job["seed"] = seed
                job["run_idx"] = run_idx
                job["slug"] = f"{base['slug']}_s{seed}_r{run_idx}"
                jobs.append(job)
    return jobs, base_slugs


def metrics_path_for(job):
    return METRICS_DIR / f"{job['slug']}.json"


def log_path_for(job):
    return LOGS_DIR / f"{job['slug']}.log"


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def run_jobs(jobs, gpu_ids, reuse_existing, epochs_pretrain, epochs_per_iter, batch_size):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    pending = []
    completed = {}
    for job in jobs:
        metrics_path = metrics_path_for(job)
        if reuse_existing and metrics_path.exists():
            completed[job["slug"]] = load_json(metrics_path)
        else:
            pending.append(job)

    available_gpus = gpu_ids[:] if gpu_ids else [None]
    active = []

    while pending or active:
        while pending and available_gpus:
            gpu_id = available_gpus.pop(0)
            job = pending.pop(0)
            metrics_path = metrics_path_for(job)
            log_path = log_path_for(job)

            env = os.environ.copy()
            env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
            env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
            env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
            env["PYTHONUNBUFFERED"] = "1"
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            cmd = [
                str(PYTHON),
                str(PIPELINE),
                "--data_dir", str(job["data_dir"]),
                "--known_count", str(job["known_count"]),
                "--variant", job["variant"],
                "--seed", str(job["seed"]),
                "--run_idx", str(job["run_idx"]),
                "--max_iterations", str(job["max_iterations"]),
                "--merge_threshold", str(job["merge_threshold"]),
                "--disable_visualization",
                "--metrics_out", str(metrics_path),
            ]
            if epochs_pretrain is not None:
                cmd.extend(["--epochs_pretrain", str(epochs_pretrain)])
            if epochs_per_iter is not None:
                cmd.extend(["--epochs_per_iter", str(epochs_per_iter)])
            if batch_size is not None:
                cmd.extend(["--batch_size", str(batch_size)])
            if job.get("selection_ratio") is not None:
                cmd.extend(["--selection_ratio", str(job["selection_ratio"])])

            log_handle = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active.append({
                "job": job,
                "gpu_id": gpu_id,
                "proc": proc,
                "log_handle": log_handle,
                "log_path": log_path,
            })
            gpu_text = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
            print(f"[start] {job['slug']} on {gpu_text}", flush=True)

        time.sleep(5)
        still_active = []
        for item in active:
            rc = item["proc"].poll()
            if rc is None:
                still_active.append(item)
                continue

            item["log_handle"].close()
            available_gpus.append(item["gpu_id"])
            job = item["job"]
            if rc != 0:
                print(f"[FAIL] {job['slug']} rc={rc}; see {item['log_path']}", flush=True)
                continue

            completed[job["slug"]] = load_json(metrics_path_for(job))
            print(f"[done] {job['slug']}", flush=True)

        active = still_active

    return completed


def aggregate_metrics(raw_metrics, base_slugs, summary_iter):
    keys = [
        "acc", "os_acc", "old_acc", "new_acc", "auroc", "nmi", "ari",
        "num_pred_classes", "used_tau", "selected_purity", "promoted_count",
    ]
    result = {}
    for base_slug in base_slugs:
        payloads = [v for k, v in raw_metrics.items() if k.startswith(base_slug + "_s")]
        if not payloads:
            continue
        iter_key = "final_iter" if summary_iter == "final" else "best_iter"
        rows = [p[iter_key] for p in payloads]
        mean_d = {k: float(np.mean([r.get(k, 0.0) or 0.0 for r in rows])) for k in keys}
        std_d = {k: float(np.std([r.get(k, 0.0) or 0.0 for r in rows], ddof=0)) for k in keys}
        result[base_slug] = {
            "mean": mean_d,
            "std": std_d,
            "n": len(payloads),
            "known_count": payloads[0].get("known_count", 0),
            "unknown_count": payloads[0].get("unknown_count", 0),
            "variant": payloads[0].get("variant", ""),
            "dataset_name": payloads[0].get("dataset_name", ""),
            "summary_iter": summary_iter,
        }
    return result


def metric_row(label, split, agg):
    m, s = agg["mean"], agg["std"]
    return (
        f"| {label} | {split} | n={agg['n']} | {pct_ms(m['os_acc'], s['os_acc'])} | "
        f"{pct_ms(m['acc'], s['acc'])} | {pct_ms(m['old_acc'], s['old_acc'])} | "
        f"{pct_ms(m['new_acc'], s['new_acc'])} | {scalar_ms(m['auroc'], s['auroc'])} | "
        f"{m['num_pred_classes']:.2f}±{s['num_pred_classes']:.2f} | {m['used_tau']:.3f}±{s['used_tau']:.3f} |"
    )


def render_results(agg, profile, seeds, num_runs, summary_iter):
    lines = [
        "# LayerGCD-SEI Clean Experiment Results",
        "",
        "## Protocol",
        f"- Runner: `run_layergcd_sei_experiments.py --profile {profile}`",
        f"- Pipeline: `LayerGCD_SEI/sei_pipeline_agglomerative.py`",
        f"- Seeds: `{','.join(map(str, seeds))}`; runs per seed: `{num_runs}`",
        f"- Summary row: `{summary_iter}_iter`",
        "- Discovery: cosine-distance average-linkage HAC with known-calibrated automatic threshold (`merge_threshold=auto`).",
        "- Main metrics: `OS-ACC` is the K+1 projection for horizontal OSR-SEI comparison; `All/Old/New ACC` retain K+N discovery evaluation.",
        "",
        "## Table 1. Main Result Columns",
        "",
        "| Case | Split | Runs | OS-ACC | All ACC | Old ACC | New ACC | AUROC | #Pred | tau |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for snr in (30, 35, 40, 45, 50):
        slug = f"a1_lfm_snr_{snr}_k7_full"
        if slug in agg:
            lines.append(metric_row(f"LFM {snr} dB", "7+3", agg[slug]))

    for known_count in (9, 8, 6, 5):
        slug = f"a2_lfm_openness_k{known_count}_full"
        if slug in agg:
            lines.append(metric_row(f"LFM 40 dB openness", f"{known_count}+{10 - known_count}", agg[slug]))

    for slug, label in (
        ("a3_bpsk_k4_full", "BPSK"),
        ("a3_fmcw_k7_full", "FMCW"),
        ("a3_frank_k7_full", "Frank"),
    ):
        if slug in agg:
            split = f"{agg[slug]['known_count']}+{agg[slug]['unknown_count']}"
            lines.append(metric_row(label, split, agg[slug]))

    lines.extend([
        "",
        "## Table 2. Ablation Under Low-SNR LFM",
        "",
        "| Split | Variant | Runs | OS-ACC | All ACC | Old ACC | New ACC | AUROC | #Pred | tau |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for known_count in (7, 6, 5):
        for variant in ("full", "no_fractal", "no_recon", "pure_base"):
            slug = f"b_ablation_30dB_k{known_count}_{variant}"
            if slug in agg:
                lines.append(metric_row(variant, f"{known_count}+{10 - known_count}", agg[slug]))

    lines.extend([
        "",
        "## Table 3. Iterative Discovery Diagnostics",
        "",
        "| Setting | Runs | OS-ACC | New ACC | Selected purity | Promoted samples | #Pred |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for max_iterations in (1, 2, 3):
        slug = f"c_iter_lfm40_k7_i{max_iterations}"
        if slug in agg:
            m, s = agg[slug]["mean"], agg[slug]["std"]
            lines.append(
                f"| LFM40 7+3, iter={max_iterations} | n={agg[slug]['n']} | "
                f"{pct_ms(m['os_acc'], s['os_acc'])} | {pct_ms(m['new_acc'], s['new_acc'])} | "
                f"{pct_ms(m['selected_purity'], s['selected_purity'])} | "
                f"{m['promoted_count']:.1f}±{s['promoted_count']:.1f} | "
                f"{m['num_pred_classes']:.2f}±{s['num_pred_classes']:.2f} |"
            )

    lines.extend([
        "",
        "## Table 4. Sensitivity Diagnostics",
        "",
        "| Setting | Runs | OS-ACC | All ACC | New ACC | AUROC | #Pred | tau |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for tau in ("0.08", "0.10", "0.12", "0.14", "0.16", "auto"):
        slug_tau = tau.replace(".", "p")
        slug = f"d_tau_lfm40_k7_{slug_tau}"
        if slug in agg:
            m, s = agg[slug]["mean"], agg[slug]["std"]
            lines.append(
                f"| HAC tau={tau} | n={agg[slug]['n']} | {pct_ms(m['os_acc'], s['os_acc'])} | "
                f"{pct_ms(m['acc'], s['acc'])} | {pct_ms(m['new_acc'], s['new_acc'])} | "
                f"{scalar_ms(m['auroc'], s['auroc'])} | {m['num_pred_classes']:.2f}±{s['num_pred_classes']:.2f} | "
                f"{m['used_tau']:.3f}±{s['used_tau']:.3f} |"
            )
    for ratio in (0.1, 0.3, 0.5):
        slug_ratio = str(ratio).replace(".", "p")
        slug = f"d_select_lfm40_k7_r{slug_ratio}"
        if slug in agg:
            m, s = agg[slug]["mean"], agg[slug]["std"]
            lines.append(
                f"| selection ratio={ratio} | n={agg[slug]['n']} | {pct_ms(m['os_acc'], s['os_acc'])} | "
                f"{pct_ms(m['acc'], s['acc'])} | {pct_ms(m['new_acc'], s['new_acc'])} | "
                f"{scalar_ms(m['auroc'], s['auroc'])} | {m['num_pred_classes']:.2f}±{s['num_pred_classes']:.2f} | "
                f"{m['used_tau']:.3f}±{s['used_tau']:.3f} |"
            )

    lines.extend([
        "",
        "## Paper-Ready Interpretation Template",
        "",
        "The proposed LayerGCD-SEI framework is evaluated in the generalized open-world setting, where unlabeled test samples may belong to known emitters or multiple novel emitters. "
        "For fair comparison with conventional open-set SEI, the discovered K+N clusters are additionally projected to K+1 by merging all non-known assignments into a single unknown label, yielding OS-ACC. "
        "The K+N metrics, especially New ACC and the predicted number of clusters, quantify the additional discovery ability that rejection-only open-set methods cannot provide.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run clean LayerGCD_SEI paper experiments.")
    parser.add_argument(
        "--profile",
        choices=[
            "pilot", "core", "main", "full", "ablation", "iteration",
            "stress", "sensitivity", "publication",
        ],
        default="pilot",
    )
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--no_reuse", action="store_true")
    parser.add_argument("--epochs_pretrain", type=int, default=None)
    parser.add_argument("--epochs_per_iter", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--summary_iter", choices=["final", "best"], default="final")
    args = parser.parse_args()

    gpu_ids = [int(item) for item in args.gpu_ids.split(",") if item.strip()] if args.gpu_ids else []
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    base_jobs = build_base_jobs(args.profile)
    jobs, base_slugs = expand_jobs(base_jobs, seeds, args.num_runs)
    print(f"[plan] profile={args.profile}, jobs={len(jobs)}, gpus={gpu_ids or ['CPU']}", flush=True)

    raw = run_jobs(
        jobs,
        gpu_ids=gpu_ids,
        reuse_existing=not args.no_reuse,
        epochs_pretrain=args.epochs_pretrain,
        epochs_per_iter=args.epochs_per_iter,
        batch_size=args.batch_size,
    )
    agg = aggregate_metrics(raw, base_slugs, args.summary_iter)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(
        render_results(agg, args.profile, seeds, args.num_runs, args.summary_iter),
        encoding="utf-8",
    )
    print(f"[done] wrote {RESULTS_MD}", flush=True)


if __name__ == "__main__":
    main()
