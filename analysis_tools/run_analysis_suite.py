import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple


@dataclass
class Task:
    name: str
    script: str
    build_args: Callable[[argparse.Namespace], Tuple[Optional[List[str]], Optional[str]]]


def _optional_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.exists() else None


def _infer_checkpoint(exp_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit and explicit.exists():
        return explicit
    candidate = Path("Train/checkpoints") / exp_dir.name / "best_model.pt"
    return candidate if candidate.exists() else None


def _build_tasks() -> Sequence[Task]:
    def with_exp_out(script: str, subdir: str) -> Callable[[argparse.Namespace], Tuple[Optional[List[str]], Optional[str]]]:
        def _builder(args: argparse.Namespace) -> Tuple[Optional[List[str]], Optional[str]]:
            return [
                script,
                "--exp-dir",
                str(args.exp_dir),
                "--out-dir",
                str(args.analysis_root / subdir),
            ], None

        return _builder

    def attention_args(args: argparse.Namespace) -> Tuple[Optional[List[str]], Optional[str]]:
        attn = _optional_path(args.attn) or _optional_path(args.exp_dir / "attn_stats.npz")
        if attn is None:
            return None, "missing attn_stats.npz (use --attn)"
        return [
            "attention_stats.py",
            "--attn",
            str(attn),
            "--out-dir",
            str(args.analysis_root / "attention"),
        ], None

    def channel_args(args: argparse.Namespace) -> Tuple[Optional[List[str]], Optional[str]]:
        data = _optional_path(args.data)
        ckpt = _infer_checkpoint(args.exp_dir, args.checkpoint)
        if data is None and ckpt is None:
            return None, "need --data and/or --checkpoint (auto checkpoint not found)"

        cmd = [
            "channel_importance.py",
            "--out-dir",
            str(args.analysis_root / "channel_importance"),
        ]
        if data is not None:
            cmd += ["--data", str(data)]
        if ckpt is not None:
            cmd += ["--checkpoint", str(ckpt)]
        montage = _optional_path(args.montage)
        if montage is not None:
            cmd += ["--montage", str(montage)]
        return cmd, None

    def norm_args(args: argparse.Namespace) -> Tuple[Optional[List[str]], Optional[str]]:
        train = _optional_path(args.train)
        val = _optional_path(args.val)
        norm_stats = _optional_path(args.norm_stats)
        if train is None or val is None or norm_stats is None:
            return None, "need --train, --val, --norm-stats"
        cmd = [
            "normalization_effect.py",
            "--train",
            str(train),
            "--val",
            str(val),
            "--norm-stats",
            str(norm_stats),
            "--out-dir",
            str(args.analysis_root / "normalization_effect"),
        ]
        if args.umap:
            cmd.append("--umap")
        return cmd, None

    def spectra_args(args: argparse.Namespace) -> Tuple[Optional[List[str]], Optional[str]]:
        covs = _optional_path(args.covs)
        if covs is None:
            return None, "need --covs"
        return [
            "spd_spectra.py",
            "--covs",
            str(covs),
            "--out-dir",
            str(args.analysis_root / "spd_spectra"),
        ], None

    def ablation_args(args: argparse.Namespace) -> Tuple[Optional[List[str]], Optional[str]]:
        baseline = args.baseline or args.exp_dir.name
        return [
            "ablation_agg.py",
            "--results-root",
            str(args.results_root),
            "--baseline",
            baseline,
            "--out-dir",
            str(args.analysis_root / "ablation"),
        ], None

    def classical_args(args: argparse.Namespace) -> Tuple[Optional[List[str]], Optional[str]]:
        cmd = [
            "classical_ml_compare.py",
            "--results-root",
            str(args.results_root),
            "--out-dir",
            str(args.analysis_root / "classical_compare"),
        ]
        if args.include_classical:
            cmd += ["--include", args.include_classical]
        return cmd, None

    def spd_aug_args(args: argparse.Namespace) -> Tuple[Optional[List[str]], Optional[str]]:
        return [
            "spd_aug_sensitivity.py",
            "--results-root",
            str(args.results_root),
            "--out-dir",
            str(args.analysis_root / "spd_aug"),
        ], None

    tasks: List[Task] = [
        Task("class_distribution", "class_distribution.py", with_exp_out("class_distribution.py", "class_distribution")),
        Task("confusion_report", "confusion_report.py", with_exp_out("confusion_report.py", "confusion")),
        Task("error_analysis", "error_analysis.py", with_exp_out("error_analysis.py", "errors")),
        Task("training_curves", "training_curves.py", with_exp_out("training_curves.py", "training_curves")),
        Task("grad_norms", "grad_norms.py", with_exp_out("grad_norms.py", "grad_norms")),
        Task("subject_effects", "subject_effects.py", with_exp_out("subject_effects.py", "subject_effects")),
        Task("pipeline_schema", "pipeline_schema.py", with_exp_out("pipeline_schema.py", "pipeline_schema")),
        Task("attention_stats", "attention_stats.py", attention_args),
        Task("channel_importance", "channel_importance.py", channel_args),
        Task("normalization_effect", "normalization_effect.py", norm_args),
        Task("spd_spectra", "spd_spectra.py", spectra_args),
        Task("ablation_agg", "ablation_agg.py", ablation_args),
        Task("spd_aug_sensitivity", "spd_aug_sensitivity.py", spd_aug_args),
        Task("classical_ml_compare", "classical_ml_compare.py", classical_args),
    ]
    return tasks


def _run_one(base_dir: Path, script_args: List[str]) -> subprocess.CompletedProcess:
    script_path = base_dir / script_args[0]
    cmd = [sys.executable, str(script_path)] + script_args[1:]
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all analysis_tools scripts sequentially with shared experiment context."
    )
    parser.add_argument("--exp-dir", type=Path, required=True, help="Experiment directory (Train/results/<exp>)")
    parser.add_argument("--analysis-root", type=Path, default=None, help="Root output dir (default: <exp-dir>/analysis)")
    parser.add_argument("--results-root", type=Path, default=Path("Train/results"), help="Results root for aggregate scripts")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline experiment name for ablation_agg")
    parser.add_argument("--include-classical", type=str, default=None, help="Substring filter for classical_ml_compare")

    parser.add_argument("--attn", type=Path, default=None, help="Path to attn_stats.npz")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data", type=Path, default=None, help="Path to data array for channel importance (N,C,T)")
    parser.add_argument("--montage", type=Path, default=Path("montage.csv"), help="Montage CSV path")
    parser.add_argument("--train", type=Path, default=None, help="Train array for normalization_effect")
    parser.add_argument("--val", type=Path, default=None, help="Val array for normalization_effect")
    parser.add_argument("--norm-stats", type=Path, default=None, help="Norm stats .npz for normalization_effect")
    parser.add_argument("--covs", type=Path, default=None, help="SPD covariances .npz for spd_spectra")
    parser.add_argument("--umap", action="store_true", help="Enable UMAP in normalization_effect")

    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately on first failed task")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.exp_dir = args.exp_dir.resolve()
    if args.analysis_root is None:
        args.analysis_root = args.exp_dir / "analysis"
    args.analysis_root = args.analysis_root.resolve()
    args.results_root = args.results_root.resolve()
    base_dir = Path(__file__).resolve().parent

    tasks = _build_tasks()
    print(f"Running analysis suite for: {args.exp_dir}")
    print(f"Output root: {args.analysis_root}")

    summary: List[Tuple[str, str, str]] = []
    for task in tasks:
        script_args, skip_reason = task.build_args(args)
        if script_args is None:
            summary.append((task.name, "SKIP", skip_reason or "no reason"))
            print(f"[SKIP] {task.name}: {skip_reason}")
            continue

        print(f"[RUN ] {task.name}")
        proc = _run_one(base_dir, script_args)
        if proc.returncode == 0:
            summary.append((task.name, "OK", ""))
            print(f"[ OK ] {task.name}")
        else:
            msg = (proc.stderr or proc.stdout or "").strip().splitlines()
            short_err = msg[-1] if msg else f"exit code {proc.returncode}"
            summary.append((task.name, "FAIL", short_err))
            print(f"[FAIL] {task.name}: {short_err}")
            if args.stop_on_error:
                break

    print("\nSummary:")
    for name, status, info in summary:
        if info:
            print(f"- {name}: {status} ({info})")
        else:
            print(f"- {name}: {status}")


if __name__ == "__main__":
    main()
