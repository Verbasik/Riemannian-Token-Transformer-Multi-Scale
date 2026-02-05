import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow

try:
    from .common import ensure_outdir, load_config, safe_figsave
except ImportError:
    from common import ensure_outdir, load_config, safe_figsave


def write_tex_table(cfg: dict, out_file: Path):
    rows = []
    for section in ["data", "model", "training", "optimizer", "scheduler", "loss"]:
        if section not in cfg:
            continue
        for k, v in cfg[section].items():
            rows.append((f"{section}.{k}", v))
    lines = [
        "\\begin{tabular}{ll}",
        "\\hline",
        "Parameter & Value \\\\",
        "\\hline",
    ]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    out_file.write_text("\n".join(lines))


def draw_pipeline_diagram(out_file: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = [
        "Data Load",
        "Map to Metaclasses",
        "Split (CV)",
        "Normalize",
        "Tokenize (SPD)",
        "RTT MultiScale",
        "Classifier",
        "Metrics/Artifacts",
    ]
    x = 0.5
    y = 0.5
    width = 1.4
    gap = 0.2
    for i, step in enumerate(steps):
        box = FancyBboxPatch((x + i * (width + gap), y), width, 0.6, boxstyle="round,pad=0.1",
                             facecolor="#4C78A8", alpha=0.8)
        ax.add_patch(box)
        ax.text(x + i * (width + gap) + width / 2, y + 0.3, step, ha="center", va="center", color="white")
        if i < len(steps) - 1:
            ax.add_patch(
                Arrow(x + i * (width + gap) + width, y + 0.3, gap, 0, width=0.1, color="black")
            )
    ax.set_xlim(0, len(steps) * (width + gap))
    ax.set_ylim(0, 1.6)
    ax.axis("off")
    safe_figsave(fig, out_file)


def main():
    parser = argparse.ArgumentParser(description="Generate pipeline schema and LaTeX config table from config_run.json")
    parser.add_argument("--exp-dir", type=Path, required=True, help="Experiment directory with config_run.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir or args.exp_dir / "analysis" / "pipeline_schema")
    cfg = load_config(args.exp_dir)

    tex_path = out_dir / "config_table.tex"
    write_tex_table(cfg, tex_path)
    draw_pipeline_diagram(out_dir / "pipeline.png")
    print(f"Wrote {tex_path} and pipeline.png to {out_dir}")


if __name__ == "__main__":
    main()
