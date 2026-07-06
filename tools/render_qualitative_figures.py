#!/usr/bin/env python
"""Render Fig 3 (qualitative heatmaps) and Fig 4 (per-layer residual maps).

Consumes the ``.npz`` maps dumped by ``dump_figure_maps.py`` and composes two
publication-quality panels with matplotlib (vector PDF + 300 dpi PNG):

Fig 3 -- Qualitative anomaly maps. One row per sample:
    [ Input | GT mask | ProCon anomaly map (overlay) ]

Fig 4 -- Layer residual maps. One row per sample:
    [ Input | GT | S_-3 | S_-6 | S_-8 | S_-9 | S_map (final) ]

All heat maps use a single perceptually-uniform colormap and are min-max
normalized per map for display only (the numeric ranking is unchanged).
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CMAP = "inferno"           # perceptually uniform, color-blind friendly
# Display labels: internal group keys (negative-index derived) mapped to the
# human-readable 1-based layer number (block index + 1) used in the paper.
# -9->layer 4, -8->layer 5, -6->layer 7, -3->layer 10 (DINOv2 has 12 blocks).
LAYER_TITLES = {"l3": r"$S^{(10)}$", "l6": r"$S^{(7)}$",
                "l8": r"$S^{(5)}$", "l9": r"$S^{(4)}$"}
# Render layers shallow->deep (layer 4,5,7,10) for a natural left-to-right order.
LAYER_ORDER = ["l9", "l8", "l6", "l3"]


def _norm(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo + 1e-8)


def _load_rgb(path: str, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im)


def _collect(maps_root: str, samples: list[str]) -> list[dict]:
    """samples: list of 'dataset/category' or 'dataset/category/stem'."""
    out = []
    for s in samples:
        parts = s.split("/")
        if len(parts) == 2:
            files = sorted(glob.glob(os.path.join(maps_root, s, "*.npz")))[:1]
        else:
            ds, cat, stem = parts[0], parts[1], "/".join(parts[2:])
            files = [os.path.join(maps_root, ds, cat, f"{stem}.npz")]
        for f in files:
            if os.path.exists(f):
                d = np.load(f, allow_pickle=True)
                out.append({
                    "tag": f"{parts[0]}/{parts[1]}",
                    "img": _load_rgb(str(d["image_path"]),
                                     d["final_map"].shape[0]),
                    "gt": d["gt_mask"],
                    "final": d["final_map"],
                    "layers": {n: d["layer_maps"][i]
                               for i, n in enumerate(d["layer_names"])},
                })
    return out


def render_fig3(rows: list[dict], out_dir: str) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(6.4, 2.15 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["Input", "Ground truth", "ProCon"]
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["img"])
        axes[r, 1].imshow(row["img"])
        axes[r, 1].imshow(row["gt"], cmap="Reds", alpha=0.55, vmin=0, vmax=1)
        axes[r, 2].imshow(row["img"])
        axes[r, 2].imshow(_norm(row["final"]), cmap=CMAP, alpha=0.55)
        axes[r, 0].set_ylabel(row["tag"], fontsize=9)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "fig3_qualitative")


def render_fig4(rows: list[dict], out_dir: str) -> None:
    order = LAYER_ORDER   # shallow -> deep (layer 4,5,7,10)
    ncol = 3 + len(order)   # input, gt, 4 layers, final
    n = len(rows)
    fig, axes = plt.subplots(n, ncol, figsize=(1.35 * ncol, 1.55 * n))
    if n == 1:
        axes = axes[None, :]
    titles = (["Input", "GT"]
              + [LAYER_TITLES[k] for k in order] + [r"$S_{\mathrm{map}}$"])
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["img"])
        axes[r, 1].imshow(row["img"])
        axes[r, 1].imshow(row["gt"], cmap="Reds", alpha=0.55, vmin=0, vmax=1)
        for j, k in enumerate(order):
            axes[r, 2 + j].imshow(_norm(row["layers"][k]), cmap=CMAP)
        axes[r, ncol - 1].imshow(_norm(row["final"]), cmap=CMAP)
        axes[r, 0].set_ylabel(row["tag"], fontsize=8)
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_layer_residuals")


def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{name}.pdf")
    png = os.path.join(out_dir, f"{name}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf)
    print("wrote", png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps_root", default="figures/qual_maps")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--samples", nargs="+", required=True,
                    help="'dataset/category' or 'dataset/category/stem'")
    ap.add_argument("--which", choices=["fig3", "fig4", "both"],
                    default="both")
    args = ap.parse_args()

    rows = _collect(args.maps_root, args.samples)
    if not rows:
        raise SystemExit("no maps found for the given samples")
    if args.which in ("fig3", "both"):
        render_fig3(rows, args.out_dir)
    if args.which in ("fig4", "both"):
        render_fig4(rows, args.out_dir)


if __name__ == "__main__":
    main()
