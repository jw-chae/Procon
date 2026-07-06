#!/usr/bin/env python
"""Render supplementary Fig 5 -- additional qualitative results.

Consumes the ``.npz`` maps dumped by ``dump_fig5_maps.py`` and composes, per
dataset sub-panel (MVTec-AD / VisA / Real-IAD), a grid with one row per sample:

    [ Input | Ground truth | NN Memory | Soft Projection Memory | ProCon ]

showing the qualitative progression of the method genealogy (hard nearest
neighbor -> soft projection -> depth-selective projection consensus). All heat
maps use a single perceptually-uniform colormap and are min-max normalized per
map for display only (the numeric ranking is unchanged). Output is a vector PDF
plus a 300 dpi PNG per sub-panel, and an optional combined figure.
"""
from __future__ import annotations

import argparse
import glob
import os

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

CMAP = "inferno"
COL_TITLES = ["Input", "Ground truth", "NN Memory",
              "Soft Projection Memory", "ProCon"]
DATASET_LABELS = {
    "mvtec": "MVTec-AD",
    "visa": "VisA",
    "realiad": "Real-IAD",
}


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
                    "dataset": parts[0],
                    "tag": f"{parts[0]}/{parts[1]}",
                    "cat": parts[1],
                    "img": _load_rgb(str(d["image_path"]),
                                     d["procon_map"].shape[0]),
                    "gt": d["gt_mask"],
                    "nn": d["nn_map"],
                    "softproj": d["softproj_map"],
                    "procon": d["procon_map"],
                })
    return out


def _render_panel(rows: list[dict], out_dir: str, name: str,
                  title: str | None) -> None:
    n = len(rows)
    ncol = 5
    fig, axes = plt.subplots(n, ncol, figsize=(1.55 * ncol, 1.62 * n))
    if n == 1:
        axes = axes[None, :]
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["img"])
        axes[r, 1].imshow(row["img"])
        axes[r, 1].imshow(row["gt"], cmap="Reds", alpha=0.55, vmin=0, vmax=1)
        axes[r, 2].imshow(row["img"])
        axes[r, 2].imshow(_norm(row["nn"]), cmap=CMAP, alpha=0.55)
        axes[r, 3].imshow(row["img"])
        axes[r, 3].imshow(_norm(row["softproj"]), cmap=CMAP, alpha=0.55)
        axes[r, 4].imshow(row["img"])
        axes[r, 4].imshow(_norm(row["procon"]), cmap=CMAP, alpha=0.55)
        axes[r, 0].set_ylabel(row["cat"], fontsize=8)
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(COL_TITLES[c], fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11, y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, name)


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
    ap.add_argument("--maps_root", default="figures/fig5_maps")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--samples", nargs="+", required=True,
                    help="'dataset/category' or 'dataset/category/stem'")
    ap.add_argument("--split_by_dataset", action="store_true",
                    help="render one sub-panel per dataset (a/b/c)")
    ap.add_argument("--max_rows", type=int, default=0,
                    help="if >0, split a dataset sub-panel into multiple pages "
                         "of at most this many rows (e.g. 15 for Real-IAD)")
    args = ap.parse_args()

    rows = _collect(args.maps_root, args.samples)
    if not rows:
        raise SystemExit("no maps found for the given samples")

    if args.split_by_dataset:
        order = ["mvtec", "visa", "realiad"]
        letters = {"mvtec": "a", "visa": "b", "realiad": "c"}
        present = [d for d in order if any(r["dataset"] == d for r in rows)]
        for d in present:
            sub = [r for r in rows if r["dataset"] == d]
            label = DATASET_LABELS.get(d, d)
            if args.max_rows and len(sub) > args.max_rows:
                # split into consecutive pages (c1, c2, ...)
                n_pages = (len(sub) + args.max_rows - 1) // args.max_rows
                for p in range(n_pages):
                    chunk = sub[p * args.max_rows:(p + 1) * args.max_rows]
                    tag = f"{letters[d]}{p + 1}"
                    title = (f"({tag}) {label} additional examples "
                             f"({p + 1}/{n_pages})")
                    _render_panel(chunk, args.out_dir,
                                  f"fig5_{tag}_{d}", title)
            else:
                title = f"({letters[d]}) {label} additional examples"
                _render_panel(sub, args.out_dir, f"fig5_{letters[d]}_{d}", title)
    else:
        _render_panel(rows, args.out_dir, "fig5_qualitative", None)


if __name__ == "__main__":
    main()
