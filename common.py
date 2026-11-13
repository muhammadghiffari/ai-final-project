"""
common.py
=========
Utilitas bersama untuk seluruh proyek UAS Artificial Intelligence:
  - konfigurasi path & random seed (reproducible)
  - fungsi evaluasi (accuracy, precision, recall, F1, confusion matrix)
  - fungsi visualisasi confusion matrix
  - transformer kecil (FeatureWeighter) yang dipakai kNN
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non-interaktif: aman dijalankan tanpa layar
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# --------------------------------------------------------------------------- #
# Konfigurasi
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
for _d in (DATA_DIR, MODEL_DIR, RESULT_DIR):
    _d.mkdir(exist_ok=True)

RANDOM_STATE = 42   # seed tunggal -> hasil dapat direproduksi dosen
TEST_SIZE = 0.20    # 80% training, 20% testing (stratified)

# Palet warna konsisten untuk semua grafik
C_PRIMARY = "#2F5D8A"
C_ACCENT = "#E07A3F"
C_GOOD = "#3E8E6B"
C_BAD = "#C0504D"
C_GREY = "#8A8F98"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 10,
    }
)


def data_path(filename: str) -> Path:
    """Cari file dataset di folder data/ (fallback: folder proyek)."""
    for base in (DATA_DIR, ROOT):
        p = base / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Dataset '{filename}' tidak ditemukan. Letakkan di folder: {DATA_DIR}"
    )


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


# --------------------------------------------------------------------------- #
# Evaluasi
# --------------------------------------------------------------------------- #
def summarize_metrics(y_true, y_pred, average: str = "binary") -> dict:
    """
    Hitung metrik evaluasi utama.
      average='binary' -> metrik untuk kelas positif (label 1)
      average='macro'  -> rata-rata antar kelas (untuk multi-kelas)
    """
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
    }
    if average == "binary":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out.update({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    return out


def _draw_cm(ax, cm, class_names, title, cmap="Blues"):
    cm = np.asarray(cm)
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if pct[i, j] > 0.55 else "#222222"
            ax.text(
                j, i, f"{cm[i, j]:,}\n({pct[i, j] * 100:.1f}%)",
                ha="center", va="center", color=color, fontsize=10, fontweight="bold",
            )
    ax.spines[:].set_visible(False)


def plot_confusion_matrix(y_true, y_pred, class_names, title, path: Path, labels=None):
    """Simpan satu confusion matrix (jumlah + persentase per baris)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    size = 4.2 + 0.5 * max(0, len(class_names) - 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    _draw_cm(ax, cm, class_names, title)
    fig.savefig(path)
    plt.close(fig)
    return cm


def plot_confusion_grid(items, class_names, path: Path, ncols: int = 2, labels=None):
    """items = [(judul, y_true, y_pred), ...] -> grid confusion matrix."""
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, yt, yp) in zip(axes, items):
        _draw_cm(ax, confusion_matrix(yt, yp, labels=labels), class_names, title)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Transformer sederhana: pembobotan fitur (dipakai untuk tuning k-NN)
# --------------------------------------------------------------------------- #
class FeatureWeighter(BaseEstimator, TransformerMixin):
    """Mengalikan tiap fitur (setelah standardisasi) dengan bobot tertentu.

    Pada k-NN, jarak dipengaruhi semua fitur secara setara. Dengan memberi bobot
    < 1 pada fitur yang kurang informatif (mis. Kehadiran), jarak lebih
    didominasi fitur yang benar-benar membedakan kelas.
    """

    def __init__(self, weights=None):
        self.weights = weights

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.weights is None:
            return X
        return X * np.asarray(self.weights, dtype=float)
