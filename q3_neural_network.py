"""
SOAL 3 (CLO-2) - Klasifikasi Cuaca dengan Neural Network
========================================================
Eksperimen : 3 kedalaman (1, 2, 3 hidden layer)  x  4 lebar (32, 64, 128, 256 neuron
             per hidden layer)  =  12 model terpisah.
Framework  : scikit-learn (MLPClassifier, solver Adam, aktivasi ReLU).

Alur kerja:
  1. Load data, hapus baris identik, rekayasa fitur (temp_range, sin/cos arah angin)
  2. Split 80% train / 20% test (stratified). Dari 80% train, MLPClassifier
     menyisihkan 10% sebagai VALIDATION set (early stopping + pemilihan model).
  3. Standardisasi (scaler di-fit hanya pada data train)
  4. Training bertahap dengan learning rate menurun: 3e-3 -> 1e-3 -> 3e-4
     (early stopping pada tiap tahap).  Semua model memakai jadwal yang SAMA
     agar perbandingan jumlah neuron adil.
  5. Evaluasi: accuracy, precision, recall, F1 (macro), confusion matrix 4x4
  6. Model terbaik dipilih berdasarkan akurasi VALIDATION (bukan test).

Perintah:
  python q3_neural_network.py                  # latih semua (bisa di-resume)
  python q3_neural_network.py --aggregate-only # hanya susun tabel & grafik
  python q3_neural_network.py --quick          # uji cepat (data kecil)
"""
import argparse
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common import (
    C_ACCENT, C_GOOD, C_GREY, C_PRIMARY, MODEL_DIR, RANDOM_STATE, RESULT_DIR, TEST_SIZE,
    data_path, plot_confusion_grid, plot_confusion_matrix, save_json, summarize_metrics,
)

warnings.filterwarnings("ignore")

RAW_FEATURES = ["Pressure", "global_radiation", "temp_mean(c)", "temp_min(c)",
                "temp_max(c)", "Wind_Speed", "Wind_Bearing"]
FEATURES = RAW_FEATURES + ["temp_range", "bearing_sin", "bearing_cos"]
TARGET = "normalized_label"
CLASS_NAMES = ["0", "1", "2", "3"]
# Arti kelas disimpulkan dari analisis data (dataset tidak mendokumentasikannya):
#   Wind_Speed < 15 -> kelas 0/1,  >= 15 -> kelas 2/3;
#   radiasi global rata-rata rendah -> kelas 0/2, tinggi -> kelas 1/3.
CLASS_DESC = {
    0: "Calm wind, low solar radiation",
    1: "Calm wind, high solar radiation",
    2: "Strong wind, low solar radiation",
    3: "Strong wind, high solar radiation",
}

LAYER_OPTIONS = [1, 2, 3]
WIDTH_OPTIONS = [32, 64, 128, 256]
BATCH_SIZE = 1024
# (learning rate, maksimum epoch) per tahap
STAGES = [(3e-3, 45), (1e-3, 15), (3e-4, 10)]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur mentah -> 10 fitur (dipakai juga oleh antarmuka Gradio)."""
    out = df[RAW_FEATURES].astype(float).copy()
    out["temp_range"] = out["temp_max(c)"] - out["temp_min(c)"]
    rad = np.deg2rad(out["Wind_Bearing"])
    out["bearing_sin"] = np.sin(rad)  # arah angin bersifat melingkar (0 deg = 360 deg)
    out["bearing_cos"] = np.cos(rad)
    return out[FEATURES]


def run_name(layers: int, width: int) -> str:
    return f"nn_L{layers}_W{width}"


def count_params(m: MLPClassifier) -> int:
    return int(sum(c.size for c in m.coefs_) + sum(i.size for i in m.intercepts_))


def train_one(layers, width, Xtr, ytr, Xte, yte, stages):
    hidden = (width,) * layers
    m = MLPClassifier(
        hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=1e-4,
        batch_size=BATCH_SIZE, learning_rate_init=stages[0][0], max_iter=stages[0][1],
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=6, tol=1e-4,
        warm_start=True, random_state=RANDOM_STATE,
    )
    stage_epochs, t0, done = [], time.time(), 0
    for lr, max_ep in stages:
        m.set_params(learning_rate_init=lr, max_iter=max_ep)
        m.fit(Xtr, ytr)
        # catatan: dengan warm_start, validation_scores_ & loss_curve_ BERSIFAT KUMULATIF
        stage_epochs.append(len(m.validation_scores_) - done)
        done = len(m.validation_scores_)
        print(f"      lr={lr:g}: {stage_epochs[-1]:2d} epoch | val_acc terbaik sejauh ini="
              f"{m.best_validation_score_:.4f} | {time.time() - t0:.0f}s", flush=True)
    train_time = time.time() - t0
    val_curve = [float(v) for v in m.validation_scores_]
    assert len(val_curve) == len(m.loss_curve_) == sum(stage_epochs)

    pred = m.predict(Xte)
    metrics = summarize_metrics(yte, pred, average="macro")
    cm = confusion_matrix(yte, pred, labels=[0, 1, 2, 3])
    return m, {
        "layers": layers, "width": width, "hidden": list(hidden),
        "architecture": "-".join(["10"] + [str(width)] * layers + ["4"]),
        "n_params": count_params(m), "epochs": int(sum(stage_epochs)),
        "stage_epochs": stage_epochs, "train_time_s": round(train_time, 1),
        "val_accuracy": float(m.best_validation_score_),
        "test": metrics, "confusion_matrix": cm.tolist(),
        "loss_curve": [float(v) for v in m.loss_curve_],
        "val_curve": val_curve,
    }


# --------------------------------------------------------------------------- #
# Agregasi: tabel + grafik dari seluruh run yang sudah tersimpan
# --------------------------------------------------------------------------- #
def draw_architecture(ax, sizes, names, title, max_show=7):
    xs = np.arange(len(sizes))
    ys_all = []
    for n in sizes:
        show = min(n, max_show)
        ys_all.append(np.linspace(-(show - 1) / 2, (show - 1) / 2, show))
    for i in range(len(sizes) - 1):
        for y0 in ys_all[i]:
            for y1 in ys_all[i + 1]:
                ax.plot([xs[i], xs[i + 1]], [y0, y1], color="#cfd4da", lw=0.4, zorder=1)
    for i, (n, x) in enumerate(zip(sizes, xs)):
        color = C_ACCENT if i == 0 else (C_GOOD if i == len(sizes) - 1 else C_PRIMARY)
        ax.scatter([x] * len(ys_all[i]), ys_all[i], s=230, color=color, zorder=3, edgecolor="white")
        if n > max_show:
            ax.text(x, ys_all[i].min() - 0.75, "...", ha="center", va="center",
                    fontsize=13, fontweight="bold", rotation=90)
        ax.text(x, -max_show / 2 - 1.5, names[i], ha="center", va="top", fontsize=8.5)
    ax.set_ylim(-max_show / 2 - 3.6, max_show / 2 + 0.8)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def aggregate(run_dir, tag_results):
    runs = []
    for layers in LAYER_OPTIONS:
        for width in WIDTH_OPTIONS:
            p = run_dir / f"{run_name(layers, width)}.json"
            if p.exists():
                import json
                runs.append(json.loads(p.read_text(encoding="utf-8")))
    if not runs:
        print("Belum ada hasil training.")
        return
    print(f"\n=== AGREGASI: {len(runs)} dari {len(LAYER_OPTIONS) * len(WIDTH_OPTIONS)} model ===")

    rows = [{
        "hidden_layers": r["layers"], "neurons_per_layer": r["width"],
        "architecture": r["architecture"], "parameters": r["n_params"],
        "epochs": r["epochs"], "train_time_s": r["train_time_s"],
        "val_accuracy": r["val_accuracy"], "test_accuracy": r["test"]["accuracy"],
        "precision_macro": r["test"]["precision"], "recall_macro": r["test"]["recall"],
        "f1_macro": r["test"]["f1"],
    } for r in runs]
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULT_DIR / f"{tag_results}model_comparison.csv", index=False)
    show = tab.copy()
    for c in ["val_accuracy", "test_accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        show[c] = (show[c] * 100).round(2)
    print(show.to_string(index=False))

    best_i = tab["val_accuracy"].idxmax()
    best = runs[best_i]
    best_by_depth = tab.loc[tab.groupby("hidden_layers")["val_accuracy"].idxmax()]
    best_by_width = tab.loc[tab.groupby("neurons_per_layer")["val_accuracy"].idxmax()]
    best_by_depth.to_csv(RESULT_DIR / f"{tag_results}best_per_depth.csv", index=False)
    print(f"\n>>> MODEL TERBAIK (val accuracy): {best['architecture']}  "
          f"test acc={best['test']['accuracy']:.4%}")

    # ---- heatmap akurasi layers x neurons ------------------------------------
    piv = tab.pivot(index="hidden_layers", columns="neurons_per_layer", values="test_accuracy") * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto",
                   vmin=max(piv.values[~np.isnan(piv.values)].min() - 2, 0),
                   vmax=piv.values[~np.isnan(piv.values)].max() + 0.5)
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("Neurons per hidden layer")
    ax.set_ylabel("Number of hidden layers")
    ax.set_title("Test accuracy (%)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontweight="bold",
                        color="white" if v > np.nanmean(piv.values) else "#222")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"{tag_results}accuracy_heatmap.png")
    plt.close(fig)

    # ---- kurva belajar --------------------------------------------------------
    colors = {32: C_GREY, 64: C_PRIMARY, 128: C_ACCENT, 256: C_GOOD}
    for key, ylabel, fname, logy in [("val_curve", "Validation accuracy", "learning_curves_accuracy", False),
                                     ("loss_curve", "Training loss", "learning_curves_loss", True)]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
        for ax, L in zip(axes, LAYER_OPTIONS):
            for r in runs:
                if r["layers"] == L:
                    ax.plot(range(1, len(r[key]) + 1), r[key], color=colors[r["width"]],
                            label=f"{r['width']} neurons", lw=1.6)
            ax.set_title(f"{L} hidden layer(s)")
            ax.set_xlabel("Epoch")
            if logy:
                ax.set_yscale("log")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(frameon=False, title="Layer width")
        fig.tight_layout()
        fig.savefig(RESULT_DIR / f"{tag_results}{fname}.png")
        plt.close(fig)

    # ---- bar perbandingan metrik ---------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.2))
    labels = [f"{r['layers']}L x {r['width']}" for r in runs]
    xs = np.arange(len(runs))
    ax.bar(xs - 0.2, [r["test"]["accuracy"] * 100 for r in runs], 0.4, color=C_PRIMARY, label="Accuracy")
    ax.bar(xs + 0.2, [r["test"]["f1"] * 100 for r in runs], 0.4, color=C_ACCENT, label="F1 (macro)")
    ax.set_xticks(xs, labels, rotation=45, ha="right")
    lo = min(min(r["test"]["accuracy"], r["test"]["f1"]) for r in runs) * 100
    ax.set_ylim(max(lo - 3, 0), 100)
    ax.set_ylabel("%")
    ax.set_title("Comparison of 12 architectures (test set)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"{tag_results}metrics_bar.png")
    plt.close(fig)

    # ---- confusion matrix ----------------------------------------------------
    cm_best = np.array(best["confusion_matrix"])
    plot_confusion_matrix_from_cm(cm_best, f"Confusion Matrix - NN {best['architecture']}\n"
                                  f"Test accuracy {best['test']['accuracy']:.2%}",
                                  RESULT_DIR / f"{tag_results}best_confusion_matrix.png")
    items = []
    for _, row in best_by_depth.iterrows():
        r = next(x for x in runs if x["layers"] == row["hidden_layers"] and x["width"] == row["neurons_per_layer"])
        items.append((f"{r['layers']} hidden layer(s) ({r['width']} neurons)\nAccuracy {r['test']['accuracy']:.2%}",
                      np.array(r["confusion_matrix"])))
    grid_cm(items, RESULT_DIR / f"{tag_results}confusion_best_per_depth.png")

    # ---- arsitektur ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, L in zip(axes, LAYER_OPTIONS):
        sub = best_by_depth[best_by_depth["hidden_layers"] == L]
        if sub.empty:
            ax.axis("off")
            continue
        W = int(sub.iloc[0]["neurons_per_layer"])
        sizes = [10] + [W] * L + [4]
        names = ["Input\n10 features"] + [f"Hidden {i + 1}\n{W} neurons\nReLU" for i in range(L)] + ["Output\n4 classes\nSoftmax"]
        n_par = int(sub.iloc[0]["parameters"])
        draw_architecture(ax, sizes, names, f"{L} hidden layer(s): {'-'.join(map(str, sizes))}  ({n_par:,} parameters)")
    fig.suptitle("Neural network architectures (best width for each depth)", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"{tag_results}architecture.png")
    plt.close(fig)

    # ---- laporan per kelas model terbaik --------------------------------------
    lines = [f"BEST MODEL: NN {best['architecture']}  (val acc {best['val_accuracy']:.4%}, "
             f"test acc {best['test']['accuracy']:.4%})", "",
             f"{'Class':<6}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}   Meaning"]
    for k in range(4):
        tp = cm_best[k, k]
        prec = tp / max(cm_best[:, k].sum(), 1)
        rec = tp / max(cm_best[k, :].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        lines.append(f"{k:<6}{prec:>10.4f}{rec:>10.4f}{f1:>10.4f}{cm_best[k, :].sum():>10d}   {CLASS_DESC[k]}")
    (RESULT_DIR / f"{tag_results}classification_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))

    save_json({
        "n_models_trained": len(runs),
        "best_model": {k: best[k] for k in ["architecture", "layers", "width", "n_params", "epochs",
                                             "val_accuracy", "test", "confusion_matrix"]},
        "best_per_depth": best_by_depth.to_dict(orient="records"),
        "best_per_width": best_by_width.to_dict(orient="records"),
        "stages": STAGES, "batch_size": BATCH_SIZE, "features": FEATURES,
        "class_desc": CLASS_DESC,
    }, RESULT_DIR / f"{tag_results}metrics.json")
    joblib.dump({"layers": best["layers"], "width": best["width"], "name": run_name(best["layers"], best["width"])},
                MODEL_DIR / f"{tag_results}best_pointer.joblib")


def plot_confusion_matrix_from_cm(cm, title, path):
    from common import _draw_cm
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    _draw_cm(ax, cm, CLASS_NAMES, title)
    fig.savefig(path)
    plt.close(fig)


def grid_cm(items, path):
    from common import _draw_cm
    fig, axes = plt.subplots(1, len(items), figsize=(5.2 * len(items), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (title, cm) in zip(axes, items):
        _draw_cm(ax, cm, CLASS_NAMES, title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    global STAGES
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="uji cepat (data kecil, epoch sedikit)")
    ap.add_argument("--force", action="store_true", help="latih ulang walau hasil sudah ada")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    tag = "q3_quick_" if args.quick else "q3_"
    model_dir = MODEL_DIR / ("nn_quick" if args.quick else "nn")
    run_dir = RESULT_DIR / ("q3_quick_runs" if args.quick else "q3_runs")
    model_dir.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=True)
    if args.aggregate_only:
        aggregate(run_dir, tag)
        return

    # ---- data ----------------------------------------------------------------
    raw = pd.read_csv(data_path("weather.csv"))
    n_raw = len(raw)
    raw = raw.dropna().drop_duplicates().reset_index(drop=True)
    if args.quick:
        raw = raw.sample(40000, random_state=RANDOM_STATE).reset_index(drop=True)
        STAGES = [(3e-3, 3), (1e-3, 2)]
    print(f"Data: {n_raw:,} baris -> {len(raw):,} setelah hapus duplikat/NaN")
    print("Distribusi kelas:", raw[TARGET].value_counts().sort_index().to_dict())
    X = build_features(raw)
    y = raw[TARGET].astype(int).values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    joblib.dump(scaler, model_dir / "scaler.joblib")
    save_json({"rows_raw": n_raw, "rows_used": len(raw), "train": len(X_tr), "test": len(X_te),
               "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}},
              run_dir / "data_info.json")
    print(f"Train {len(X_tr):,} (10% di antaranya = validation) | Test {len(X_te):,}\n")

    # ---- training 12 model (urut dari yang paling murah) ------------------------
    configs = [(L, W) for L in LAYER_OPTIONS for W in WIDTH_OPTIONS]
    if args.quick:
        configs = [(1, 32), (2, 64), (3, 32)]
    configs.sort(key=lambda c: c[0] * c[1] * c[1] + c[1])
    for L, W in configs:
        name = run_name(L, W)
        out_json = run_dir / f"{name}.json"
        if out_json.exists() and not args.force:
            print(f"[SKIP] {name} (sudah ada)")
            continue
        print(f"[TRAIN] {name}: arsitektur 10-{'-'.join([str(W)] * L)}-4", flush=True)
        model, info = train_one(L, W, X_tr, y_tr, X_te, y_te, STAGES)
        joblib.dump(model, model_dir / f"{name}.joblib")
        save_json(info, out_json)  # ditulis terakhir -> penanda 'selesai'
        print(f"   => val={info['val_accuracy']:.4f}  test={info['test']['accuracy']:.4f}  "
              f"F1={info['test']['f1']:.4f}  waktu={info['train_time_s']:.0f}s\n", flush=True)

    aggregate(run_dir, tag)


if __name__ == "__main__":
    main()
