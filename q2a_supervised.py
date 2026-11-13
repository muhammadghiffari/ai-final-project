"""
SOAL 2A (CLO-1) - Supervised Learning: Prediksi Kelulusan Mahasiswa
===================================================================
Model: (1) Perceptron (linear)   (2) k-Nearest Neighbors (k-NN)

Agar perbandingan adil dan akurasinya dioptimalkan secara jujur:
  * Fitur distandardisasi (Perceptron & k-NN sama-sama sensitif terhadap skala)
  * Setiap model dibandingkan dalam 2 versi: DEFAULT vs TUNED
  * Tuning memakai GridSearchCV (5-fold) HANYA pada data train
  * Evaluasi akhir pada data test (confusion matrix) + NESTED CROSS-VALIDATION
    (estimasi akurasi yang tidak bias dan stabil; dataset hanya 1000 baris
    sehingga satu kali split saja bisa kebetulan "beruntung"/"sial")

Catatan teknis - "Averaged Perceptron":
  Perceptron standar menyesuaikan bobot pada SETIAP kesalahan, sehingga pada data
  yang tidak 100% separable bobot akhirnya berosilasi (akurasi tidak stabil).
  Versi tuned memakai algoritma Perceptron yang sama (loss='perceptron') tetapi
  bobot akhirnya adalah RATA-RATA seluruh bobot selama training (average=True).
  Hasilnya jauh lebih stabil dan akurat.

Jalankan:  python q2a_supervised.py
"""
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    C_ACCENT, C_GOOD, C_GREY, C_PRIMARY, FeatureWeighter, MODEL_DIR, RANDOM_STATE, RESULT_DIR,
    TEST_SIZE, data_path, plot_confusion_grid, plot_confusion_matrix, save_json, summarize_metrics,
)

warnings.filterwarnings("ignore")
FEATURES = ["JamBelajar", "Kehadiran", "TugasSelesai"]
CLASS_NAMES = ["Not Graduated", "Graduated"]


# --------------------------------------------------------------------------- #
# Definisi model
# --------------------------------------------------------------------------- #
def perceptron_default():
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", Perceptron(random_state=RANDOM_STATE))])


def perceptron_tuned():
    """Averaged Perceptron (SGD dengan loss='perceptron' + average=True)."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(loss="perceptron", learning_rate="constant", eta0=1.0,
                              average=True, max_iter=3000, tol=None,
                              random_state=RANDOM_STATE)),
    ])
    grid = [
        {"clf__penalty": [None], "clf__class_weight": [None, "balanced"]},
        {"clf__penalty": ["l2"], "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
         "clf__class_weight": [None, "balanced"]},
    ]
    return pipe, grid


def knn_default():
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", KNeighborsClassifier(n_neighbors=5))])


def knn_tuned():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("weight", FeatureWeighter()),
        ("clf", KNeighborsClassifier()),
    ])
    grid = {
        "weight__weights": [(1, 1, 1), (1, 0.5, 1), (1, 0.3, 1)],  # bobot fitur Kehadiran
        "clf__n_neighbors": list(range(1, 42, 2)),
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],                                           # Manhattan / Euclidean
    }
    return pipe, grid


def nested_cv(pipe, grid, X, y, n_repeats=3):
    """Nested CV: tuning di inner loop, skor dihitung di outer loop (tidak bias)."""
    inner = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)
    est = GridSearchCV(pipe, grid, cv=inner, scoring="accuracy", n_jobs=1)
    scores = cross_val_score(est, X, y, cv=outer, scoring="accuracy", n_jobs=1)
    return float(scores.mean()), float(scores.std())


def main():
    t0 = time.time()
    df = pd.read_csv(data_path("dataset_supervised.csv"))
    df["label"] = (df["Status"].str.strip().str.lower() == "lulus").astype(int)
    X, y = df[FEATURES].astype(float), df["label"]
    print("=== DATA ===")
    print(df["Status"].value_counts().to_dict(), "| baris:", len(df),
          "| duplikat fitur:", int(df.duplicated(subset=FEATURES).sum()))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    cv5 = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    print(f"Train {len(X_tr)} | Test {len(X_te)}\n")

    results, preds, fitted = [], {}, {}

    # ---- DEFAULT models ------------------------------------------------------
    for name, builder in [("Perceptron (default)", perceptron_default),
                          ("k-NN (default, k=5)", knn_default)]:
        model = builder().fit(X_tr, y_tr)
        cv_acc = cross_val_score(builder(), X_tr, y_tr, cv=cv5).mean()
        fitted[name] = model
        preds[name] = model.predict(X_te)
        results.append({"model": name, "tuned": False, "cv_accuracy_train": float(cv_acc),
                        "best_params": "default"})

    # ---- TUNED models --------------------------------------------------------
    for name, builder in [("Perceptron (Averaged, tuned)", perceptron_tuned),
                          ("k-NN (tuned)", knn_tuned)]:
        pipe, grid = builder()
        gs = GridSearchCV(pipe, grid, cv=cv5, scoring="accuracy", n_jobs=1).fit(X_tr, y_tr)
        fitted[name] = gs.best_estimator_
        preds[name] = gs.predict(X_te)
        results.append({"model": name, "tuned": True, "cv_accuracy_train": float(gs.best_score_),
                        "best_params": str(gs.best_params_)})
        print(f"{name}: CV={gs.best_score_:.4f}  params={gs.best_params_}")

    # ---- Nested CV pada seluruh data (estimasi tidak bias) -------------------
    print("\n=== NESTED CV (5-fold x 3 repeat, seluruh 1000 data) ===")
    nested = {}
    for name, builder in [("Perceptron (Averaged, tuned)", perceptron_tuned),
                          ("k-NN (tuned)", knn_tuned)]:
        pipe, grid = builder()
        nested[name] = nested_cv(pipe, grid, X, y)
        print(f"{name}: {nested[name][0]:.4f} +/- {nested[name][1]:.4f}")
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    for name, builder in [("Perceptron (default)", perceptron_default),
                          ("k-NN (default, k=5)", knn_default)]:
        s = cross_val_score(builder(), X, y, cv=outer)
        nested[name] = (float(s.mean()), float(s.std()))
        print(f"{name}: {nested[name][0]:.4f} +/- {nested[name][1]:.4f}")

    # ---- Evaluasi pada data test ---------------------------------------------
    for r in results:
        m = summarize_metrics(y_te, preds[r["model"]])
        r.update({"test_accuracy": m["accuracy"], "precision": m["precision"],
                  "recall": m["recall"], "f1": m["f1"],
                  "TN": m["TN"], "FP": m["FP"], "FN": m["FN"], "TP": m["TP"],
                  "nested_cv_mean": nested[r["model"]][0], "nested_cv_std": nested[r["model"]][1]})
    table = pd.DataFrame(results)
    table.to_csv(RESULT_DIR / "q2a_model_comparison.csv", index=False)
    print("\n=== HASIL DATA TEST ===")
    print(table[["model", "cv_accuracy_train", "test_accuracy", "precision", "recall", "f1",
                 "nested_cv_mean", "nested_cv_std"]].round(4).to_string(index=False))

    # ---- Grafik -------------------------------------------------------------
    order = ["Perceptron (default)", "Perceptron (Averaged, tuned)",
             "k-NN (default, k=5)", "k-NN (tuned)"]
    plot_confusion_grid(
        [(f"{n}\nTest accuracy {table.set_index('model').loc[n, 'test_accuracy']:.2%}",
          y_te, preds[n]) for n in order],
        CLASS_NAMES, RESULT_DIR / "q2a_confusion_matrices.png", labels=[0, 1])
    for n, fname in [("Perceptron (Averaged, tuned)", "q2a_cm_perceptron.png"),
                     ("k-NN (tuned)", "q2a_cm_knn.png")]:
        plot_confusion_matrix(y_te, preds[n], CLASS_NAMES, n, RESULT_DIR / fname, labels=[0, 1])

    t = table.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(order))
    w = 0.38
    b1 = ax.bar(xs - w / 2, t["nested_cv_mean"] * 100, w, yerr=t["nested_cv_std"] * 100,
                color=C_PRIMARY, label="Nested CV (mean, all data)", capsize=3)
    b2 = ax.bar(xs + w / 2, t["test_accuracy"] * 100, w, color=C_ACCENT, label="Test accuracy")
    ax.set_xticks(xs, [o.replace(" (", "\n(") for o in order])
    ax.set_ylim(60, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Perceptron vs k-NN: default vs tuned")
    ax.legend(loc="lower right", frameon=False)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}",
                    ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "q2a_model_comparison.png")
    plt.close(fig)

    # Scatter data + decision boundary perceptron (proyeksi JamBelajar vs TugasSelesai)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    rng = np.random.default_rng(RANDOM_STATE)
    jit = lambda s: s + rng.normal(0, 0.12, len(s))  # noqa: E731  (jitter: nilai bulat bertumpuk)
    for lab, c, nm in [(0, C_GREY, "Not Graduated"), (1, C_GOOD, "Graduated")]:
        sub = df[df["label"] == lab]
        ax.scatter(jit(sub["JamBelajar"]), jit(sub["TugasSelesai"]), s=14, alpha=0.55, c=c, label=nm)
    ax.set_xlabel("Study hours (JamBelajar)")
    ax.set_ylabel("Assignments completed (TugasSelesai)")
    ax.set_title("Data distribution: study hours vs assignments")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "q2a_data_scatter.png")
    plt.close(fig)

    # Bobot Perceptron (interpretasi pentingnya fitur; dalam skala standar)
    coef = fitted["Perceptron (Averaged, tuned)"].named_steps["clf"].coef_[0]
    importance = dict(zip(FEATURES, [float(c) for c in coef]))
    print("\nBobot Perceptron (fitur terstandar):", importance)

    # ---- Simpan -------------------------------------------------------------
    joblib.dump({"perceptron": fitted["Perceptron (Averaged, tuned)"],
                 "knn": fitted["k-NN (tuned)"],
                 "features": FEATURES, "class_names": CLASS_NAMES},
                MODEL_DIR / "supervised_models.joblib")
    save_json({"results": results, "perceptron_weights_standardized": importance,
               "data_rows": len(df), "random_state": RANDOM_STATE},
              RESULT_DIR / "q2a_metrics.json")
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik.")


if __name__ == "__main__":
    main()
