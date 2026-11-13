"""
SOAL 1 - Analisis UNCERTAINTY (ketidakpastian) model Naive Bayes
================================================================
Naive Bayes menghasilkan probabilitas posterior P(spam | pesan). Skrip ini menilai
seberapa dapat dipercaya probabilitas tersebut:
  (a) distribusi probabilitas untuk pesan spam vs ham
  (b) reliability diagram + Brier score + ECE (kalibrasi)
  (c) strategi 'abstain': prediksi yakin otomatis, prediksi ragu dikirim ke manusia
  (d) daftar pesan yang salah diklasifikasi beserta probabilitasnya

Jalankan SETELAH q1_spam_naive_bayes.py:   python q1_uncertainty.py
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from common import (C_ACCENT, C_BAD, C_GOOD, C_GREY, C_PRIMARY, MODEL_DIR, RANDOM_STATE, RESULT_DIR,
                    TEST_SIZE, save_json)
from q1_spam_naive_bayes import load_data


def main():
    df, _ = load_data()
    _, X_te, _, y_te = train_test_split(df["Message"], df["label"], test_size=TEST_SIZE,
                                        random_state=RANDOM_STATE, stratify=df["label"])
    model = joblib.load(MODEL_DIR / "spam_nb.joblib")
    p = model.predict_proba(X_te)[:, 1]
    y = y_te.values
    pred = (p >= 0.5).astype(int)

    # ---- kalibrasi ------------------------------------------------------------
    brier = float(brier_score_loss(y, p))
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    rel, ece = [], 0.0
    for b in range(10):
        m = idx == b
        if m.sum():
            rel.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
            ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())

    # ---- abstain: hanya otomatis jika yakin ---------------------------------------
    conf = np.maximum(p, 1 - p)
    bands = {}
    for thr in (0.80, 0.90, 0.95, 0.99):
        m = conf >= thr
        bands[f"confidence>={thr}"] = {
            "coverage": float(m.mean()), "n_auto": int(m.sum()),
            "accuracy_auto": float((pred[m] == y[m]).mean()),
            "n_to_human": int((~m).sum()),
            "errors_remaining": int((pred[m] != y[m]).sum()),
        }
    wrong = np.where(pred != y)[0]
    mistakes = [{"true": "spam" if y[i] else "ham", "p_spam": round(float(p[i]), 4),
                 "text": X_te.iloc[i][:110]} for i in wrong]

    # ---- grafik ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    ax.hist(p[y == 0], bins=bins, alpha=0.8, color=C_PRIMARY, label="actual ham")
    ax.hist(p[y == 1], bins=bins, alpha=0.8, color=C_ACCENT, label="actual spam")
    ax.set_yscale("log")
    ax.axvline(0.5, color=C_GREY, ls="--")
    ax.set_xlabel("Predicted P(spam)")
    ax.set_ylabel("Number of test messages (log scale)")
    ax.set_title("Posterior probabilities are extreme")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], "--", color=C_GREY, label="perfect calibration")
    ax.plot([r[0] for r in rel], [r[1] for r in rel], "o-", color=C_PRIMARY, label="Naive Bayes")
    for r in rel:
        ax.annotate(f"n={r[2]}", (r[0], r[1]), textcoords="offset points", xytext=(4, -11), fontsize=7)
    ax.set_xlabel("Mean predicted P(spam)")
    ax.set_ylabel("Observed fraction of spam")
    ax.set_title(f"Reliability diagram (Brier={brier:.4f}, ECE={ece:.4f})")
    ax.legend(frameon=False)

    ax = axes[2]
    thr = [float(k.split(">=")[1]) for k in bands]
    ax.plot(thr, [v["accuracy_auto"] * 100 for v in bands.values()], "o-", color=C_GOOD, label="accuracy of auto-decisions")
    ax.set_ylabel("Accuracy of auto-decisions (%)", color=C_GOOD)
    ax.set_xlabel("Confidence threshold  max(P(spam), P(ham))")
    ax2 = ax.twinx()
    ax2.plot(thr, [v["coverage"] * 100 for v in bands.values()], "s--", color=C_BAD, label="coverage")
    ax2.set_ylabel("Messages decided automatically (%)", color=C_BAD)
    ax.set_title("Abstain strategy: send uncertain cases to a human")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "q1_uncertainty.png")
    plt.close(fig)

    save_json({"brier_score": brier, "ece": float(ece), "reliability_bins": rel,
               "abstain_analysis": bands, "misclassified": mistakes,
               "share_extreme_predictions": float(((p < 0.01) | (p > 0.99)).mean())},
              RESULT_DIR / "q1_uncertainty.json")
    print(f"Brier={brier:.4f}  ECE={ece:.4f}  extreme(<1% or >99%)={((p < 0.01) | (p > 0.99)).mean():.1%}")
    for k, v in bands.items():
        print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
    print("\nMisclassified:")
    for m in mistakes:
        print(f"  true={m['true']:4s} p_spam={m['p_spam']:.3f}  {m['text']}")


if __name__ == "__main__":
    main()
