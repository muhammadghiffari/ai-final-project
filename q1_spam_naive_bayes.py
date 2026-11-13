"""
SOAL 1 (CLO-3) - Klasifikasi Spam dengan Naive Bayes
====================================================
Alur kerja:
  1. Load & bersihkan data (hapus pesan duplikat -> mencegah data leakage)
  2. Split 80/20 stratified (proporsi spam/ham terjaga)
  3. Bandingkan beberapa konfigurasi Naive Bayes + representasi teks,
     masing-masing di-tuning dengan GridSearchCV (5-fold) HANYA pada data train
  4. Pilih model terbaik berdasarkan CV accuracy (bukan berdasarkan data test)
  5. Evaluasi final pada data test: confusion matrix, accuracy, precision,
     recall, F1
  6. Simpan model + grafik + metrik untuk dipakai antarmuka Gradio

Jalankan:  python q1_spam_naive_bayes.py
"""
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from common import (
    C_ACCENT, C_GOOD, C_PRIMARY, MODEL_DIR, RANDOM_STATE, RESULT_DIR, TEST_SIZE,
    data_path, plot_confusion_matrix, save_json, summarize_metrics,
)

warnings.filterwarnings("ignore")
CLASS_NAMES = ["Ham (not spam)", "Spam"]


def load_data() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(data_path("mail_data.csv"))
    df.columns = ["Category", "Message"]
    n_raw = len(df)
    df = df.dropna().copy()
    df["Message"] = df["Message"].astype(str).str.strip()
    # Pesan identik di train & test = kebocoran data (model "menghafal"),
    # sehingga duplikat dibuang sebelum split.
    df = df.drop_duplicates(subset="Message").reset_index(drop=True)
    df["label"] = (df["Category"].str.lower() == "spam").astype(int)
    info = {
        "n_raw": n_raw,
        "n_duplicates_removed": n_raw - len(df),
        "n_clean": len(df),
        "n_spam": int(df["label"].sum()),
        "n_ham": int((df["label"] == 0).sum()),
    }
    return df, info


def build_candidates():
    """Daftar (pipeline, grid hyperparameter) yang dibandingkan."""
    cv_word = lambda: CountVectorizer(lowercase=True)  # noqa: E731
    tf_word = lambda: TfidfVectorizer(lowercase=True)  # noqa: E731
    tf_char = lambda: TfidfVectorizer(analyzer="char_wb", min_df=2)  # noqa: E731
    word_char = lambda: FeatureUnion(  # noqa: E731
        [
            ("w", TfidfVectorizer(ngram_range=(1, 2))),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
        ]
    )
    return {
        "Count-Word + MultinomialNB": (
            Pipeline([("vec", cv_word()), ("nb", MultinomialNB())]),
            {"vec__ngram_range": [(1, 1), (1, 2)], "vec__min_df": [1, 2],
             "nb__alpha": [0.03, 0.1, 0.3, 1.0]},
        ),
        "TFIDF-Word + MultinomialNB": (
            Pipeline([("vec", tf_word()), ("nb", MultinomialNB())]),
            {"vec__ngram_range": [(1, 1), (1, 2), (1, 3)], "vec__sublinear_tf": [True, False],
             "nb__alpha": [0.01, 0.03, 0.1, 0.3]},
        ),
        "TFIDF-Char + MultinomialNB": (
            Pipeline([("vec", tf_char()), ("nb", MultinomialNB())]),
            {"vec__ngram_range": [(2, 4), (2, 5)], "nb__alpha": [0.01, 0.03, 0.05, 0.1, 0.3]},
        ),
        "TFIDF-Word+Char + MultinomialNB": (
            Pipeline([("vec", word_char()), ("nb", MultinomialNB())]),
            {"nb__alpha": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]},
        ),
        "TFIDF-Word + ComplementNB": (
            Pipeline([("vec", tf_word()), ("nb", ComplementNB())]),
            {"vec__ngram_range": [(1, 1), (1, 2), (1, 3)], "nb__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]},
        ),
        "Binary-Word + BernoulliNB": (
            Pipeline([("vec", CountVectorizer(binary=True)), ("nb", BernoulliNB())]),
            {"vec__ngram_range": [(1, 1), (1, 2)], "nb__alpha": [0.01, 0.03, 0.1, 0.5]},
        ),
    }


def plot_top_words(texts, labels, path, top_n=15):
    """Kata paling 'berbau spam' vs 'berbau ham' (rasio log-probabilitas NB)."""
    vec = CountVectorizer(min_df=5, stop_words="english")
    X = vec.fit_transform(texts)
    nb = MultinomialNB(alpha=1.0).fit(X, labels)
    ratio = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    words = np.array(vec.get_feature_names_out())
    order = np.argsort(ratio)
    spam_w, spam_s = words[order[-top_n:]][::-1], ratio[order[-top_n:]][::-1]
    ham_w, ham_s = words[order[:top_n]], ratio[order[:top_n]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].barh(spam_w[::-1], spam_s[::-1], color=C_ACCENT)
    axes[0].set_title("Top SPAM indicator words")
    axes[1].barh(ham_w[::-1], -ham_s[::-1], color=C_PRIMARY)
    axes[1].set_title("Top HAM indicator words")
    for ax in axes:
        ax.set_xlabel("log P(word|class) - log P(word|other class)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"spam": list(spam_w), "ham": list(ham_w)}


def main():
    t0 = time.time()
    df, info = load_data()
    print("=== DATA ===")
    print(info)

    X_train, X_test, y_train, y_test = train_test_split(
        df["Message"], df["label"], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=df["label"],
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)} "
          f"| spam train {y_train.mean():.1%}, spam test {y_test.mean():.1%}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}

    # ---- Baseline: Naive Bayes "textbook" tanpa tuning -----------------------
    base = Pipeline([("vec", CountVectorizer()), ("nb", MultinomialNB())]).fit(X_train, y_train)
    pred = base.predict(X_test)
    m = summarize_metrics(y_test, pred)
    rows.append({"model": "BASELINE: Count + MultinomialNB (default)", "cv_accuracy": np.nan,
                 "test_accuracy": m["accuracy"], "precision": m["precision"],
                 "recall": m["recall"], "f1": m["f1"], "best_params": "default"})
    baseline_metrics = m

    # ---- Tuning semua kandidat ----------------------------------------------
    print("\n=== GRID SEARCH (5-fold CV pada data train) ===")
    for name, (pipe, grid) in build_candidates().items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=1, refit=True)
        gs.fit(X_train, y_train)
        pred = gs.predict(X_test)
        m = summarize_metrics(y_test, pred)
        fitted[name] = gs
        rows.append({"model": name, "cv_accuracy": gs.best_score_,
                     "test_accuracy": m["accuracy"], "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1"], "best_params": str(gs.best_params_)})
        print(f"{name:34s} CV={gs.best_score_:.4f} | Test acc={m['accuracy']:.4f} "
              f"F1={m['f1']:.4f}  {gs.best_params_}")

    table = pd.DataFrame(rows)
    table.to_csv(RESULT_DIR / "q1_model_comparison.csv", index=False)

    # ---- Pilih model terbaik berdasarkan CV (bukan test!) ---------------------
    tuned = table.dropna(subset=["cv_accuracy"])
    best_name = tuned.sort_values(["cv_accuracy", "f1"], ascending=False).iloc[0]["model"]
    best = fitted[best_name].best_estimator_
    print(f"\n>>> Model terpilih (CV accuracy tertinggi): {best_name}")

    # ---- Evaluasi final -----------------------------------------------------
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    metrics = summarize_metrics(y_test, y_pred)
    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    print("\n=== EVALUASI FINAL (DATA TEST) ===")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    print(metrics)

    cm = plot_confusion_matrix(
        y_test, y_pred, CLASS_NAMES,
        f"Confusion Matrix - {best_name}\nAccuracy {metrics['accuracy']:.2%}",
        RESULT_DIR / "q1_confusion_matrix.png", labels=[0, 1],
    )
    (RESULT_DIR / "q1_classification_report.txt").write_text(
        classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4), encoding="utf-8"
    )

    # ---- Grafik perbandingan model -----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.4))
    plot_df = table.copy()
    plot_df["short"] = plot_df["model"].str.replace("BASELINE: ", "BASELINE\n", regex=False)
    colors = [C_ACCENT if m == best_name else C_PRIMARY for m in plot_df["model"]]
    ax.barh(plot_df["short"], plot_df["test_accuracy"] * 100, color=colors)
    ax.set_xlim(95, 100)
    ax.invert_yaxis()
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("Naive Bayes variants (orange = selected via CV)")
    for i, v in enumerate(plot_df["test_accuracy"] * 100):
        ax.text(v + 0.03, i, f"{v:.2f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "q1_model_comparison.png")
    plt.close(fig)

    top_words = plot_top_words(X_train, y_train, RESULT_DIR / "q1_top_words.png")

    # ---- Simpan artefak -----------------------------------------------------
    joblib.dump(best, MODEL_DIR / "spam_nb.joblib")
    save_json(
        {
            "data": info,
            "split": {"train": len(X_train), "test": len(X_test), "random_state": RANDOM_STATE},
            "best_model": best_name,
            "best_params": fitted[best_name].best_params_,
            "cv_accuracy": float(fitted[best_name].best_score_),
            "test_metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "confusion_matrix": cm.tolist(),
            "top_words": top_words,
        },
        RESULT_DIR / "q1_metrics.json",
    )
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik. Model -> models/spam_nb.joblib")


if __name__ == "__main__":
    main()
