"""
app.py - Unified Gradio interface for the Artificial Intelligence final project
==============================================================================
Run:    python app.py          (then open the URL printed in the terminal,
                                normally http://127.0.0.1:7860)
Option: python app.py --share  (temporary public link, requires internet)

Tabs
  1.  Spam Classifier ............. Naive Bayes            (Question 1)
  2A. Graduation Prediction ....... Perceptron and k-NN    (Question 2 - supervised)
  2B. Customer Segmentation ....... K-Means                (Question 2 - unsupervised)
  3.  Weather Classification ...... Neural Networks        (Question 3)

All models are trained by the q*.py scripts and loaded from /models.
"""
import argparse
import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split

import q3_neural_network as q3
import ui_theme
from common import (  # noqa: F401  (FeatureWeighter must be importable to unpickle k-NN)
    C_ACCENT, C_GOOD, C_GREY, C_PRIMARY, MODEL_DIR, RANDOM_STATE, RESULT_DIR, TEST_SIZE,
    FeatureWeighter, data_path,
)

# ------------------------------- EDIT THESE ---------------------------------
MEMBERS = [
    ("Muhammad Ghiffari", "001202500212"),
    ("Fhutka Hiban Dairoby", "001202500055"),
]
# -----------------------------------------------------------------------------
APP_TITLE = "Artificial Intelligence - Final Project (Semester 20261)"


# ============================== helpers ======================================
def load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_model(path):
    path = Path(path)
    return joblib.load(path) if path.exists() else None


def img(name):
    p = RESULT_DIR / name
    return str(p) if p.exists() else None


def pct(x, nd=2):
    return f"{x * 100:.{nd}f}%"


def need(script):
    raise gr.Error(f"Model not found. Please run:  python {script}")


def table_pct(df, cols):
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = (df[c] * 100).round(2)
    return df


# ============================== Q1: SPAM =====================================
SPAM = load_model(MODEL_DIR / "spam_nb.joblib")
Q1 = load_json(RESULT_DIR / "q1_metrics.json")

SPAM_EXAMPLES = [
    ["WINNER!! As a valued network customer you have been selected to receive a $900 prize reward! "
     "To claim call 09061701461. Claim code KL341. Valid 12 hours only."],
    ["Free entry in 2 a weekly competition to win FA Cup final tickets. Text FA to 87121 to receive entry now!"],
    ["URGENT! Your mobile number has won a 2000 pounds bonus caller prize. Call 09050000327 from landline."],
    ["Hey, are we still meeting for lunch tomorrow at 12? Let me know if you want to change the place."],
    ["Ok, I will call you later. Don't wait up for me tonight."],
    ["Can you send me the notes from yesterday's lecture? I missed the second half."],
]


def predict_spam(text):
    if SPAM is None:
        need("q1_spam_naive_bayes.py")
    if not text or not text.strip():
        raise gr.Error("Please type or paste a message first.")
    p_spam = float(SPAM.predict_proba([text])[0][1])
    verdict = "SPAM" if p_spam >= 0.5 else "NOT SPAM (HAM)"
    md = (f"## Prediction: **{verdict}**\n\nProbability of spam: **{p_spam:.1%}** "
          f"&nbsp;|&nbsp; Probability of ham: **{1 - p_spam:.1%}**")
    return {"Spam": p_spam, "Ham (not spam)": 1 - p_spam}, md


def q1_summary_md():
    if not Q1:
        return "Run `python q1_spam_naive_bayes.py` to generate results."
    t, d = Q1["test_metrics"], Q1["data"]
    return (
        f"**Selected model:** {Q1['best_model']}  \n"
        f"**Best hyper-parameters:** `{Q1['best_params']}`  \n"
        f"**Data:** {d['n_raw']:,} messages, {d['n_duplicates_removed']:,} exact duplicates removed "
        f"-> {d['n_clean']:,} used ({d['n_spam']:,} spam / {d['n_ham']:,} ham). "
        f"Split {Q1['split']['train']:,} train / {Q1['split']['test']:,} test (stratified).\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Accuracy | **{pct(t['accuracy'])}** |\n| Precision (spam) | {pct(t['precision'])} |\n"
        f"| Recall (spam) | {pct(t['recall'])} |\n| F1-score (spam) | {pct(t['f1'])} |\n"
        f"| ROC-AUC | {t['roc_auc']:.4f} |\n"
        f"| TN / FP / FN / TP | {t['TN']} / {t['FP']} / {t['FN']} / {t['TP']} |\n"
    )


def q1_interpretation_md():
    if not Q1:
        return ""
    t, b = Q1["test_metrics"], Q1["baseline_metrics"]
    return (
        f"**Reading the confusion matrix.** Of {t['TN'] + t['FP']} legitimate messages, "
        f"{t['FP']} were wrongly flagged as spam (false positives); of {t['TP'] + t['FN']} spam messages, "
        f"{t['FN']} slipped through (false negatives). In a spam filter a false positive is the costlier "
        f"error (a real message is lost), and this model keeps it very low (precision {pct(t['precision'])}).\n\n"
        f"**Why tuning helped.** The un-tuned textbook baseline (word counts + MultinomialNB, alpha=1) "
        f"reached {pct(b['accuracy'])} accuracy; the tuned model reaches {pct(t['accuracy'])}. "
        f"The selection was made by 5-fold cross-validation on the training data only "
        f"(CV accuracy {pct(Q1['cv_accuracy'])}), so the test set stayed untouched.\n\n"
        f"**Data-quality note.** Exact duplicate messages were removed *before* splitting; otherwise the same "
        f"text could appear in both train and test and inflate the accuracy."
    )


def q1_uncertainty_md():
    U = load_json(RESULT_DIR / "q1_uncertainty.json")
    if not U:
        return "Run `python q1_uncertainty.py` to generate the uncertainty analysis."
    b = U["abstain_analysis"]["confidence>=0.95"]
    n_err = len(U["misclassified"])
    return (
        f"**Uncertainty estimation.** Naive Bayes outputs a posterior probability P(spam | message), which is the "
        f"model's own estimate of its uncertainty. On the test set the probabilities are well calibrated on average "
        f"(Brier score {U['brier_score']:.4f}, expected calibration error {U['ece']:.4f}), **but** "
        f"{pct(U['share_extreme_predictions'], 1)} of all predictions are extreme (below 1% or above 99%). "
        f"This overconfidence is typical of Naive Bayes: its conditional-independence assumption multiplies many "
        f"word probabilities together and pushes the posterior towards 0 or 1.\n\n"
        f"**Consequence for an 'abstain' strategy.** If only decisions with confidence >= 95% were automated, "
        f"{b['n_to_human']} of {b['n_auto'] + b['n_to_human']} messages would go to a human, yet "
        f"{b['errors_remaining']} of the {n_err} errors would remain - most mistakes are made with very high "
        f"confidence (mostly spam messages that look like normal chat). So the probability is useful for ranking, "
        f"but it should not be read as a guarantee of correctness."
    )


# ============================== Q2A: SUPERVISED ===============================
SUP = load_model(MODEL_DIR / "supervised_models.joblib")
Q2A = load_json(RESULT_DIR / "q2a_metrics.json")


def predict_graduation(hours, attendance, assignments):
    if SUP is None:
        need("q2a_supervised.py")
    X = pd.DataFrame([[hours, attendance, assignments]], columns=SUP["features"])
    p_knn = float(SUP["knn"].predict_proba(X)[0][1])
    score = float(SUP["perceptron"].decision_function(X)[0])
    p_per = 1.0 / (1.0 + np.exp(-score))  # sigmoid of the decision score (indicative only)
    pred_per, pred_knn = int(score > 0), int(p_knn >= 0.5)
    names = {1: "GRADUATED", 0: "NOT GRADUATED"}
    agree = "Both models agree." if pred_per == pred_knn else "The two models DISAGREE - borderline case."
    md = (f"| Model | Prediction |\n|---|---|\n| Perceptron (Averaged) | **{names[pred_per]}** |\n"
          f"| k-NN | **{names[pred_knn]}** |\n\n{agree}")
    return ({"Graduated": p_per, "Not graduated": 1 - p_per},
            {"Graduated": p_knn, "Not graduated": 1 - p_knn}, md)


def q2a_table():
    if not Q2A:
        return pd.DataFrame()
    df = pd.DataFrame(Q2A["results"])
    df["nested CV (mean +/- std)"] = [f"{m * 100:.2f} +/- {s * 100:.2f}" for m, s in
                                       zip(df["nested_cv_mean"], df["nested_cv_std"])]
    out = df[["model", "test_accuracy", "precision", "recall", "f1", "nested CV (mean +/- std)"]]
    out = table_pct(out, ["test_accuracy", "precision", "recall", "f1"])
    return out.rename(columns={"model": "Model", "test_accuracy": "Test accuracy (%)",
                               "precision": "Precision (%)", "recall": "Recall (%)", "f1": "F1 (%)"})


def q2a_interpretation_md():
    if not Q2A:
        return "Run `python q2a_supervised.py` to generate results."
    R = {r["model"]: r for r in Q2A["results"]}
    pt, kt = R["Perceptron (Averaged, tuned)"], R["k-NN (tuned)"]
    pd_, kd = R["Perceptron (default)"], R["k-NN (default, k=5)"]
    w = Q2A["perceptron_weights_standardized"]
    rank = sorted(w.items(), key=lambda kv: -abs(kv[1]))
    winner = "Perceptron" if pt["nested_cv_mean"] >= kt["nested_cv_mean"] else "k-NN"
    gap = abs(pt["nested_cv_mean"] - kt["nested_cv_mean"]) * 100
    return (
        f"**Result.** On the held-out test set the tuned Perceptron reached **{pct(pt['test_accuracy'])}** and "
        f"the tuned k-NN **{pct(kt['test_accuracy'])}**. The more reliable estimate is nested cross-validation "
        f"over all 1000 students: Perceptron {pct(pt['nested_cv_mean'])} vs k-NN {pct(kt['nested_cv_mean'])}. "
        f"The {winner} is better by about {gap:.1f} points; with a 200-student test set one split is noisy "
        f"(+/- ~2 points), so the CV figure should be preferred.\n\n"
        f"**Effect of optimisation.** The default Perceptron is unstable ({pct(pd_['test_accuracy'])} test, "
        f"CV std {pd_['nested_cv_std'] * 100:.1f} points) because the classes are not perfectly separable and "
        f"its weights keep oscillating. Averaging the weights (Averaged Perceptron) stabilises it. "
        f"k-NN improves less after tuning ({pct(kd['test_accuracy'])} -> {pct(kt['test_accuracy'])}).\n\n"
        f"**Reading the Perceptron confusion matrix.** It found {pt['TP']} of {pt['TP'] + pt['FN']} students "
        f"who graduated (recall {pct(pt['recall'])}) and {pt['FP']} students were wrongly predicted to "
        f"graduate (precision {pct(pt['precision'])}).\n\n"
        f"**Which features matter.** Perceptron weights on standardised features rank: "
        + ", ".join(f"{k} ({v:+.2f})" for k, v in rank) + ". "
        f"Features with a small weight add little to the decision.\n\n"
        f"**Why a linear model does so well.** The graduation boundary appears close to linear in the "
        f"features, so a linear Perceptron captures it directly, whereas k-NN lets every feature - including "
        f"weak ones - influence the distance and so adds noise."
    )


# ============================== Q2B: K-MEANS ==================================
KM = load_model(MODEL_DIR / "kmeans_model.joblib")
CLUSTER_DATA = (pd.read_csv(RESULT_DIR / "q2b_clustered_data.csv")
                if (RESULT_DIR / "q2b_clustered_data.csv").exists() else None)
SEG_ADVICE = {
    "Budget": "Price-sensitive. Promotions, bundles and entry-level products work best.",
    "Mid-range": "Steady spenders. Loyalty programmes and cross-selling are effective.",
    "Premium": "Highest-value customers. Offer premium products, early access and personalised service.",
}
PALETTE = [C_PRIMARY, C_ACCENT, C_GOOD, "#8E5EA2", C_GREY]


def segment_plot(age=None, spend=None, cid=None):
    fig = Figure(figsize=(7.4, 4.8))
    ax = fig.subplots()
    if CLUSTER_DATA is not None:
        for c in sorted(CLUSTER_DATA["cluster"].unique()):
            s = CLUSTER_DATA[CLUSTER_DATA["cluster"] == c]
            ax.scatter(s["Umur"], s["Pembelian_per_bulan"] / 1e6, s=14, alpha=0.55,
                       color=PALETTE[int(c) % 5], label=KM["names"][int(c)] if KM else f"C{c}")
    if age is not None:
        ax.scatter([age], [spend / 1e6], s=260, marker="*", color="black", edgecolor="white",
                   linewidth=1.2, zorder=6, label="Your customer")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Monthly spending (million IDR)")
    ax.set_title("Customer segments (K-Means)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def predict_segment(age, spend):
    if KM is None:
        need("q2b_unsupervised.py")
    Z = KM["scaler"].transform(pd.DataFrame([[age, spend]], columns=KM["features"]))
    cid = int(KM["remap"][int(KM["kmeans"].predict(Z)[0])])
    name = KM["names"][cid]
    tier = name.split(" - ")[0]
    prof = KM["profile"].loc[cid]
    md = (f"## Segment {cid}: **{name}**\n\n"
          f"Typical members: age {prof['rentang_umur']}, spending {prof['rentang_belanja']} "
          f"({int(prof['jumlah'])} customers in the dataset).\n\n"
          f"**Suggested approach:** {SEG_ADVICE.get(tier, 'Tailor offers to this segment.')}")
    return md, segment_plot(age, spend, cid)


def q2b_profile_table():
    if KM is None:
        return pd.DataFrame()
    p = KM["profile"].copy()
    return pd.DataFrame({
        "Cluster": p.index, "Segment": p["nama_segmen"], "Customers": p["jumlah"].astype(int),
        "Age range": p["rentang_umur"], "Avg age": p["Umur_mean"].round(1),
        "Spending range": p["rentang_belanja"],
        "Avg spending (IDR)": p["Pembelian_mean"].round(0).astype(int),
        "Silhouette": p["silhouette"].round(3),
    })


def q2b_interpretation_md():
    Q = load_json(RESULT_DIR / "q2b_metrics.json")
    if not Q:
        return "Run `python q2b_unsupervised.py` to generate results."
    ev = {r["k"]: r for r in Q["evaluation"]}
    k = Q["best_k"]
    return (
        f"**Choosing k.** K-Means was evaluated for k = 2..7 (the exam allows k = 3 to 5). "
        f"k = {k} has the highest silhouette score ({ev[k]['silhouette']:.3f}) and the lowest "
        f"Davies-Bouldin index ({ev[k]['davies_bouldin']:.3f}) within the allowed range, "
        f"so the data naturally forms {k} groups.\n\n"
        f"**Why scaling is required.** Spending is in the hundreds of thousands to millions while age is "
        f"only 18-64. Without standardisation the distance would be driven almost entirely by spending.\n\n"
        f"**Cluster meanings** are given in the table above: the segments differ mainly by monthly spending, "
        f"and age separates the young low spenders from the older and the working-age groups.\n\n"
        f"**Data note.** The supplied file contains {Q['rows_in_file']} rows (the exam text says 1000)."
    )


# ============================== Q3: NEURAL NETWORK ============================
NN_DIR = MODEL_DIR / "nn"
NN_RUN_DIR = RESULT_DIR / "q3_runs"
NN_SCALER = load_model(NN_DIR / "scaler.joblib")
NN_CACHE = {}


def weather_cache():
    p = RESULT_DIR / "q3_app_cache.json"
    if p.exists():
        return load_json(p)
    try:
        raw = pd.read_csv(data_path("weather.csv")).dropna().drop_duplicates().reset_index(drop=True)
    except Exception:
        return None
    y = raw[q3.TARGET].astype(int).values
    _, te = train_test_split(np.arange(len(raw)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    rng = np.random.default_rng(7)
    examples = []
    for k in range(4):
        i = int(rng.choice(te[y[te] == k]))
        examples.append({"label": k, "values": [float(raw.loc[i, f]) for f in q3.RAW_FEATURES]})
    ranges = {f: [float(raw[f].min()), float(raw[f].max()), float(raw[f].median())] for f in q3.RAW_FEATURES}
    data = {"ranges": ranges, "examples": examples}
    p.write_text(json.dumps(data), encoding="utf-8")
    return data


WC = weather_cache()


def nn_runs():
    runs = {}
    for L in q3.LAYER_OPTIONS:
        for W in q3.WIDTH_OPTIONS:
            r = load_json(NN_RUN_DIR / f"{q3.run_name(L, W)}.json")
            if r and (NN_DIR / f"{q3.run_name(L, W)}.joblib").exists():
                runs[q3.run_name(L, W)] = r
    return runs


NN_RUNS = nn_runs()
Q3 = load_json(RESULT_DIR / "q3_metrics.json")
BEST_NAME = (q3.run_name(Q3["best_model"]["layers"], Q3["best_model"]["width"])
             if Q3 else (max(NN_RUNS, key=lambda n: NN_RUNS[n]["val_accuracy"]) if NN_RUNS else None))


def nn_choices():
    out = []
    for name, r in NN_RUNS.items():
        star = "  [BEST]" if name == BEST_NAME else ""
        out.append((f"{r['layers']} hidden layer(s) x {r['width']} neurons - test acc "
                    f"{r['test']['accuracy'] * 100:.2f}%{star}", name))
    return out


def predict_weather(model_name, pressure, radiation, t_mean, t_min, t_max, wind_speed, wind_bearing):
    if not NN_RUNS or NN_SCALER is None:
        need("q3_neural_network.py")
    if t_min > t_max or not (t_min <= t_mean <= t_max):
        gr.Warning("Inconsistent temperatures: expected min <= mean <= max. Prediction is still shown.")
    if model_name not in NN_CACHE:
        NN_CACHE[model_name] = joblib.load(NN_DIR / f"{model_name}.joblib")
    model = NN_CACHE[model_name]
    row = pd.DataFrame([[pressure, radiation, t_mean, t_min, t_max, wind_speed, wind_bearing]],
                       columns=q3.RAW_FEATURES)
    X = NN_SCALER.transform(q3.build_features(row).values).astype(np.float32)
    proba = model.predict_proba(X)[0]
    top = int(np.argmax(proba))
    r = NN_RUNS[model_name]
    md = (f"## Predicted class: **{top}** - {q3.CLASS_DESC[top]}\n\n"
          f"Confidence: **{proba[top]:.1%}**\n\n"
          f"Model: `{r['architecture']}` ({r['layers']} hidden layer(s), {r['width']} neurons each, "
          f"{r['n_params']:,} parameters), test accuracy {pct(r['test']['accuracy'])}.")
    return {f"Class {k}: {q3.CLASS_DESC[k]}": float(proba[k]) for k in range(4)}, md


def q3_table():
    p = RESULT_DIR / "q3_model_comparison.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = table_pct(df, ["val_accuracy", "test_accuracy", "precision_macro", "recall_macro", "f1_macro"])
    return df.rename(columns={
        "hidden_layers": "Hidden layers", "neurons_per_layer": "Neurons / layer", "architecture": "Architecture",
        "parameters": "Parameters", "epochs": "Epochs", "train_time_s": "Train time (s)",
        "val_accuracy": "Validation acc (%)", "test_accuracy": "Test acc (%)",
        "precision_macro": "Precision (%)", "recall_macro": "Recall (%)", "f1_macro": "F1 (%)"})


def q3_analysis_md():
    p = RESULT_DIR / "q3_model_comparison.csv"
    if not p.exists() or not Q3:
        return "Run `python q3_neural_network.py` to generate results."
    df = pd.read_csv(p)
    b = Q3["best_model"]
    lines = [f"**Best model (chosen by validation accuracy):** `{b['architecture']}` - "
             f"{b['layers']} hidden layer(s) x {b['width']} neurons. Test accuracy "
             f"**{pct(b['test']['accuracy'])}**, macro precision {pct(b['test']['precision'])}, "
             f"macro recall {pct(b['test']['recall'])}, macro F1 {pct(b['test']['f1'])}.", ""]
    lines.append("**Best number of neurons for each depth** (by validation accuracy):")
    for _, r in pd.DataFrame(Q3["best_per_depth"]).iterrows():
        lines.append(f"- {int(r['hidden_layers'])} hidden layer(s): **{int(r['neurons_per_layer'])} neurons** "
                     f"(test accuracy {pct(r['test_accuracy'])})")
    by_w = df.groupby("neurons_per_layer")["test_accuracy"].mean()
    by_d = df.groupby("hidden_layers")["test_accuracy"].mean()
    lines += ["",
              "**Average test accuracy by number of neurons:** " +
              ", ".join(f"{int(k)} -> {pct(v)}" for k, v in by_w.items()) + ".",
              "**Average test accuracy by depth:** " +
              ", ".join(f"{int(k)} layer(s) -> {pct(v)}" for k, v in by_d.items()) + ".", "",
              "**Interpretation.** The wind-related part of the class is learned almost immediately; the harder "
              "part is the radiation-related boundary, which is strongly non-linear. Networks with too few "
              "neurons run out of capacity and plateau early, while wider/deeper networks keep improving. "
              "All 12 models use the same optimiser, batch size and learning-rate schedule, so differences "
              "come from the architecture. Note the cost: larger networks take considerably longer to train."]
    return "\n".join(lines)


def q3_report_text():
    p = RESULT_DIR / "q3_classification_report.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


# ============================== UI ============================================
def landing_metrics():
    """Real numbers (from results/*.json) shown on the landing page."""
    m = {}
    try:
        t = Q1["test_metrics"]
        m.update(spam_acc=t["accuracy"] * 100, spam_prec=t["precision"] * 100, spam_auc=t["roc_auc"])
        R = {r["model"]: r for r in Q2A["results"]}
        pt, kt = R["Perceptron (Averaged, tuned)"], R["k-NN (tuned)"]
        m.update(grad_acc=pt["test_accuracy"] * 100, knn_acc=kt["test_accuracy"] * 100,
                 grad_cv=pt["nested_cv_mean"] * 100)
        qb = load_json(RESULT_DIR / "q2b_metrics.json")
        m.update(segments=qb["best_k"], rows=qb["rows_in_file"])
        b = Q3["best_model"]
        m.update(weather_acc=b["test"]["accuracy"] * 100, wx_layers=b["layers"], wx_width=b["width"])
    except Exception:
        pass  # missing results -> landing page falls back to zeros
    return m


def build_ui():
    with gr.Blocks(title=APP_TITLE, fill_width=True) as demo:
        gr.HTML(ui_theme.header_html(APP_TITLE, MEMBERS))
        with gr.Tabs(elem_classes="nx-tabs-root"):
            with gr.Tab("Home"):
                gr.HTML(ui_theme.landing_html(landing_metrics()))
            # ------------------------------------------------ Q1
            with gr.Tab("1. Spam Classifier (Naive Bayes)"):
                with gr.Tabs():
                    with gr.Tab("Try it"):
                        with gr.Row():
                            with gr.Column():
                                txt = gr.Textbox(label="Message", lines=6,
                                                 placeholder="Paste an SMS / e-mail message here...")
                                btn = gr.Button("Classify", variant="primary")
                                gr.Examples(SPAM_EXAMPLES, inputs=txt, label="Examples (click to load)")
                            with gr.Column():
                                lab = gr.Label(label="Probabilities", num_top_classes=2)
                                out_md = gr.Markdown()
                        btn.click(predict_spam, txt, [lab, out_md], api_name="predict_spam")
                        txt.submit(predict_spam, txt, [lab, out_md], api_name=False)
                    with gr.Tab("Evaluation (confusion matrix)"):
                        gr.Markdown(q1_summary_md())
                        gr.Image(img("q1_confusion_matrix.png"), show_label=False, interactive=False, height=420)
                        gr.Markdown(q1_interpretation_md())
                    with gr.Tab("Uncertainty"):
                        gr.Image(img("q1_uncertainty.png"), show_label=False, interactive=False)
                        gr.Markdown(q1_uncertainty_md())
                    with gr.Tab("Model comparison"):
                        p = RESULT_DIR / "q1_model_comparison.csv"
                        if p.exists():
                            d = pd.read_csv(p)
                            d = table_pct(d, ["cv_accuracy", "test_accuracy", "precision", "recall", "f1"])
                            gr.Dataframe(d, interactive=False, wrap=True)
                        gr.Image(img("q1_model_comparison.png"), show_label=False, interactive=False)
                        gr.Image(img("q1_top_words.png"), show_label=False, interactive=False)

            # ------------------------------------------------ Q2A
            with gr.Tab("2A. Graduation Prediction (Perceptron & k-NN)"):
                with gr.Tabs():
                    with gr.Tab("Try it"):
                        with gr.Row():
                            with gr.Column():
                                s_hours = gr.Slider(0, 10, value=5, step=1, label="Study hours")
                                s_att = gr.Slider(0, 100, value=80, step=1, label="Attendance %")
                                s_asg = gr.Slider(0, 10, value=6, step=1, label="Assignments completed")
                                b2 = gr.Button("Predict", variant="primary")
                                gr.Examples([[2, 60, 2], [4, 75, 5], [7, 90, 9], [5, 55, 6]],
                                            inputs=[s_hours, s_att, s_asg], label="Examples")
                            with gr.Column():
                                l_per = gr.Label(label="Perceptron (score mapped with sigmoid - indicative)",
                                                 num_top_classes=2)
                                l_knn = gr.Label(label="k-NN (neighbour vote)", num_top_classes=2)
                                m2 = gr.Markdown()
                        b2.click(predict_graduation, [s_hours, s_att, s_asg], [l_per, l_knn, m2],
                                 api_name="predict_graduation")
                    with gr.Tab("Evaluation (confusion matrices)"):
                        gr.Markdown("Default vs tuned models. Tuning used 5-fold CV on the training data only.")
                        gr.Image(img("q2a_confusion_matrices.png"), show_label=False, interactive=False)
                        gr.Dataframe(q2a_table(), interactive=False, wrap=True)
                        gr.Image(img("q2a_model_comparison.png"), show_label=False, interactive=False)
                    with gr.Tab("Interpretation"):
                        gr.Markdown(q2a_interpretation_md())

            # ------------------------------------------------ Q2B
            with gr.Tab("2B. Customer Segmentation (K-Means)"):
                with gr.Tabs():
                    with gr.Tab("Try it"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                a_age = gr.Slider(15, 80, value=30, step=1, label="Age (years)")
                                a_sp = gr.Number(value=800_000, label="Monthly spending (IDR)")
                                b3 = gr.Button("Find segment", variant="primary")
                                m3 = gr.Markdown()
                            with gr.Column(scale=2):
                                plot3 = gr.Plot(value=segment_plot() if KM else None, label="Segments")
                        b3.click(predict_segment, [a_age, a_sp], [m3, plot3], api_name="predict_segment")
                    with gr.Tab("Clusters & interpretation"):
                        gr.Image(img("q2b_clusters.png"), show_label=False, interactive=False)
                        gr.Dataframe(q2b_profile_table(), interactive=False, wrap=True)
                        gr.Image(img("q2b_k_selection.png"), show_label=False, interactive=False)
                        gr.Markdown(q2b_interpretation_md())

            # ------------------------------------------------ Q3
            with gr.Tab("3. Weather Classification (Neural Networks)"):
                with gr.Tabs():
                    with gr.Tab("Try it"):
                        choices = nn_choices()
                        with gr.Row():
                            with gr.Column():
                                dd = gr.Dropdown(choices, value=BEST_NAME, label="Neural network model")
                                R = (WC or {}).get("ranges", {})

                                def sl(feat, label, step):
                                    lo, hi, med = R.get(feat, [0, 100, 0])
                                    lo = float(np.floor(lo)); hi = float(np.ceil(hi))
                                    return gr.Slider(lo, hi, value=round(med, 2), step=step, label=label)

                                f_p = sl("Pressure", "Pressure (hPa)", 0.1)
                                f_r = sl("global_radiation", "Global radiation", 0.01)
                                f_tm = sl("temp_mean(c)", "Mean temperature (C)", 0.1)
                                f_tn = sl("temp_min(c)", "Min temperature (C)", 0.1)
                                f_tx = sl("temp_max(c)", "Max temperature (C)", 0.1)
                                f_ws = sl("Wind_Speed", "Wind speed", 0.1)
                                f_wb = sl("Wind_Bearing", "Wind bearing (degrees)", 1)
                                inputs = [f_p, f_r, f_tm, f_tn, f_tx, f_ws, f_wb]
                                b4 = gr.Button("Predict", variant="primary")
                                if WC:
                                    gr.Examples([e["values"] for e in WC["examples"]], inputs=inputs,
                                                label="Real test-set examples (true class 0, 1, 2, 3 in order)")
                            with gr.Column():
                                l4 = gr.Label(label="Class probabilities", num_top_classes=4)
                                m4 = gr.Markdown()
                        b4.click(predict_weather, [dd] + inputs, [l4, m4], api_name="predict_weather")
                    with gr.Tab("Architecture"):
                        gr.Image(img("q3_architecture.png"), show_label=False, interactive=False)
                        gr.Markdown("Input = 7 raw features + 3 engineered features (temperature range, sin/cos of "
                                    "wind bearing). Hidden layers use ReLU, the output layer is a 4-class softmax. "
                                    "Optimiser: Adam, batch size 1024, learning rate 3e-3 -> 1e-3 -> 3e-4 with "
                                    "early stopping.")
                    with gr.Tab("Comparison of 12 models"):
                        gr.Dataframe(q3_table(), interactive=False, wrap=True)
                        with gr.Row():
                            gr.Image(img("q3_accuracy_heatmap.png"), show_label=False, interactive=False)
                            gr.Image(img("q3_metrics_bar.png"), show_label=False, interactive=False)
                        gr.Markdown(q3_analysis_md())
                    with gr.Tab("Learning curves"):
                        gr.Image(img("q3_learning_curves_accuracy.png"), show_label=False, interactive=False)
                        gr.Image(img("q3_learning_curves_loss.png"), show_label=False, interactive=False)
                    with gr.Tab("Confusion matrix (best)"):
                        gr.Image(img("q3_best_confusion_matrix.png"), show_label=False, interactive=False, height=460)
                        gr.Image(img("q3_confusion_best_per_depth.png"), show_label=False, interactive=False)
                        gr.Textbox(q3_report_text(), label="Per-class report (best model)", lines=8,
                                   interactive=False)
                        gr.Markdown("*Class meanings were inferred from data analysis (wind speed threshold ~15, "
                                    "mean global radiation); the dataset itself does not document them.*")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    build_ui().launch(theme=gr.themes.Base(), css=ui_theme.CSS, head=ui_theme.HEAD, favicon_path="favicon.svg", share=args.share, server_port=args.port, inbrowser=True)
