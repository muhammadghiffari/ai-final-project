# Artificial Intelligence Final Project &mdash; Semester 20261

[![Live Web App](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-67e8f9?style=for-the-badge&logo=githubpages&logoColor=black)](https://muhammadghiffari.github.io/ai-final-project/)
[![Python](https://img.shields.io/badge/Python-3.12-38bdf8?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.31.0-ff7c00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-violet?style=for-the-badge)](LICENSE)

An end-to-end artificial intelligence project fulfilling all requirements for the **Artificial Intelligence Final Examination (Semester 20261)**. 

The project delivers four rigorous machine learning pipelines covering text classification, linear and non-parametric supervised prediction, unsupervised customer clustering, and deep multi-layer neural networks. It features a unified **Gradio interface with a modern high-contrast dark glassmorphism design system** and an **interactive static web application deployed on GitHub Pages** for instant access by examiners and peers without local installation.

---

## 👥 Authors & Team Information

- **Study Program:** Informatics
- **Academic Term:** Semester 20261
- **Course:** Artificial Intelligence

| No | Full Name | Student ID (NIM) | Role & Contribution |
|:---|:---|:---|:---|
| 1 | **Muhammad Ghiffari** | `001202500212` | **Lead ML Engineer & System Architect**<br>&bull; End-to-end data preprocessing, deduplication, cleaning, and feature engineering<br>&bull; Model training and optimization across all 4 modules (Naive Bayes, Perceptron, k-NN, K-Means, and 12 Neural Network architectures)<br>&bull; Statistical validation, nested cross-validation protocols, and posterior uncertainty calibration<br>&bull; Full-stack software architecture (Dark-Glass Gradio interface and GitHub Pages live web suite) |
| 2 | **Fhutka Hiban Dairoby** | `001202500055` | **Co-ML Engineer & Evaluation Analyst**<br>&bull; Comparative model benchmarking, hyperparameter analysis, and performance auditing<br>&bull; Diagnostic visualization generation (confusion matrices, learning curves, and heatmaps)<br>&bull; Customer segment profiling, business strategy mapping, and presentation documentation |

---

## 🌐 Live Interactive Web Application

Experience all four models, interactive prediction sandboxes, and visual diagnostic reports directly in your browser:

👉 **[https://muhammadghiffari.github.io/ai-final-project/](https://muhammadghiffari.github.io/ai-final-project/)**

---

## 🔬 Examination Question Breakdown & Methodology

### Question 1: SMS Spam Classification (Multinomial Naive Bayes)
- **Script:** [`q1_spam_naive_bayes.py`](q1_spam_naive_bayes.py), [`q1_uncertainty.py`](q1_uncertainty.py)
- **Model Selected:** `TFIDF-Word+Char + MultinomialNB` (Alpha = 0.1)
- **Data Integrity:** **403 exact duplicates were purged prior to dataset splitting** to avoid test data contamination. Partitioned into an 80/20 stratified split.
- **Key Results:**
  - **Test Accuracy:** **98.93%**
  - **Spam Precision:** **98.35%** (Only 2 false positives out of 904 legitimate test messages)
  - **Spam Recall:** **92.97%**
  - **F1-Score:** **95.58%**
  - **ROC-AUC:** **0.9891**
- **Uncertainty & Calibration:**
  - Brier Score: `0.0089`
  - Expected Calibration Error (ECE): `0.0076`
  - *Analysis:* While probabilities are well-calibrated in aggregate, 95.8% of predictions have extreme probabilities (<1% or >99%). This overconfidence is an innate mathematical byproduct of Naive Bayes' conditional independence assumption.

---

### Question 2-A: Student Graduation Prediction (Supervised Learning)
- **Script:** [`q2a_supervised.py`](q2a_supervised.py)
- **Models Compared:** Perceptron (Default vs. Averaged Tuned) vs. k-Nearest Neighbors (Default k=5 vs. Tuned k=27)
- **Features Used:** Study Hours (`JamBelajar`), Attendance Rate (`Kehadiran`), Assignments Completed (`TugasSelesai`)
- **Key Results:**

| Model Configuration | Test Accuracy | Precision | Recall | F1-Score | Nested 5-Fold CV (Mean &plusmn; Std) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Perceptron (Averaged, Tuned)** | **94.00%** | **92.86%** | **90.28%** | **91.55%** | **94.37% &plusmn; 1.45%** |
| **k-NN (Tuned, k=27, Distance)** | **91.50%** | **88.73%** | **87.50%** | **88.11%** | **92.23% &plusmn; 1.81%** |
| k-NN (Default, k=5) | 90.50% | 88.41% | 84.72% | 86.52% | 91.50% &plusmn; 1.59% |
| Perceptron (Default) | 82.00% | 78.12% | 69.44% | 73.53% | 86.60% &plusmn; 5.79% |

- **Scientific Insight:** The default Perceptron oscillates significantly (&plusmn;5.79% std) because student performance data is not linearly separable. Weight averaging (**Averaged Perceptron**) stabilizes the hyperplane. Feature attribution indicates **Study Hours (+1.95)** and **Assignments Completed (+1.69)** dominate the graduation threshold, while Attendance (+0.53) provides secondary support.

---

### Question 2-B: Customer Segmentation (Unsupervised Learning)
- **Script:** [`q2b_unsupervised.py`](q2b_unsupervised.py)
- **Algorithm:** K-Means Clustering on `StandardScaler` normalized coordinates.
- **Evaluation:** Evaluated across $k \in [2, 7]$. $k = 3$ achieved the highest **Silhouette Score (0.672)** and lowest **Davies-Bouldin Index (0.428)**.
- **Segment Profiles:**

| Cluster ID | Segment Name | Customer Count | Age Range | Average Age | Spending Range (IDR) | Strategic Action |
|:---:|:---|:---:|:---:|:---:|:---|:---|
| **0** | **Budget &ndash; Young** | 300 | 18 &ndash; 29 yrs | 23.8 | 102,368 &ndash; 299,569 | Promotional bundles, student discounts, entry-tier products. |
| **1** | **Mid-range &ndash; Senior** | 307 | 42 &ndash; 64 yrs | 53.9 | 302,050 &ndash; 799,883 | Loyalty reward schemes, recurring subscriptions, cross-selling. |
| **2** | **Premium &ndash; Adult** | 293 | 30 &ndash; 44 yrs | 36.8 | 701,062 &ndash; 1,499,666 | VIP early access, premium concierge service, high-value offerings. |

---

### Question 3: Synoptic Weather Classification (Deep Neural Networks)
- **Script:** [`q3_neural_network.py`](q3_neural_network.py)
- **Dataset:** 121,500 synoptic atmospheric observations across 7 raw meteorological variables.
- **Feature Engineering:** Added 3 engineered variables:
  1. Diurnal temperature amplitude ($\Delta T = T_{\text{max}} - T_{\text{min}}$)
  2. Trigonometric wind orientation: $\sin(\text{Bearing})$ and $\cos(\text{Bearing})$
- **Grid Architecture:** 12 configurations evaluated (1, 2, 3 hidden layers $\times$ 32, 64, 128, 256 neurons).
- **Optimization:** Adam optimizer, mini-batch size 1,024, piecewise learning rate decay ($3\times 10^{-3} \to 1\times 10^{-3} \to 3\times 10^{-4}$) with early stopping on validation loss.
- **Champion Architecture:**
  - **Topology:** `10 -> 256 -> 256 -> 256 -> 4` (3 hidden layers, 256 neurons each)
  - **Total Parameters:** 135,428 weights
  - **Validation Accuracy:** **99.84%**
  - **Test Accuracy:** **99.80%**
  - **Macro Precision / Recall / F1:** **99.79% / 99.72% / 99.76%**
  - *Class Interpretation:* Wind speed (threshold $\approx 15\,\text{km/h}$) and global radiation (threshold $\approx 1.2\,\text{MJ/m}^2$) form the multi-class partition. Deeper and wider networks successfully capture the non-linear solar radiation boundary.

---

## 📊 Summary Benchmark Matrix

```
┌─────────────────────────┬───────────────────────────────┬────────────────────┬───────────────┐
│ Module / Task           │ Model Architecture            │ Validation Metric  │ Test Accuracy │
├─────────────────────────┼───────────────────────────────┼────────────────────┼───────────────┤
│ Q1 Spam Filter          │ TF-IDF + Multinomial NB       │ 99.03% (5-fold CV) │ 98.93%        │
│ Q2-A Graduation Predict │ Averaged Perceptron (Tuned)   │ 94.37% (Nested CV) │ 94.00%        │
│ Q2-A Graduation Predict │ k-NN (k=27, Distance Weighted)│ 92.23% (Nested CV) │ 91.50%        │
│ Q2-B Segmentation       │ K-Means (k=3, Standardized)   │ Silhouette: 0.672  │ Optimal Fit   │
│ Q3 Weather Classify     │ 3-Layer MLP (256 Neurons)     │ 99.84% (Val Acc)   │ 99.80%        │
└─────────────────────────┴───────────────────────────────┴────────────────────┴───────────────┘
```

---

## 🚀 Local Installation & Execution

All models are pre-trained and serialized under the [`models/`](models/) directory. **You do not need to re-train the models to run the application.**

### 1. Clone Repository
```bash
git clone https://github.com/muhammadghiffari/ai-final-project.git
cd ai-final-project
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Gradio Web Application
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`. Use the **Full Screen** button at the top right (or press `F11`) for presentation mode.

*(Optional)* To generate a public temporary shareable link:
```bash
python app.py --share
```

*(Optional)* To retrain all models from scratch:
```bash
python run_all.py
```

---

## 📁 Repository Structure

```
AI_Final_Project/
├── index.html                   # High-fidelity web application (deployed to GitHub Pages)
├── app.py                       # Unified dark-glass Gradio application
├── ui_theme.py                  # Modern dark-glass design system, CSS injection & layout logic
├── common.py                    # Shared metrics, plot styling & path utilities
├── q1_spam_naive_bayes.py       # Q1: Naive Bayes training & evaluation
├── q1_uncertainty.py            # Q1: Posterior uncertainty & calibration analysis
├── q2a_supervised.py            # Q2-A: Perceptron vs. k-NN comparative benchmark
├── q2b_unsupervised.py          # Q2-B: K-Means clustering & silhouette evaluation
├── q3_neural_network.py         # Q3: 12 Neural Network architectures training
├── run_all.py                   # Automated pipeline runner
├── requirements.txt             # Python package dependencies
├── data/                        # Cleaned CSV datasets
│   ├── mail_data.csv            # 5,572 SMS records
│   ├── dataset_supervised.csv   # 1,000 student academic profiles
│   ├── dataset_unsupervised.csv # 900 customer demographic profiles
│   └── weather.csv              # 121,500 synoptic weather records
├── models/                      # Serialized pre-trained .joblib pipelines
│   ├── spam_nb.joblib
│   ├── supervised_models.joblib
│   ├── kmeans_model.joblib
│   ├── q3_best_pointer.joblib
│   └── nn/                      # 12 trained MLP model checkpoints
└── results/                     # Diagnostic charts, JSON metrics & benchmark CSVs
    ├── q1_confusion_matrix.png
    ├── q1_uncertainty.png
    ├── q2a_confusion_matrices.png
    ├── q2b_clusters.png
    ├── q2b_k_selection.png
    ├── q3_architecture.png
    ├── q3_accuracy_heatmap.png
    ├── q3_learning_curves_loss.png
    └── ...
```

---

## 📜 Academic Integrity & License

This project is developed for academic evaluation in the **Artificial Intelligence** course, Informatics Study Program, Semester 20261. Released under the [MIT License](LICENSE).
