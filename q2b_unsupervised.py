"""
SOAL 2B (CLO-1) - Unsupervised Learning: Segmentasi Pelanggan dengan K-Means
============================================================================
Fitur : Umur, Pembelian_per_bulan
Syarat: jumlah klaster n = 3 sampai 5

Langkah:
  1. Standardisasi fitur. WAJIB: Pembelian bernilai ratusan ribu - jutaan,
     Umur hanya 18-64. Tanpa standardisasi, jarak K-Means praktis hanya
     ditentukan oleh Pembelian dan Umur diabaikan.
  2. Jalankan K-Means untuk k = 2..7 (rentang 3-5 = kandidat sah; sisanya
     sebagai pembanding), hitung: inertia (elbow), silhouette, Davies-Bouldin,
     Calinski-Harabasz.
  3. Pilih k dalam rentang 3-5 dengan silhouette tertinggi (didukung
     Davies-Bouldin terendah).
  4. Profilkan tiap klaster -> interpretasi bisnis.

Jalankan:  python q2b_unsupervised.py
"""
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from common import (
    C_ACCENT, C_GOOD, C_GREY, C_PRIMARY, MODEL_DIR, RANDOM_STATE, RESULT_DIR, data_path, save_json,
)

warnings.filterwarnings("ignore")
FEATURES = ["Umur", "Pembelian_per_bulan"]
PALETTE = [C_PRIMARY, C_ACCENT, C_GOOD, "#8E5EA2", C_GREY]


def rupiah(x: float) -> str:
    return f"IDR {x / 1_000:,.0f}k" if x < 1_000_000 else f"IDR {x / 1_000_000:,.2f}M"


def name_clusters(profile: pd.DataFrame) -> dict:
    """Beri nama segmen otomatis berdasarkan peringkat pengeluaran & usia."""
    names = {}
    spend_rank = profile["Pembelian_mean"].rank(method="first").astype(int)  # 1 = terendah
    k = len(profile)
    for cid in profile.index:
        age = profile.loc[cid, "Umur_mean"]
        age_lbl = "Young" if age < 30 else ("Adult" if age < 45 else "Senior")
        r = spend_rank[cid]
        if r == 1:
            tier = "Budget"
        elif r == k:
            tier = "Premium"
        else:
            tier = "Mid-range" if k <= 3 else f"Mid-{'low' if r <= (k + 1) / 2 else 'high'}"
        names[int(cid)] = f"{tier} - {age_lbl}"
    return names


def main():
    df = pd.read_csv(data_path("dataset_unsupervised.csv"))
    print(f"Jumlah baris di file: {len(df)}  (catatan: soal menyebut 1000 sampel)")
    print(df.describe().round(1).to_string(), "\n")

    scaler = StandardScaler()
    Z = scaler.fit_transform(df[FEATURES])

    # ---- Evaluasi k = 2..7 ---------------------------------------------------
    rows = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit(Z)
        rows.append({
            "k": k, "inertia": km.inertia_,
            "silhouette": silhouette_score(Z, km.labels_),
            "davies_bouldin": davies_bouldin_score(Z, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(Z, km.labels_),
            "in_allowed_range": 3 <= k <= 5,
        })
    ev = pd.DataFrame(rows)
    ev.to_csv(RESULT_DIR / "q2b_k_evaluation.csv", index=False)
    print(ev.round(4).to_string(index=False))

    allowed = ev[ev["in_allowed_range"]]
    best_k = int(allowed.sort_values("silhouette", ascending=False).iloc[0]["k"])
    print(f"\n>>> k terpilih (rentang 3-5, silhouette tertinggi): {best_k}")

    # ---- Grafik evaluasi ------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, title in zip(axes, ["inertia", "silhouette", "davies_bouldin"],
                              ["Elbow (Inertia)", "Silhouette (higher is better)",
                               "Davies-Bouldin (lower is better)"]):
        ax.plot(ev["k"], ev[col], "o-", color=C_PRIMARY)
        ax.axvspan(2.5, 5.5, color=C_ACCENT, alpha=0.08)
        ax.scatter([best_k], [ev.loc[ev["k"] == best_k, col]], s=130, color=C_ACCENT, zorder=5)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_title(title, fontsize=10)
    fig.suptitle("Choosing k (orange area = range allowed by the exam)", y=1.04)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "q2b_k_selection.png")
    plt.close(fig)

    # ---- Model final ------------------------------------------------------------
    km = KMeans(n_clusters=best_k, n_init=50, random_state=RANDOM_STATE).fit(Z)
    df["cluster"] = km.labels_
    sil_samples = silhouette_samples(Z, km.labels_)

    profile = df.groupby("cluster").agg(
        jumlah=("Umur", "size"),
        Umur_mean=("Umur", "mean"), Umur_min=("Umur", "min"), Umur_max=("Umur", "max"),
        Pembelian_mean=("Pembelian_per_bulan", "mean"),
        Pembelian_min=("Pembelian_per_bulan", "min"),
        Pembelian_max=("Pembelian_per_bulan", "max"),
    )
    profile["silhouette"] = pd.Series(sil_samples, index=df.index).groupby(df["cluster"]).mean()
    names = name_clusters(profile)
    # urutkan klaster dari pengeluaran terendah -> tertinggi agar rapi
    order = profile.sort_values("Pembelian_mean").index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    df["cluster"] = df["cluster"].map(remap)
    profile = profile.loc[order].reset_index(drop=True)
    names = {remap[o]: names[o] for o in order}
    profile["nama_segmen"] = profile.index.map(names)
    profile["rentang_umur"] = profile.apply(lambda r: f"{int(r.Umur_min)}-{int(r.Umur_max)} yrs", axis=1)
    profile["rentang_belanja"] = profile.apply(
        lambda r: f"{rupiah(r.Pembelian_min)} - {rupiah(r.Pembelian_max)}", axis=1)
    profile.round(2).to_csv(RESULT_DIR / "q2b_cluster_profile.csv")
    df.to_csv(RESULT_DIR / "q2b_clustered_data.csv", index=False)
    print("\n=== PROFIL KLASTER ===")
    print(profile[["nama_segmen", "jumlah", "rentang_umur", "rentang_belanja", "silhouette"]]
          .round(3).to_string())

    # ---- Scatter plot klaster ---------------------------------------------------
    centers = scaler.inverse_transform(km.cluster_centers_)
    centers = centers[order]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for cid in range(best_k):
        sub = df[df["cluster"] == cid]
        ax.scatter(sub["Umur"], sub["Pembelian_per_bulan"] / 1e6, s=18, alpha=0.65,
                   color=PALETTE[cid], label=f"C{cid}: {names[cid]} (n={len(sub)})")
    ax.scatter(centers[:, 0], centers[:, 1] / 1e6, marker="X", s=240, c="black",
               edgecolor="white", linewidth=1.5, label="Centroid", zorder=6)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Monthly spending (million IDR)")
    ax.set_title(f"K-Means result (k={best_k}), silhouette {silhouette_score(Z, km.labels_):.3f}")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "q2b_clusters.png")
    plt.close(fig)

    # ---- Simpan ----------------------------------------------------------------
    # urutan label model disusun ulang agar cluster 0 = belanja terendah ... dst.
    remap_arr = np.array([remap[i] for i in range(best_k)])
    joblib.dump({"scaler": scaler, "kmeans": km, "remap": remap_arr, "names": names,
                 "features": FEATURES, "profile": profile, "best_k": best_k},
                MODEL_DIR / "kmeans_model.joblib")
    save_json({"best_k": best_k, "rows_in_file": len(df),
               "evaluation": ev.to_dict(orient="records"),
               "profile": profile.reset_index().to_dict(orient="records")},
              RESULT_DIR / "q2b_metrics.json")
    print("\nSelesai. Model -> models/kmeans_model.joblib")


if __name__ == "__main__":
    main()
