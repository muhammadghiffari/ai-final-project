"""Melatih ulang SEMUA model secara berurutan (opsional - model hasil latih sudah tersedia di folder models/)."""
import subprocess
import sys

STEPS = [
    "q1_spam_naive_bayes.py",   # ~4 menit
    "q1_uncertainty.py",        # detik
    "q2b_unsupervised.py",      # detik
    "q2a_supervised.py",        # ~3-6 menit (nested cross-validation)
    "q3_neural_network.py",     # 12 jaringan: ~30-60 menit di CPU (bisa dilanjutkan jika terputus)
]
for step in STEPS:
    print(f"\n{'=' * 70}\n>>> {step}\n{'=' * 70}", flush=True)
    code = subprocess.call([sys.executable, step])
    if code != 0:
        sys.exit(f"Langkah {step} gagal (kode {code}).")
print("\nSelesai. Jalankan antarmuka dengan:  python app.py")
