"""
run_pipeline.py
Runs the complete Movie Genre Classification pipeline:
  1. Generate data  →  2. Preprocess + EDA  →  3. Train  →  4. Predict demo
"""

import subprocess, sys, os

STEPS = [
    ("generate_data.py", "Step 1/4 — Generating movie dataset"),
    ("preprocess.py",    "Step 2/4 — Preprocessing & EDA"),
    ("train_model.py",   "Step 3/4 — Training TF-IDF + Logistic Regression"),
    ("predict.py",       "Step 4/4 — Demo predictions"),
]

def banner(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)

def run(script, label):
    banner(label)
    r = subprocess.run([sys.executable, script], capture_output=False)
    if r.returncode != 0:
        print(f"\n❌  {script} failed.")
        sys.exit(1)

def main():
    banner("Movie Genre Classification — Pipeline Starting")
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    for script, label in STEPS:
        run(script, label)

    banner("Pipeline complete!")
    print("\nOutputs:")
    print("  data/movies.csv               ← raw dataset (96 movies, 8 genres)")
    print("  data/movies_clean.csv         ← cleaned + features")
    print("  models/genre_classifier.pkl   ← trained pipeline (TF-IDF + LR)")
    print("  models/label_encoder.pkl      ← genre label encoder")
    print("  models/metrics.json           ← accuracy, F1, CV scores")
    print("  outputs/1_eda.png             ← genre distribution & word lengths")
    print("  outputs/2_top_words.png       ← top keywords per genre")
    print("  outputs/3_confusion_matrix.png")
    print("  outputs/4_cv_scores.png       ← cross-validation results")
    print("  outputs/5_per_class_f1.png    ← F1 per genre")
    print("\nTo classify your own movies:")
    print("  python predict.py --interactive")

if __name__ == "__main__":
    main()
