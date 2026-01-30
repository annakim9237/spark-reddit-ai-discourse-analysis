#!/usr/bin/env python3
"""
Binary Visualization — saves ALL plots automatically.

Run inside:
    code/Anna-ML/

Images saved to:
    ../data/plots/ML_Anna/ML_Score/{VERSION}/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# =======================================================
# 1) Set version here
# =======================================================
VERSION = "with_neoliberal"
#VERSION = "without_neoliberal"


# =======================================================
# 2) Paths
# =======================================================
BASE_DIR = (
    Path(__file__).resolve().parents[2] /
    "data" / "csv" / "ML_Anna" / "ML_binary"
)

SAVE_DIR = (
    Path(__file__).resolve().parents[2] /
    "data" / "plots" / "ML_Anna" / "ML_binary" / VERSION
)

SAVE_DIR.mkdir(parents=True, exist_ok=True)


conf_path    = BASE_DIR / f"ml_binary_{VERSION}_confusion_matrix.csv"
label_path   = BASE_DIR / f"ml_binary_{VERSION}_label_dist.csv"
metrics_path = BASE_DIR / f"ml_binary_{VERSION}_metrics.csv"
word_path    = BASE_DIR / f"ml_full_{VERSION}_word_coefs.csv"


# =======================================================
# PLOT FUNCTIONS
# =======================================================

# 1) Confusion Matrix
def plot_confusion_matrix():
    df = pd.read_csv(conf_path)
    pivot = df.pivot(index="label_binary", columns="prediction", values="count")

    plt.figure(figsize=(6,5))
    sns.heatmap(pivot, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix — {VERSION}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"confusion_matrix_{VERSION}.png", dpi=300)
    plt.close()


# 2) Label Distribution
def plot_label_distribution():
    df = pd.read_csv(label_path)

    plt.figure(figsize=(5,4))
    plt.bar(df["label_binary"], df["count"], color=["#4DAAF5", "#E74C3C"])
    plt.xticks([0,1], ["Non-viral (0)", "Viral (1)"])
    plt.title(f"Label Distribution — {VERSION}")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"label_distribution_{VERSION}.png", dpi=300)
    plt.close()


# 3) Metrics Plot (Accuracy, F1, ROC-AUC, PR-AUC)
def plot_metrics():
    df = pd.read_csv(metrics_path)

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x="metric", y="value", data=df)
    plt.title(f"Performance Metrics — {VERSION}")
    plt.xticks(rotation=25)

    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2,
            height,
            f"{height:.3f}",         
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"metrics_{VERSION}.png", dpi=300)
    plt.close()



# 4) Top positive + negative word coefficients
def plot_word_coefs(top_n=15):
    df = pd.read_csv(word_path)

    # column name can be term or word
    term = "word" if "word" in df.columns else "term"

    top_pos = df.nlargest(top_n, "coef")
    top_neg = df.nsmallest(top_n, "coef")

    # Positive (viral boosting)
    plt.figure(figsize=(8,6))
    sns.barplot(x="coef", y=term, data=top_pos, palette="Reds")
    plt.title(f"Top Words Increasing Viral Probability — {VERSION}")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"top_positive_words_{VERSION}.png", dpi=300)
    plt.close()

    # Negative (viral suppressing)
    plt.figure(figsize=(8,6))
    sns.barplot(x="coef", y=term, data=top_neg, palette="Blues")
    plt.title(f"Top Words Decreasing Viral Probability — {VERSION}")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"top_negative_words_{VERSION}.png", dpi=300)
    plt.close()


# =======================================================
if __name__ == "__main__":
    print(f"Running Binary Visualization — VERSION = {VERSION}")
    print(f"Saving to → {SAVE_DIR}")

    plot_confusion_matrix()
    plot_label_distribution()
    plot_metrics()
    plot_word_coefs(top_n=20)

    print("\n✨ All plots saved successfully!")
    print(f"Check here:  data/plots/ML_Anna/ML_binary/{VERSION}/")
