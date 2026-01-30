#!/usr/bin/env python3
"""
Score Regression Visualization (Lasso)

- Plots:
  * Top positive / negative word coefficients
  * Topic coefficients
  * Metadata feature coefficients

Run this from: code/Anna-ML/

Images will be saved to:
  ../data/plots/ML_Anna/ML_score/{VERSION}/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =======================================================
# 1) Config
# =======================================================

#VERSION = "with_neoliberal"
VERSION = "without_neoliberal"

project_root = Path(__file__).resolve().parents[2]

BASE_DIR = project_root / "data" / "csv" / "ML_Anna" / "ML_score"
SAVE_DIR = project_root / "data" / "plots" / "ML_Anna" / "ML_score" / VERSION
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# File mapping (with vs without)
if VERSION == "with_neoliberal":
    prefix = "ml_full"
else:  # "without_neoliberal"
    prefix = "ml_full_without_neoliberal"

word_path  = BASE_DIR / f"{prefix}_word_coefs.csv"
topic_path = BASE_DIR / f"{prefix}_topic_coefs.csv"
meta_path  = BASE_DIR / f"{prefix}_meta_coefs.csv"


# =======================================================
# 2) Plot functions
# =======================================================

def plot_word_coefs(top_n=20):
    df = pd.read_csv(word_path)

    term_col = "word" if "word" in df.columns else "term"

    top_pos = df.nlargest(top_n, "coef")
    top_neg = df.nsmallest(top_n, "coef")

    plt.figure(figsize=(8, 6))
    sns.barplot(x="coef", y=term_col, data=top_pos)
    plt.title(f"Top {top_n} Words Increasing Score — {VERSION}")
    plt.xlabel("Coefficient (Lasso)")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"top_positive_words_{VERSION}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.barplot(x="coef", y=term_col, data=top_neg)
    plt.title(f"Top {top_n} Words Decreasing Score — {VERSION}")
    plt.xlabel("Coefficient (Lasso)")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"top_negative_words_{VERSION}.png", dpi=300)
    plt.close()


def plot_topic_coefs():
    df = pd.read_csv(topic_path)   # columns: topic_id, coef

    df_sorted = df.sort_values("coef", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="topic_id", y="coef", data=df_sorted)
    plt.title(f"Topic Coefficients — {VERSION}")
    plt.xlabel("Topic ID")
    plt.ylabel("Coefficient")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"topic_coefs_{VERSION}.png", dpi=300)
    plt.close()


def plot_meta_coefs():
    df = pd.read_csv(meta_path)   # columns: feature_name, coef

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x="feature_name", y="coef", data=df)
    plt.title(f"Metadata Feature Coefficients — {VERSION}")
    plt.xlabel("Feature")
    plt.ylabel("Coefficient")

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
    plt.savefig(SAVE_DIR / f"meta_coefs_{VERSION}.png", dpi=300)
    plt.close()


# =======================================================
# 3) Main
# =======================================================

if __name__ == "__main__":
    print(f"Score Regression Visualization — VERSION = {VERSION}")
    print(f"  CSV from : {BASE_DIR}")
    print(f"  Plots to : {SAVE_DIR}")

    plot_word_coefs(top_n=20)
    plot_topic_coefs()
    plot_meta_coefs()

    print("\n✨ All score plots saved!")
