"""
preprocess.py
Cleans movie data, runs EDA, and prepares text for classification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import os
from collections import Counter

os.makedirs("outputs", exist_ok=True)

RAW_FILE       = "data/movies.csv"
CLEAN_FILE     = "data/movies_clean.csv"


# ── Text cleaning ────────────────────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","it","its","he","she","they","we","you","i","this",
    "that","his","her","their","be","as","are","was","were","been","have",
    "has","had","who","which","when","where","while","after","before","into",
    "up","out","about","after","over","into","then","than","so","do","does",
    "did","not","no","but","if","all","each","both","more","also","just",
    "only","even","what","there","him","them","us","will","would","could",
    "should","may","might","can","an","am","through","during","between",
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # keep only letters + spaces
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="id")
    print(f"Loaded {len(df)} movies")

    # Combine title + overview for richer signal
    df["text"] = df["title"] + " " + df["overview"]
    df["clean_text"] = df["text"].apply(clean_text)
    df["text_length"] = df["overview"].apply(lambda x: len(x.split()))

    df.to_csv(CLEAN_FILE, index=True)
    print(f"Cleaned data saved → {CLEAN_FILE}")
    return df


# ── EDA plots ────────────────────────────────────────────────────────────────
def plot_eda(df: pd.DataFrame):
    genre_counts = df["genre"].value_counts()
    genres = genre_counts.index.tolist()
    colors = cm.tab10(np.linspace(0, 1, len(genres)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#F8F8F8")
    fig.suptitle("Movie Genre Dataset — EDA", fontsize=15, fontweight="bold")

    # 1. Bar chart — genre distribution
    ax = axes[0]
    bars = ax.barh(genre_counts.index, genre_counts.values, color=colors)
    ax.set_title("Movies per Genre", fontweight="bold")
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    ax.set_facecolor("white")
    for bar, val in zip(bars, genre_counts.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)

    # 2. Pie chart
    ax = axes[1]
    ax.pie(genre_counts.values, labels=genre_counts.index, colors=colors,
           autopct="%1.0f%%", startangle=140,
           wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.set_title("Genre Share", fontweight="bold")

    # 3. Overview word-count distribution
    ax = axes[2]
    for i, genre in enumerate(genres):
        lengths = df[df["genre"] == genre]["text_length"]
        ax.hist(lengths, bins=10, alpha=0.5, label=genre, color=colors[i])
    ax.set_title("Overview Length Distribution", fontweight="bold")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_facecolor("white")

    plt.tight_layout()
    plt.savefig("outputs/1_eda.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → outputs/1_eda.png")


def plot_top_words(df: pd.DataFrame):
    """Top 15 words per genre."""
    genres = sorted(df["genre"].unique())
    n = len(genres)
    cols = 4
    rows = (n + cols - 1) // cols
    colors = cm.tab10(np.linspace(0, 1, n))

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4), facecolor="#F8F8F8")
    fig.suptitle("Top 15 Keywords per Genre", fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for i, (genre, color) in enumerate(zip(genres, colors)):
        words = " ".join(df[df["genre"] == genre]["clean_text"]).split()
        top = Counter(words).most_common(15)
        words_list, counts = zip(*top)

        ax = axes[i]
        ax.barh(words_list, counts, color=color, alpha=0.85)
        ax.set_title(genre, fontweight="bold", color=color)
        ax.invert_yaxis()
        ax.set_facecolor("white")
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("outputs/2_top_words.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → outputs/2_top_words.png")


def main():
    df = load_and_clean(RAW_FILE)
    print("\nGenre distribution:")
    print(df["genre"].value_counts().to_string())
    print(f"\nAvg overview length: {df['text_length'].mean():.1f} words")
    plot_eda(df)
    plot_top_words(df)


if __name__ == "__main__":
    main()
