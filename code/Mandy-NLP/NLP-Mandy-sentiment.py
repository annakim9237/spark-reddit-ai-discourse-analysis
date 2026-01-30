import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# 1. LOAD DATA
# =========================================================

path = "../data/csv/EDA_Mandy/EDA-Mandy-Top-Examples_per_subreddit.csv"
df = pd.read_csv(path)

# Use titles for sentiment/emotion
df["text"] = df["title"].fillna("")

print("Loaded rows:", len(df))
df.head()


# =========================================================
# 2. OFFLINE NRC-STYLE EMOTION LEXICON
#    (Expandable if you want more precision)
# =========================================================

emotion_lexicon = {
    "happy": ["joy"],
    "joy": ["joy"],
    "love": ["joy", "trust"],
    "excited": ["joy", "anticipation"],
    "win": ["joy"],
    "great": ["joy", "positive"],
    "amazing": ["joy", "positive"],
    "cool": ["joy", "positive"],
    "beautiful": ["joy", "positive"],

    "fear": ["fear"],
    "scared": ["fear"],
    "threat": ["fear"],
    "risk": ["fear"],
    "unsafe": ["fear"],

    "angry": ["anger"],
    "rage": ["anger"],
    "mad": ["anger"],
    "hate": ["anger", "negative"],
    "furious": ["anger"],

    "sad": ["sadness"],
    "loss": ["sadness"],
    "cry": ["sadness"],
    "depressed": ["sadness"],

    "shock": ["surprise"],
    "surprised": ["surprise"],
    "unexpected": ["surprise"],

    "bad": ["negative"],
    "wrong": ["negative"],
    "problem": ["negative", "fear"],
    "danger": ["fear"],
    "fail": ["negative"],
    "terrible": ["negative"],

    "trust": ["trust"],
    "secure": ["trust"],
    "safe": ["trust"],

    "anticipate": ["anticipation"],
    "future": ["anticipation"]
}

emotions = [
    "joy", "sadness", "anger", "fear",
    "surprise", "anticipation", "trust",
    "positive", "negative"
]


# =========================================================
# 3. EMOTION SCORING FUNCTION
# =========================================================

def compute_emotions(text):
    text = text.lower()
    words = re.findall(r"\b[a-z]+\b", text)
    
    scores = defaultdict(int)
    
    for w in words:
        if w in emotion_lexicon:
            for emo in emotion_lexicon[w]:
                scores[emo] += 1

    total = sum(scores.values()) or 1
    return {emo: scores[emo] / total for emo in emotions}


# =========================================================
# 4. APPLY EMOTION SCORING
# =========================================================

emotion_scores = df["text"].apply(compute_emotions)
emotion_df = pd.concat([df, emotion_scores.apply(pd.Series)], axis=1)

emotion_df.head()


# =========================================================
# 5. AGGREGATE EMOTIONS PER SUBREDDIT
# =========================================================

subreddit_emotions = emotion_df.groupby("subreddit")[emotions].mean().reset_index()
print(subreddit_emotions)


# =========================================================
# 6. RADAR CHART GENERATOR
# =========================================================

def make_radar(subreddit, row):
    values = row[emotions].values.flatten().tolist()
    values += values[:1]   # close the loop
    
    angles = np.linspace(0, 2*np.pi, len(emotions), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotions)
    ax.set_title(f"Emotion Radar: {subreddit}")

    plt.tight_layout()
    filename = f"emotion_radar_{subreddit}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


# =========================================================
# 7. GENERATE RADAR CHART FOR *EACH SUBREDDIT*
# =========================================================

for _, row in subreddit_emotions.iterrows():
    make_radar(row["subreddit"], row)

print("\n✓ All emotion radar charts generated!")
