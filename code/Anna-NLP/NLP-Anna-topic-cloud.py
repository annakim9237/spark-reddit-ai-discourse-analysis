import pandas as pd
import ast
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import math

df = pd.read_csv("./data/csv/NLP_Anna/NLP_with_neoliberal/NLPQ1_Anna_spark_lda_topics_ver2_1.csv")

n_topics = len(df)
cols = 5                             
rows = math.ceil(n_topics / cols)     

fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
axes = axes.flatten()  

for i, (_, row) in enumerate(df.iterrows()):
    topic_id = row["topic"]
    words = ast.literal_eval(row["terms_words"])
    weights = ast.literal_eval(row["termWeights"])

    freq = {w: wgt for w, wgt in zip(words, weights)}

    wc = WordCloud(width=400, height=300, background_color="white")
    wc = wc.generate_from_frequencies(freq)

    ax = axes[i]
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Topic {topic_id}")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()


plt.savefig("topic_wordclouds_all_topics.png", dpi=300, bbox_inches="tight")