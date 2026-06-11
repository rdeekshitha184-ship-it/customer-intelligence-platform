import pandas as pd
import re

from nltk.corpus import stopwords
from collections import Counter

stop_words = set(stopwords.words("english"))

reviews = []

with open("data/train.ft.txt", "r", encoding="utf-8") as file:

    for i in range(10000):

        line = file.readline()

        if not line:
            break

        parts = line.split(" ", 1)

        reviews.append(parts[1])

# Clean text
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return words

all_words = []

for review in reviews:

    words = clean_text(review)

    all_words.extend(words)

word_counts = Counter(all_words)

print("\nPotential Product Features:\n")

for word, count in word_counts.most_common(200):

    if len(word) > 5:
        print(f"{word}: {count}")