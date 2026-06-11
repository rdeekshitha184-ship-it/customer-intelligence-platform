import pandas as pd
import re
from nltk.corpus import stopwords

# Stopwords
stop_words = set(stopwords.words("english"))

# Load first 50,000 reviews
reviews = []
labels = []

with open("data/train.ft.txt", "r", encoding="utf-8") as file:

    for i in range(50000):

        line = file.readline()

        if not line:
            break

        parts = line.split(" ", 1)

        labels.append(parts[0])
        reviews.append(parts[1])

# Create DataFrame
df = pd.DataFrame({
    "label": labels,
    "review": reviews
})

# Convert labels
df["label"] = df["label"].map({
    "__label__1": 0,
    "__label__2": 1
})

# Cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Create cleaned review column
df["cleaned_review"] = df["review"].apply(clean_text)

# Show results
print(df[["review", "cleaned_review"]].head())