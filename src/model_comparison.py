import pandas as pd
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# Stopwords
stop_words = set(stopwords.words("english"))

reviews = []
labels = []

# Load first 50,000 reviews
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

# Cleaning Function
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Clean Reviews
df["cleaned_review"] = df["review"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["cleaned_review"])

y = df["label"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

print("\nMODEL COMPARISON")
print("-" * 50)

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results.append([name, accuracy])

    print(f"{name}: {accuracy:.4f}")

# Results Table
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy"]
)

print("\nFinal Results:")
print(results_df.sort_values(
    by="Accuracy",
    ascending=False
))