import re
import joblib

from collections import Counter
from nltk.corpus import stopwords

# Load Model and Vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

# Complaint Keywords
complaint_words = [
    "bad",
    "worst",
    "poor",
    "broken",
    "issue",
    "problem",
    "defect",
    "disappointed",
    "heating",
    "slow",
    "damage",
    "damaged",
    "refund",
    "return"
]

# Text Cleaning
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Keyword Extraction
def extract_keywords(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words and len(word) > 3
    ]

    counts = Counter(words)

    return [word for word, count in counts.most_common(5)]

# Complaint Detection
def detect_complaints(text):

    text = text.lower()

    found = []

    for word in complaint_words:

        if word in text:
            found.append(word)

    return found

# User Input
review = input("Enter Review: ")

# Sentiment Prediction
cleaned_review = clean_text(review)

review_vector = vectorizer.transform([cleaned_review])


prediction = model.predict(review_vector)[0]

probabilities = model.predict_proba(review_vector)[0]

confidence = max(probabilities) * 100

# Keywords
keywords = extract_keywords(review)

# Complaints
complaints = detect_complaints(review)

# Output
print("\n===== CUSTOMER INTELLIGENCE REPORT =====")

if prediction == 1:
    print("\nSentiment : Positive 😊")
else:
    print("\nSentiment : Negative 😞")

print(f"Confidence: {confidence:.2f}%")     

print("\nTop Keywords:")
for keyword in keywords:
    print("-", keyword)

if complaints:
    print("\nPotential Complaints:")
    for complaint in complaints:
        print("-", complaint)
else:
    print("\nPotential Complaints: None")