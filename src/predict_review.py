import re
import joblib

from nltk.corpus import stopwords

# Load saved model and vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Take review input
review = input("Enter a review: ")

# Clean review
cleaned_review = clean_text(review)

# Convert to TF-IDF
review_vector = vectorizer.transform([cleaned_review])

# Predict
prediction = model.predict(review_vector)[0]

# Display result
if prediction == 1:
    print("\nPrediction: Positive Review ")
else:
    print("\nPrediction: Negative Review ")