import re
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

sample_review = "Amazing!!! This product is very good and I love it."

# Convert to lowercase
review = sample_review.lower()

# Remove punctuation
review = re.sub(r"[^a-zA-Z\s]", "", review)

# Remove stopwords
words = review.split()

clean_words = []

for word in words:
    if word not in stop_words:
        clean_words.append(word)

clean_review = " ".join(clean_words)

print("Original Review:")
print(sample_review)

print("\nCleaned Review:")
print(clean_review)