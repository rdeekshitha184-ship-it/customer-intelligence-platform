import re
from collections import Counter
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def extract_keywords(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words and len(word) > 3
    ]

    word_counts = Counter(words)

    return word_counts.most_common(5)

review = input("Enter Review: ")

keywords = extract_keywords(review)

print("\nTop Keywords:")

for word, count in keywords:
    print(word)