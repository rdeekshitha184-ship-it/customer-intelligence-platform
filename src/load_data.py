import pandas as pd

# Lists to store reviews and labels
reviews = []
labels = []

# Dataset path
file_path = "data/train.ft.txt"

# Read first 50,000 reviews
with open(file_path, "r", encoding="utf-8") as file:

    for i in range(50000):

        line = file.readline()

        if not line:
            break

        # Split label and review text
        parts = line.split(" ", 1)

        label = parts[0]
        review = parts[1]

        labels.append(label)
        reviews.append(review)

# Create DataFrame
df = pd.DataFrame({
    "label": labels,
    "review": reviews
})

# Display first few rows
print("First 5 Rows:")
print(df.head())

# Shape of dataset
print("\nShape:", df.shape)

# Count labels
print("\nLabel Distribution:")
print(df["label"].value_counts())

# Convert labels to numeric values
df["label"] = df["label"].map({
    "__label__1": 0,
    "__label__2": 1
})

# Display after mapping
print("\nAfter Mapping:")
print(df.head())

# Calculate review length

df["review_length"] = df["review"].apply(len)

print("\nReview Length Statistics:")
print(df["review_length"].describe())

print("\nShortest Review:")
print(df.loc[df["review_length"].idxmin(), "review"])

print("\nLongest Review Length:")
print(df["review_length"].max())