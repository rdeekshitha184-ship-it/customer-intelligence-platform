import joblib

joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model Saved Successfully!")