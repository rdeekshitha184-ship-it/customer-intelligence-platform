# import streamlit as st
# import re
# import joblib
# import plotly.graph_objects as go

# from collections import Counter
# from nltk.corpus import stopwords

# # =====================================
# # PAGE CONFIG
# # =====================================

# st.set_page_config(
#     page_title="Customer Intelligence Platform",
#     page_icon="🛒",
#     layout="wide"
# )

# # =====================================
# # LOAD MODEL
# # =====================================

# model = joblib.load("models/logistic_model.pkl")
# vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# stop_words = set(stopwords.words("english"))

# # =====================================
# # TEXT CLEANING
# # =====================================

# def clean_text(text):

#     text = text.lower()

#     text = re.sub(r"[^a-zA-Z\s]", "", text)

#     words = text.split()

#     words = [
#         word
#         for word in words
#         if word not in stop_words
#     ]

#     return " ".join(words)

# # =====================================
# # KEYWORD EXTRACTION
# # =====================================

# def extract_keywords(text):

#     text = text.lower()

#     text = re.sub(r"[^a-zA-Z\s]", "", text)

#     words = text.split()

#     words = [
#         word
#         for word in words
#         if word not in stop_words and len(word) > 3
#     ]

#     counts = Counter(words)

#     return [word for word, count in counts.most_common(5)]

# # =====================================
# # COMPLAINT DETECTION
# # =====================================

# def detect_complaints(text):

#     text = text.lower()

#     complaint_patterns = {
#         "🔥 Heating Issue": ["heat", "heats", "heating", "overheat"],
#         "🔋 Battery Issue": ["battery drain", "battery problem"],
#         "⚡ Charging Issue": ["charging", "charger"],
#         "💰 Refund Request": ["refund", "return"],
#         "🐢 Performance Issue": ["slow", "lag", "freeze", "freezes"]
#     }

#     found = []

#     for issue, keywords in complaint_patterns.items():

#         for keyword in keywords:

#             if keyword in text:
#                 found.append(issue)
#                 break

#     return found

# # =====================================
# # HEADER
# # =====================================

# st.title("🛒 Customer Intelligence Platform for E-Commerce using ML and NLP")

# st.caption(
#     "Analyze customer reviews using Machine Learning, Natural Language Processing, Sentiment Analysis, Topic Extraction, and Complaint Detection."
# )

# st.markdown("---")

# # =====================================
# # PROJECT STATS
# # =====================================

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.metric("Dataset Size", "50,000 Reviews")

# with col2:
#     st.metric("Best Accuracy", "87.93%")

# with col3:
#     st.metric("Model", "Logistic Regression")

# st.markdown("---")

# # =====================================
# # INPUT
# # =====================================

# review = st.text_area(
#     "Enter Customer Review",
#     height=180,
#     placeholder="Example: This phone has excellent battery life but gets very hot while charging."
# )

# # =====================================
# # ANALYSIS
# # =====================================

# if st.button("Analyze Review"):

#     if review.strip():

#         cleaned_review = clean_text(review)

#         review_vector = vectorizer.transform(
#             [cleaned_review]
#         )

#         prediction = model.predict(
#             review_vector
#         )[0]

#         probabilities = model.predict_proba(
#             review_vector
#         )[0]

#         confidence = max(probabilities) * 100

#         keywords = extract_keywords(review)

#         complaints = detect_complaints(review)

#         st.markdown("---")

#         st.subheader("📊 Analysis Results")

#         if prediction == 1:

#             st.success(
#                 f"Positive Review 😊 ({confidence:.2f}% confidence)"
#             )

#         else:

#             st.error(
#                 f"Negative Review 😞 ({confidence:.2f}% confidence)"
#             )

#         # =====================================
#         # CONFIDENCE GAUGE
#         # =====================================

#         fig = go.Figure(
#             go.Indicator(
#                 mode="gauge+number",
#                 value=confidence,
#                 title={"text": "Prediction Confidence (%)"},
#                 gauge={
#                     "axis": {"range": [0, 100]},
#                     "bar": {"thickness": 0.4},
#                     "steps": [
#                         {"range": [0, 50], "color": "lightcoral"},
#                         {"range": [50, 80], "color": "khaki"},
#                         {"range": [80, 100], "color": "lightgreen"}
#                     ]

#                 }
#             )
#         )


#         fig.update_layout(
#              height=350
#         )

#         st.plotly_chart(
#             fig,
#             use_container_width=True
#      )

#         # =====================================
#         # REVIEW SUMMARY
#         # =====================================

#         st.subheader("📋 Review Summary")

#         if prediction == 1:

#             st.write(
#                 f"""
#                 The customer sentiment is positive with
#                 {confidence:.2f}% confidence.
#                 """
#             )

#         else:

#             st.write(
#                 f"""
#                 The customer sentiment is negative with
#                 {confidence:.2f}% confidence.
#                 """
#             )

#         # =====================================
#         # TWO COLUMN LAYOUT
#         # =====================================

#         col_left, col_right = st.columns(2)

#         with col_left:

#             st.subheader("📝 Key Topics Discussed")

#             if keywords:

#                 for keyword in keywords:
#                     st.write(f"• {keyword}")

#             else:
#                 st.write("No keywords found.")

#         with col_right:

#             st.subheader("⚠️ Potential Complaints")

#             if complaints:

#                 for complaint in complaints:
#                     st.write(f"• {complaint}")

#             else:

#                 st.success("No complaints detected.")

# # =====================================
# # FOOTER
# # =====================================

# st.markdown("---")

# st.caption(
#     "Built using Python, NLP, TF-IDF, Logistic Regression, Scikit-Learn, Streamlit, and Plotly."
# )



import streamlit as st
import re
import joblib
import plotly.graph_objects as go

from collections import Counter
from nltk.corpus import stopwords

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🛒",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

# =====================================
# TEXT CLEANING
# =====================================

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

# =====================================
# KEYWORD EXTRACTION
# =====================================

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

# =====================================
# COMPLAINT DETECTION
# =====================================

def detect_complaints(text):

    text = text.lower()

    complaint_patterns = {
        "🔥 Heating Issue": ["heat", "heats", "heating", "overheat"],
        "🔋 Battery Issue": ["battery drain", "battery problem"],
        "⚡ Charging Issue": ["charging", "charger"],
        "💰 Refund Request": ["refund", "return"],
        "🐢 Performance Issue": ["slow", "lag", "freeze", "freezes"]
    }

    found = []

    for issue, keywords in complaint_patterns.items():

        for keyword in keywords:

            if keyword in text:
                found.append(issue)
                break

    return found

# =====================================
# BUSINESS INSIGHTS
# =====================================

def generate_business_insights(
    prediction,
    keywords,
    complaints
):

    insights = []

    recommendations = []

    if prediction == 1:

        insights.append(
            "Overall customer sentiment is positive."
        )

    else:

        insights.append(
            "Overall customer sentiment is negative."
        )

    if complaints:

        for complaint in complaints:

            if "Heating" in complaint:

                recommendations.append(
                    "Investigate thermal management issues."
                )

            elif "Battery" in complaint:

                recommendations.append(
                    "Improve battery performance and optimization."
                )

            elif "Charging" in complaint:

                recommendations.append(
                    "Review charging system reliability."
                )

            elif "Refund" in complaint:

                recommendations.append(
                    "Analyze product quality and customer satisfaction."
                )

            elif "Performance" in complaint:

                recommendations.append(
                    "Optimize product performance."
                )

    return insights, recommendations

# =====================================
# HEADER
# =====================================

st.title(
    "🛒 Customer Intelligence Platform for E-Commerce using ML and NLP"
)

st.caption(
    "Analyze customer reviews using Machine Learning, Natural Language Processing, Sentiment Analysis, Topic Extraction, Complaint Detection, and Business Insights."
)

st.markdown("---")

# =====================================
# PROJECT STATS
# =====================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Size", "50,000 Reviews")

with col2:
    st.metric("Best Accuracy", "87.93%")

with col3:
    st.metric("Model", "Logistic Regression")

st.markdown("---")

# =====================================
# INPUT
# =====================================

review = st.text_area(
    "Enter Customer Review",
    height=180,
    placeholder="Example: This phone has excellent battery life but gets very hot while charging."
)

# =====================================
# ANALYSIS
# =====================================

if st.button("Analyze Review"):

    if review.strip():

        cleaned_review = clean_text(review)

        review_vector = vectorizer.transform(
            [cleaned_review]
        )

        prediction = model.predict(
            review_vector
        )[0]

        probabilities = model.predict_proba(
            review_vector
        )[0]

        confidence = max(probabilities) * 100

        keywords = extract_keywords(review)

        complaints = detect_complaints(review)

        insights, recommendations = generate_business_insights(
            prediction,
            keywords,
            complaints
        )

        st.markdown("---")

        st.subheader("📊 Analysis Results")

        if prediction == 1:

            st.success(
                f"Positive Review 😊 ({confidence:.2f}% confidence)"
            )

        else:

            st.error(
                f"Negative Review 😞 ({confidence:.2f}% confidence)"
            )

        # =====================================
        # CONFIDENCE GAUGE
        # =====================================

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={"text": "Prediction Confidence (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"thickness": 0.4},
                    "steps": [
                        {"range": [0, 50], "color": "lightcoral"},
                        {"range": [50, 80], "color": "khaki"},
                        {"range": [80, 100], "color": "lightgreen"}
                    ]
                }
            )
        )

        fig.update_layout(
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =====================================
        # REVIEW SUMMARY
        # =====================================

        st.subheader("📋 Review Summary")

        if prediction == 1:

            st.write(
                f"""
                The customer sentiment is positive with
                {confidence:.2f}% confidence.
                """
            )

        else:

            st.write(
                f"""
                The customer sentiment is negative with
                {confidence:.2f}% confidence.
                """
            )

        # =====================================
        # TWO COLUMN LAYOUT
        # =====================================

        col_left, col_right = st.columns(2)

        with col_left:

            st.subheader("📝 Key Topics Discussed")

            if keywords:

                for keyword in keywords:
                    st.write(f"• {keyword}")

            else:
                st.write("No keywords found.")

        with col_right:

            st.subheader("⚠️ Potential Complaints")

            if complaints:

                for complaint in complaints:
                    st.write(f"• {complaint}")

            else:

                st.success("No complaints detected.")

        # =====================================
        # BUSINESS INSIGHTS
        # =====================================

        st.markdown("---")

        st.subheader("📈 Business Insights")

        for insight in insights:

            st.write(f"• {insight}")

        # =====================================
        # RECOMMENDATIONS
        # =====================================

        st.subheader("💡 Recommendations")

        if recommendations:

            for recommendation in recommendations:

                st.write(f"• {recommendation}")

        else:

            st.success(
                "No major product issues detected."
            )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.caption(
    "Built using Python, NLP, TF-IDF, Logistic Regression, Scikit-Learn, Streamlit, and Plotly."
)

