# 🛒 Customer Intelligence Platform for E-Commerce using ML and NLP

## 📌 Overview

Customer reviews are one of the most valuable sources of feedback for e-commerce businesses. However, manually analyzing thousands of reviews is time-consuming and inefficient.

The **Customer Intelligence Platform for E-Commerce using ML and NLP** is a Machine Learning-powered application that automatically analyzes customer reviews and extracts meaningful business insights. The platform performs sentiment analysis, keyword extraction, complaint detection, and recommendation generation to help businesses understand customer feedback at scale.

The project combines **Machine Learning**, **Natural Language Processing (NLP)**, and **Interactive Data Visualization** to transform raw customer reviews into actionable insights.

---

## 🚀 Problem Statement

E-commerce platforms receive thousands of customer reviews every day.

Manually identifying:

* Customer sentiment
* Product issues
* Common complaints
* Positive product aspects
* Improvement opportunities

is difficult and resource-intensive.

Businesses need an automated system that can analyze reviews and generate meaningful insights for decision-making.

---

## 🎯 Solution

This platform automatically:

✅ Classifies customer reviews as Positive or Negative

✅ Calculates prediction confidence scores

✅ Extracts important keywords from reviews

✅ Detects potential customer complaints

✅ Generates business insights

✅ Provides actionable recommendations

✅ Visualizes results through an interactive dashboard

---

## 🏗️ System Architecture

```text
Customer Review
       │
       ▼
Text Preprocessing
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Logistic Regression Model
       │
       ▼
Prediction Engine
       │
       ├── Sentiment Analysis
       ├── Confidence Score
       ├── Keyword Extraction
       ├── Complaint Detection
       ├── Business Insights
       └── Recommendations
       │
       ▼
Streamlit Dashboard
```

---

## 📂 Dataset

### Dataset Used

Amazon Review Dataset

### Dataset Size

* 50,000 Customer Reviews

### Features

| Column | Description                   |
| ------ | ----------------------------- |
| Review | Customer Review Text          |
| Label  | Positive / Negative Sentiment |

### Class Distribution

* Positive Reviews: 25,506
* Negative Reviews: 24,494

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

* Lowercase conversion
* Removal of special characters
* Stopword removal using NLTK
* Text cleaning and normalization

### Example

**Input**

```text
Amazing!!! This product is very good and I love it.
```

**Output**

```text
amazing product good love
```

---

## 🔍 Feature Engineering

TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization was used to convert textual reviews into numerical feature vectors.

### Configuration

* Maximum Features: 5000
* Sparse Matrix Representation

### Generated Feature Matrix

```text
(50000, 5000)
```

---

## 🤖 Machine Learning Models Evaluated

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 87.93%   |
| Naive Bayes         | 85.10%   |
| Random Forest       | 84.49%   |

### Best Model

**Logistic Regression**

Accuracy Achieved:

```text
87.93%
```

---

## 📊 Classification Report

```text
Precision : 0.88
Recall    : 0.88
F1 Score  : 0.88
Accuracy  : 0.8793
```

---

## 🌟 Features

### 1. Sentiment Analysis

Predicts whether a customer review is:

* Positive 😊
* Negative 😞

---

### 2. Confidence Score

Displays prediction confidence using an interactive gauge chart.

Example:

```text
Positive Review (80.88% Confidence)
```

---

### 3. Keyword Extraction

Identifies the most important topics discussed in the review.

Example:

```text
battery
camera
performance
design
quality
```

---

### 4. Complaint Detection

Detects common customer issues.

Supported Complaint Categories:

* Heating Issue
* Battery Issue
* Charging Issue
* Refund Request
* Performance Issue

---

### 5. Business Insights

Generates business-focused observations from customer reviews.

Example:

```text
Overall customer sentiment is positive.
```

---

### 6. Recommendation Engine

Provides actionable recommendations.

Example:

```text
Investigate thermal management issues.
Review charging system reliability.
```

---

## 💻 Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-Learn
* Logistic Regression
* Naive Bayes
* Random Forest

### Natural Language Processing

* NLTK
* TF-IDF Vectorization

### Data Processing

* Pandas
* NumPy

### Visualization

* Plotly

### Web Application

* Streamlit

### Model Persistence

* Joblib

---

## 📸 Application Screenshots

Add screenshots after deployment.

Example:

```text
screenshots/homepage.png
screenshots/analysis_result.png
screenshots/business_insights.png
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-intelligence-platform.git
```

### Move into Project Directory

```bash
cd customer-intelligence-platform
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python -m streamlit run app.py
```

---

## 📈 Sample Output

### Input Review

```text
Battery life is amazing but the phone heats up while charging.
```

### Output

```text
Positive Review 😊 (80.88% Confidence)

Key Topics:
battery
life
phone
heats

Potential Complaints:
Heating Issue
Charging Issue

Business Insights:
Overall customer sentiment is positive.

Recommendations:
Investigate thermal management issues.
Review charging system reliability.
```

---

## 🎯 Future Enhancements

* Aspect-Based Sentiment Analysis
* Deep Learning Models
* BERT-based Classification
* Multi-class Sentiment Analysis
* Product Category Detection
* Customer Trend Analytics
* Review Summarization

---

## 👨‍💻 Author

**Deekshitha R**

Final Year B.E Student

Machine Learning | NLP | Data Science | Software Development

---

## ⭐ Project Highlights

* Built using Machine Learning and NLP
* Analyzed 50,000 customer reviews
* Achieved 87.93% accuracy
* Real-time review analytics dashboard
* Business insight generation
* Interactive Streamlit deployment
* End-to-end ML project lifecycle implementation
* Industry-oriented customer feedback intelligence system

---

## 📜 License

This project is developed for educational and portfolio purposes.
It demonstrates the application of Machine Learning and NLP techniques for customer review analytics in e-commerce environments.
