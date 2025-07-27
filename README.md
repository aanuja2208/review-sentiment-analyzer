# review-sentiment-analyzer
# Review Sentiment Analyzer

This project implements a complete Natural Language Processing (NLP) pipeline for large-scale product review analysis. It processes over 50,000 customer reviews to extract sentiment information using the VADER model and identify the most prominent keywords through TF-IDF vectorization. The final results are visualized in a multi-panel Matplotlib dashboard that compares numeric star ratings with textual sentiment scores, highlighting mismatches and trends over time.

---

## Project Overview

- **Data Size**: 50,000+ product reviews
- **Tech Stack**: Python, Pandas, NLTK, Scikit-learn, Matplotlib
- **Approach**: Modular pipeline for data preprocessing, sentiment analysis, keyword extraction, and visual analytics
- **Result**: Identified a 25% mismatch between user ratings and textual sentiment, emphasizing the importance of contextual NLP over numeric scores alone

---

## Key Features

- Cleans and preprocesses large-scale review data (timestamp conversion, deduplication, score filtering)
- Implements VADER sentiment analysis to classify reviews as Positive, Neutral, or Negative
- Extracts the top 15 most informative keywords using TF-IDF with custom tokenization
- Generates a visual report with six subplots:
  - Distribution of 1–5 star ratings
  - Star-based sentiment percentage breakdown
  - VADER-based sentiment percentage breakdown
  - Top keywords ranked by TF-IDF score
  - Rolling average trends of both rating and VADER sentiment over time
- Fully offline, reproducible, and designed for scalability

---

## Repository Structure
<pre> ```text review-sentiment-analyzer/ ├── 1_clean_data.py # Cleans raw dataset, filters reviews, converts timestamps ├── 2_sentiment_and_topics.py # Applies VADER sentiment and keyword extraction ├── visualize_results.py # Creates a multi-panel Matplotlib report ├── detailed_review_report.png # Final output dashboard image ├── sentiment_distribution.png # Supporting sentiment distribution chart ├── cleaned_reviews.csv # Output from Step 1 (excluded from repo) ├── analyzed_reviews.csv # Output from Step 2 (excluded from repo) ├── README.md ``` </pre>

---

## Screenshots

### 1. Raw Review Input Table

This is a preview of the original data before any cleaning or processing.

<p align="center">
  <img src="screenshots/raw_reviews_table.png" width="700" alt="Raw Review Table">
</p>

---

### 2. Sentiment-Tagged Output

Output after applying VADER sentiment analysis, including sentiment labels and compound scores.

<p align="center">
  <img src="screenshots/analyzed_reviews_table.png" width="700" alt="Analyzed Sentiment Table">
</p>

---

### 3. Final Dashboard Visualization

Six-panel summary comparing star ratings, sentiment labels, keyword trends, and time-series analysis.

<p align="center">
  <img src="screenshots/dashboard_summary.png" width="800" alt="Sentiment Dashboard Summary">
</p>

---
