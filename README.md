# Review Sentiment Analyzer

This project implements a comprehensive Natural Language Processing (NLP) pipeline for analysing large-scale product reviews. It processes over 50,000 customer reviews to extract sentiment information using the VADER model and identify the most prominent keywords through TF-IDF vectorisation. The final results are visualised in a multi-panel Matplotlib dashboard that compares numeric star ratings with textual sentiment scores, highlighting mismatches and trends over time.

---

## Project Overview

This project implements a modular NLP pipeline to analyse over **50,000 product reviews**, extracting sentiment and keyword insights from unstructured text. Using rule-based sentiment classification (VADER) and TF-IDF keyword extraction, it visualises review behaviour through a multi-panel Matplotlib dashboard.

- **Data Size**: 50,000+ product reviews  
- **Pipeline**: Data cleaning → Sentiment analysis → Keyword extraction → Visualization  
- **Outcome**: Identified a **25% mismatch** between star ratings and textual sentiment, highlighting the importance of contextual NLP over numeric scoring  
- **Execution**: Fully script-driven, offline, and reproducible

---

## Tech Stack

- **Language**: Python 3  
- **Libraries**: Pandas, NLTK, Matplotlib, Scikit-learn, tqdm  
- **NLP**: VADER sentiment analysis, TF-IDF keyword extraction  
- **Visualisation**: Multi-panel Matplotlib dashboard (6 subplots)  
- **Data Format**: CSV (Text, Score, Time columns)

---

## Key Features

- Preprocesses raw review data (timestamp parsing, neutral filtering, deduplication)  
- Applies VADER model to classify sentiment (Positive / Neutral / Negative)  
- Extracts top keywords using TF-IDF with token filtering  
- Generates a comprehensive 6-panel visual report:
  - Star rating distribution  
  - Sentiment breakdown from ratings and VADER  
  - Top keywords by TF-IDF  
  - Rolling average plots of star ratings and sentiment over time  
- Modular codebase: 3 standalone Python scripts for cleaning, analysis, and visualisation

---
## Project Structure
```text
review-sentiment-analyzer/
├── 1_clean_data.py              # Cleans raw dataset, filters reviews, converts timestamps
├── 2_sentiment_and_topics.py   # Applies VADER sentiment analysis and TF-IDF keyword extraction
├── visualize_results.py        # Generates 6-panel Matplotlib sentiment dashboard
├── cleaned_reviews.csv         # Output from Step 1 (excluded from GitHub due to size)
├── analyzed_reviews.csv        # Output from Step 2 (excluded from GitHub due to size)
├── detailed_review_report.png  # Final dashboard summarising sentiment and keywords
├── sentiment_distribution.png  # VADER sentiment breakdown bar chart
├── screenshots/                # Contains all visual screenshots
│   ├── raw_reviews_table.png.png
│   ├── analysed_reviews_table.png.png
│   └── dashboard_summary.png.png
├── requirements.txt            # Python dependencies for this project
└── README.md                   # Project overview and instructions
```

---

## Screenshots

### 1. Raw Review Input Table

Preview of the raw dataset containing unprocessed customer reviews with timestamps and 1–5 star ratings.

<p align="center">
  <img src="Screenshots/raw_reviews_table.png.png" width="700" alt="Raw Review Table">
</p>

---

### 2. Sentiment-Tagged Output

After applying VADER sentiment analysis, each review is labelled with sentiment (Positive/Negative) along with its compound score.

<p align="center">
  <img src="Screenshots/analysed_reviews_table.png.png" width="700" alt="Analyzed Sentiment Table">
</p>

---

### 3. Final Dashboard Visualisation

Six-panel visualisation generated using Matplotlib, showing rating distribution, sentiment breakdowns, TF-IDF keyword analysis, and time-based sentiment trends.

<p align="center">
  <img src="Screenshots/dashboard_summary.png.png" width="800" alt="Dashboard Summary Visualization">
</p>
