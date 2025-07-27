# review-sentiment-analyzer
End-to-end NLP pipeline using VADER + TF-IDF with Matplotlib visualizations
An end-to-end sentiment analysis pipeline built for large-scale product review datasets (50,000+ entries), combining natural language processing with interactive visualizations to uncover meaningful customer insights.

Highlights:
1. Automated sentiment tagging using the VADER model on real-world review text, achieving over 90% classification coverage
2. TF-IDF-based keyword extraction to identify the most frequent and impactful customer concerns and product features
3. Data cleaning, deduplication, and timestamp conversion built into a modular Python workflow
4. 6-panel analytics dashboard built in Matplotlib, comparing star ratings with VADER sentiment trends over time
5. Revealed a 25% mismatch between written sentiment and numeric ratings — demonstrating the value of NLP in user experience analysis
6. Designed for full offline reproducibility using 3 clean Python scripts: data cleaning, sentiment modeling, and report generation
