import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from tqdm import tqdm
import nltk

# Ensure NLTK data is available
nltk.download("punkt")
nltk.download("stopwords")

# Load cleaned data
df = pd.read_csv("cleaned_reviews.csv")

# --- Sentiment Analysis ---
print("\n🔍 Running sentiment analysis...")
analyzer = SentimentIntensityAnalyzer()

tqdm.pandas(desc="Analyzing sentiment")
df["VADER_Score"] = df["Text"].astype(str).progress_apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

# --- Keyword Extraction (like topic modeling) ---
print("\n🔍 Extracting keywords...")
stop_words = set(stopwords.words("english"))
all_words = []

for review in tqdm(df["Text"].astype(str), desc="Processing reviews"):
    tokens = word_tokenize(review.lower())
    words = [w for w in tokens if w.isalpha() and w not in stop_words]
    all_words.extend(words)

# Top 20 most common words
top_words = Counter(all_words).most_common(20)

# Print them as "topics"
print("\n🧠 Top 20 Keywords in Reviews:")
for word, count in top_words:
    print(f"{word}: {count}")

# Save final output
df.to_csv("analyzed_reviews.csv", index=False)
print("\n✅ Sentiment + keyword analysis saved to 'analyzed_reviews.csv'")

