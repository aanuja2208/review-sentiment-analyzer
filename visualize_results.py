import nltk
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer


def load_and_merge(clean_path: Path, analyzed_path: Path) -> pd.DataFrame:
    # 1) Load the “cleaned” version (no VADER)…
    clean = pd.read_csv(clean_path, parse_dates=['Time'])
    # 2) Load the VADER scores from the “analyzed” file…
    vader = pd.read_csv(
        analyzed_path,
        usecols=['Text', 'Time', 'VADER_Score'],
        parse_dates=['Time']
    )
    # 3) Merge on Text+Time so every clean row picks up its VADER_Score
    df = pd.merge(clean, vader, on=['Text', 'Time'], how='left')
    # 4) Drop any blank reviews
    df = df[df['Text'].str.strip().astype(bool)]
    # 5) Drop exact duplicate texts (just in case)
    df = df.drop_duplicates(subset=['Text'])
    # 6) Sort in time order so our trend chart is really chronological
    df = df.sort_values('Time').reset_index(drop=True)
    return df


def map_score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the 1–5 star 'Score' into simple labels:
      4,5 → Positive
      3   → Neutral
      1,2 → Negative
    """
    def f(s):
        if s >= 4:    return 'Positive'
        if s <= 2:    return 'Negative'
        return 'Neutral'
    df['Score_Sentiment'] = df['Score'].apply(f)
    return df


def map_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use standard VADER thresholds on the compound score:
      ≥ 0.05 → Positive
      ≤ -0.05 → Negative
      else   → Neutral
    """
    def f(c):
        if c >=  0.05: return 'Positive'
        if c <= -0.05: return 'Negative'
        return 'Neutral'
    df['VADER_Sentiment'] = df['VADER_Score'].fillna(0).apply(f)
    return df


def extract_top_keywords(texts, top_n=15, max_features=5000):
    """
    Run a TF-IDF vectorizer on all review text to find the top_n words
    that have the highest overall TF-IDF weight. We’ll show those later.
    """
    tokenizer = TreebankWordTokenizer()
    stops = set(stopwords.words('english'))

    def tok(doc):
        t = tokenizer.tokenize(doc.lower())
        return [w for w in t if w.isalpha() and w not in stops and len(w) > 2]

    vectorizer = TfidfVectorizer(tokenizer=tok, lowercase=True, max_features=max_features)
    X = vectorizer.fit_transform(texts)
    sums = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, sums), key=lambda x: x[1], reverse=True)
    return zip(*ranked[:top_n])  # returns (keywords, scores)


def plot_full_report(df: pd.DataFrame, keywords, kw_scores, out_path: Path):
    """
    Build a 2x3 grid of charts:
    [0,0] Star-rating counts
    [0,1] Score_Sentiment %
    [1,0] VADER_Sentiment %
    [1,1] Top TF-IDF keywords
    [2,:] Rolling trends of Score & VADER_Score over time
    """
    # prepare counts & percentages
    score_counts = df['Score'].value_counts().sort_index()
    score_labels = [1,2,3,4,5]
    score_counts = score_counts.reindex(score_labels, fill_value=0)

    ss = df['Score_Sentiment'].value_counts(normalize=True).mul(100)
    vs = df['VADER_Sentiment'].value_counts(normalize=True).mul(100)

    rolling_score = df['Score'].rolling(200, min_periods=1).mean()
    rolling_vader = df['VADER_Score'].rolling(200, min_periods=1).mean()

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.tight_layout(pad=4)
    fig.suptitle("Detailed Review Analysis", fontsize=20, y=1.02)

    # 1. star-rating histogram
    bars1 = axes[0,0].bar(score_labels, score_counts[score_labels])
    axes[0,0].set_title("Star Rating Distribution")
    axes[0,0].set_xlabel("Stars")
    axes[0,0].set_ylabel("Count")
    # annotate counts
    for bar in bars1:
        h = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2, h, f"{int(h)}", ha='center', va='bottom')

    # 2. Score_Sentiment %
    bars2 = axes[0,1].bar(ss.index, ss.values, color=['green','gray','red'])
    axes[0,1].set_title("Sentiment from Stars (%)")
    axes[0,1].set_ylabel("% of Reviews")
    # annotate percentages
    for bar in bars2:
        h = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2, h, f"{h:.1f}%", ha='center', va='bottom')

    # 3. VADER_Sentiment %
    bars3 = axes[1,0].bar(vs.index, vs.values, color=['green','gray','red'])
    axes[1,0].set_title("Sentiment from VADER (%)")
    axes[1,0].set_ylabel("% of Reviews")
    # annotate percentages
    for bar in bars3:
        h = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2, h, f"{h:.1f}%", ha='center', va='bottom')

    # 4. TF-IDF keywords
    axes[1,1].bar(keywords, kw_scores)
    axes[1,1].set_title("Top TF-IDF Keywords")
    axes[1,1].tick_params(axis='x', rotation=45)

    # 5. Rolling trends
    ax = axes[2,0]
    ax.plot(df['Time'], rolling_score, label='Star Rating (rolling mean)')
    ax.plot(df['Time'], rolling_vader, label='VADER Score (rolling mean)')
    ax.set_title("How Ratings & VADER Drift Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.legend()

    # hide unused subplot (bottom right)
    axes[2,1].axis('off')

    plt.subplots_adjust(top=0.92)
    plt.savefig(out_path, dpi=300)
    print(f"✅ Report written to {out_path}")


def main():
    # make sure NLTK has what it needs
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    here = Path(__file__).parent
    clean_file = here / 'cleaned_reviews.csv'
    analyzed_file = here / 'analyzed_reviews.csv'
    output_png = here / 'detailed_review_report.png'

    print("Loading and merging data…")
    df = load_and_merge(clean_file, analyzed_file)

    print("Mapping star scores to sentiment labels…")
    df = map_score_sentiment(df)

    print("Mapping VADER scores to sentiment labels…")
    df = map_vader_sentiment(df)

    print("Finding top keywords with TF-IDF…")
    kw, scores = extract_top_keywords(df['Text'])

    print("Plotting the full report…")
    plot_full_report(df, kw, scores, output_png)


if __name__ == '__main__':
    main()