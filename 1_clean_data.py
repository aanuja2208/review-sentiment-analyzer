import pandas as pd

df = pd.read_csv("reviews.csv")
df = df[["Text", "Time", "Score"]]  # keep relevant columns
df.dropna(inplace=True)
df = df[df["Score"] != 3]  # Remove neutral for clarity

# Convert timestamp
df["Time"] = pd.to_datetime(df["Time"], unit="s")

# Label sentiment
df["Sentiment"] = df["Score"].apply(lambda x: "Positive" if x > 3 else "Negative")

df.to_csv("cleaned_reviews.csv", index=False)
print("✅ Data cleaned and saved.")
