import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import csv

df = pd.read_csv("twitter_training.csv", 
                 header=None, 
                 encoding='latin-1', 
                 dtype={0: str}, 
                 quoting=csv.QUOTE_ALL,
                 quotechar='"')

df.columns = ['Tweet_ID', 'Entity', 'Sentiment', 'Tweet']

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.lower()
    else:
        return ""

df['Clean_Tweet'] = df['Tweet'].apply(clean_text)

df = df[df['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Sentiment', palette='pastel')
plt.title("Tweet Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.tight_layout()
plt.savefig("sentiment_distribution_fixed.png")
plt.show()

print("\n Sample Positive Tweet:")
print(df[df['Sentiment'] == 'Positive']['Tweet'].iloc[0])

print("\n Sample Negative Tweet:")
print(df[df['Sentiment'] == 'Negative']['Tweet'].iloc[0])

print("\n Sample Neutral Tweet:")
print(df[df['Sentiment'] == 'Neutral']['Tweet'].iloc[0])
