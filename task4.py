import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

df = pd.read_csv('twitter_training.csv', header=None)
df.columns = ['ID', 'Entity', 'Sentiment', 'Text']

print("--- DATASET PREVIEW ---")
print(df.head())
print("\nSentiment Category Counts:")
print(df['Sentiment'].value_counts())

# VISUALIZATION 1: Overall Sentiment Distribution
plt.figure(figsize=(8, 5))
sentiment_counts = df['Sentiment'].value_counts()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Overall Sentiment Distribution in Social Media Data', fontsize=14, fontweight='bold')
plt.xlabel('Sentiment Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('task4_sentiment_distribution.png', dpi=300)
plt.show()

# VISUALIZATION 2: Sentiment Breakdown by Top Brands/Topics
plt.figure(figsize=(12, 6))
top_entities = df['Entity'].value_counts().head(6).index
df_top = df[df['Entity'].isin(top_entities)]

sns.countplot(data=df_top, x='Entity', hue='Sentiment', palette='Set2')

plt.title('Sentiment Distribution Across Top Entities/Brands', fontsize=14, fontweight='bold')
plt.xlabel('Entity / Topic', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=25)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('task4_sentiment_by_entity.png', dpi=300)
plt.show()