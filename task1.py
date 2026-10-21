import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
df_pop = pd.read_csv('API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv', skiprows=4)
df_country = pd.read_csv('Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv')
df_merged = df_pop.merge(df_country[['Country Code', 'Region', 'IncomeGroup']], on='Country Code', how='left')
df_countries = df_merged.dropna(subset=['Region']).copy()

plt.figure(figsize=(10, 5))
region_counts = df_countries['Region'].value_counts()
sns.barplot(x=region_counts.values, y=region_counts.index, palette='mako')

plt.title('Distribution of Countries by Region', fontsize=14, fontweight='bold')
plt.xlabel('Number of Countries', fontsize=12)
plt.ylabel('Region', fontsize=12)
plt.tight_layout()
plt.savefig('region_distribution.png', dpi=300)
plt.show()
plt.figure(figsize=(10, 5))
sns.histplot(df_countries['2023'].dropna() / 1e6, bins=25, kde=True, color='teal')

plt.title('Distribution of Country Populations in 2023 (in Millions)', fontsize=14, fontweight='bold')
plt.xlabel('Population (Millions)', fontsize=12)
plt.ylabel('Frequency (Number of Countries)', fontsize=12)
plt.tight_layout()
plt.savefig('population_histogram.png', dpi=300)
plt.show()