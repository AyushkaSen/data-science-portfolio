import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
df_train = pd.read_csv('train.csv')
print("--- 1. DATA INSPECTION ---")
print(df_train.head())
print("\nDataset Info:")
print(df_train.info())
print("\n--- 2. DATA CLEANING ---")
print("Missing values before cleaning:\n", df_train.isnull().sum())

df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_train.drop(columns=['Cabin'], inplace=True)
print("\nMissing values after cleaning:\n", df_train.isnull().sum())

print("\n--- 3. EXPLORATORY DATA ANALYSIS (EDA) ---")
# Summary statistics
print(df_train.describe())


# VISUALIZATION 1: Survival Rate by Gender
plt.figure(figsize=(8, 5))
sns.barplot(data=df_train, x='Sex', y='Survived', palette='pastel')
plt.title('Survival Rate by Gender', fontsize=14, fontweight='bold')
plt.ylabel('Survival Probability', fontsize=12)
plt.xlabel('Gender', fontsize=12)
plt.savefig('task2_survival_by_sex.png', dpi=300)
plt.show()

# VISUALIZATION 2: Age Distribution by Survival
plt.figure(figsize=(10, 5))
sns.histplot(data=df_train, x='Age', hue='Survived', kde=True, bins=30, palette=['tomato', 'mediumseagreen'])
plt.title('Age Distribution by Survival Status', fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Survived', labels=['No (0)', 'Yes (1)'])
plt.savefig('task2_age_distribution.png', dpi=300)
plt.show()

# VISUALIZATION 3: Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df_train.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Titanic Features', fontsize=14, fontweight='bold')
plt.savefig('task2_correlation_matrix.png', dpi=300)
plt.show()
