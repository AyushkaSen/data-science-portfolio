import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")

# 1. Load the dataset (using a sample if the full file is too large for local memory)
print("Loading dataset...")
df = pd.read_csv('US_Accidents_March23.csv') # Replace with your actual filename if different

print("Dataset Preview:")
print(df.head(2))

# ==========================================
# VISUALIZATION 1: Accidents by Time of Day (Hour)
# ==========================================
if 'Start_Time' in df.columns:
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Hour'] = df['Start_Time'].dt.hour

    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Hour', palette='crest')
    plt.title('Accident Frequency by Time of Day (Hour)', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of the Day (24-hour format)', fontsize=12)
    plt.ylabel('Number of Accidents', fontsize=12)
    plt.tight_layout()
    plt.savefig('task5_accidents_by_hour.png', dpi=300)
    plt.show()

# ==========================================
# VISUALIZATION 2: Accidents by Weather Condition
# ==========================================
if 'Weather_Condition' in df.columns:
    plt.figure(figsize=(12, 6))
    top_weather = df['Weather_Condition'].value_counts().head(10)
    
    sns.barplot(x=top_weather.values, y=top_weather.index, palette='magma')
    plt.title('Top 10 Weather Conditions During Accidents', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Accidents', fontsize=12)
    plt.ylabel('Weather Condition', fontsize=12)
    plt.tight_layout()
    plt.savefig('task5_accidents_by_weather.png', dpi=300)
    plt.show()

# ==========================================
# VISUALIZATION 3: Severity Breakdown
# ==========================================
if 'Severity' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Severity', palette='Set2')
    plt.title('Distribution of Accident Severity Levels', fontsize=14, fontweight='bold')
    plt.xlabel('Severity Level', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('task5_accident_severity.png', dpi=300)
    plt.show()