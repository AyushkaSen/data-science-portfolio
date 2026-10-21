import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('bank-additional.csv', sep=';')

print("--- DATASET PREVIEW ---")
print(df.head(2))
print("\nDataset Shape:", df.shape)

df_encoded = pd.get_dummies(df, drop_first=True)
target_col = [col for col in df_encoded.columns if col.startswith('y_')][0]

X = df_encoded.drop(columns=[target_col])
y = df_encoded[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# VISUALIZATION 1: Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Decision Tree - Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('task3_confusion_matrix.png', dpi=300)
plt.show()

# VISUALIZATION 2: Decision Tree Structure
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns.tolist(), class_names=['No', 'Yes'], filled=True, rounded=True, max_depth=3)
plt.title('Decision Tree Classifier Structure (Top 3 Levels)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_decision_tree.png', dpi=300)
plt.show()