# Converted from notebook: phishing_email_detection.ipynb
# Cleaned script generated automatically.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Loading the dataset
df = pd.read_csv('/content/phishing_emails.csv')
df.head()



# Preprocessing data
df = df.dropna()  # Drop any missing values
df = pd.get_dummies(df, drop_first=True)  # Convert categorical to numerical
X = df.drop('label', axis=1)  # Features
y = df['label']  # Target variable



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)



# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_cm = confusion_matrix(y_test, rf_predictions)
print(f'Random Forest Accuracy: {rf_accuracy}')
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.show()



# Train Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_cm = confusion_matrix(y_test, lr_predictions)
print(f'Logistic Regression Accuracy: {lr_accuracy}')
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.show()


