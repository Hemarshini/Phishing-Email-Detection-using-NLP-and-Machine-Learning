# Phishing Email Detection using NLP and Machine Learning

**# Summary
The models above (Random Forest and Logistic Regression) are trained on the phishing email dataset and evaluated using accuracy and confusion matrix.**

## Objectives
- Build a machine learning pipeline that classifies emails as phishing or legitimate using NLP techniques.
- Demonstrate data preprocessing, feature extraction, model training, and evaluation with reproducible code.

## Dataset
Used a CSV file named phishing_emails.csv.

## Technologies & Libraries

This project uses Python and common machine learning / NLP libraries. See `requirements.txt` for details.

## Project Workflow

1. Data loading and cleaning
2. Text preprocessing and feature extraction (CountVectorizer / TF-IDF)
3. Model training and selection (e.g., LogisticRegression, SVM, RandomForest)
4. Evaluation using accuracy, precision, recall, F1-score and confusion matrix

## How to run

```bash
git clone https://github.com/<your-username>/Phishing_Email_Detection_using_NLP_and_Machine_Learning.git
cd "Phishing_Email_Detection_using_NLP_and_Machine_Learning"
pip install -r requirements.txt
python src/model_training.py
```
Note: If your dataset is stored on Google Drive or Kaggle, place the dataset files inside the `data/` directory and update paths in `src/model_training.py` accordingly.

## Results

The notebook contains the following result snippets or metrics (best-effort extraction):

```
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
```

```
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
```

```
# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_cm = confusion_matrix(y_test, rf_predictions)
print(f'Random Forest Accuracy: {rf_accuracy}')
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.show()
```

```
# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_cm = confusion_matrix(y_test, rf_predictions)
print(f'Random Forest Accuracy: {rf_accuracy}')
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.show()
```

```
# Train Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_cm = confusion_matrix(y_test, lr_predictions)
print(f'Logistic Regression Accuracy: {lr_accuracy}')
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.show()
```

```
# Train Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_cm = confusion_matrix(y_test, lr_predictions)
print(f'Logistic Regression Accuracy: {lr_accuracy}')
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.show()
```

```
# Summary
The models above (Random Forest and Logistic Regression) are trained on the phishing email dataset and evaluated using accuracy and confusion matrix.
```

## Future Enhancements

- Improve model by using deep learning (LSTM / Transformers)
- Deploy the model as an API or web app
- Add model interpretability (SHAP / LIME)

## Author

- Divvela Hemarshini

## License

This project is released under the MIT License. See `LICENSE` for details.
