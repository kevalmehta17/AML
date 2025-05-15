import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
classification_report
# Step 2: Load Dataset and also check if there null vallues in dataset
file_path = 'health care diabetes.csv' # Adjust this if uploaded differently
df = pd.read_csv(file_path)
print("First 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
# Step 3: Preprocessing
df = df.dropna()
df = df.drop_duplicates()
# Step 4: Feature and Label Separation and splitting the data
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Step 6: Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
# Step 7: Predictions
y_pred = model.predict(X_test_scaled)
# Step 8: Evaluation
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
# Accuracy and Classification Report
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))