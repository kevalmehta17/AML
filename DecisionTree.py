import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load CSV data
df = pd.read_csv("sample_data.csv")  # Change path if needed

# Step 2: Handle missing data if any (optional)
df = df.fillna(method='ffill')  # or df.fillna(df.mode().iloc[0])

# Step 3: Encode categorical columns
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Step 4: Split features and target
X = df.drop("Loan_Status", axis=1)  # Features
y = df["Loan_Status"]               # Target

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Step 8: Visualize the Decision Tree (optional)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.title("Decision Tree")
plt.show()

# DataSet:-
# Age,Gender,Married,Education,Income,Loan_Status
# 25,Male,Yes,Graduate,5000,Y
# 35,Female,No,Not Graduate,3000,N
# 45,Male,Yes,Graduate,8000,Y
# ...
