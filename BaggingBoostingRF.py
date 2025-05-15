import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Load your dataset (replace with your actual path)
df = pd.read_csv("Loan_Data.csv")

# Fill missing values if needed
df = df.ffill()

# Encode categorical columns
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Split into features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
bagging_model.fit(X_train, y_train)
bagging_preds = bagging_model.predict(X_test)
print("Bagging Accuracy:", accuracy_score(y_test, bagging_preds))

# Random Forest and Boosting
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))

# Boosting
boost_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
boost_model.fit(X_train, y_train)
boost_preds = boost_model.predict(X_test)
print("Boosting Accuracy:", accuracy_score(y_test, boost_preds))