import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import matplotlib.pyplot as plt

df1 = pd.read_csv(r"C:\Users\keval\Downloads\Train_Data.csv")
df1.head(10)

#Here we handle the missing data and transfrom it into the 0-1 using the LabelEncoding
df1['Gender'] = df1['Gender'].ffill()
le = LabelEncoder()
df1["Gender"] = le.fit_transform(df1['Gender'])
print(df1["Gender"].unique())

print("Before the value", df1["Married"].unique())
df1["Married"] = df1["Married"].bfill()
df1["Married"] = le.fit_transform(df1["Married"])
print("After the value", df1["Married"].unique())

df1['Dependents'] = df1['Dependents'].replace("3+",3)
df1['Dependents'] = df1['Dependents'].fillna(df1['Dependents'].mode()[0])
df1.head(10)

df1 = df1.drop(columns=["Loan_ID"])
df1.head(10)


df1["Education"] = le.fit_transform(df1["Education"])
print(df1["Education"].unique())
df1.head(10)

df1["Self_Employed"] = df1["Self_Employed"].bfill()
df1["Self_Employed"] = le.fit_transform(df1['Self_Employed'])
df1.head(10)


x = np.array(df1['ApplicantIncome'])
y = preprocessing.normalize([x])
df1['ApplicantIncome'] = y.flatten()
df1.head(10)

x = np.array(df1['CoapplicantIncome'])
y = preprocessing.normalize([x])
df1['CoapplicantIncome'] = y.flatten()
df1.head(10)

df1['LoanAmount'] = df1['LoanAmount'].fillna(df1['LoanAmount']).mean()
df1.head(10)

df1['Loan_Amount_Term'] = df1['Loan_Amount_Term'].fillna(df1['Loan_Amount_Term']).mean()
df1.head(10)

df1['Credit_History'] = df1['Credit_History'].fillna(df1['Credit_History'].mode()[0])
df1.head(10)

df1['Property_Area'] = df1['Property_Area'].ffill()
df1['Property_Area'] = le.fit_transform(df1['Property_Area'])
df1['Loan_Status'] = le.fit_transform(df1['Loan_Status'])
df1.head(10)

plt.boxplot(df1["ApplicantIncome"])

Q1 = df1["ApplicantIncome"].quantile(0.25)
Q3 = df1["ApplicantIncome"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataframe to remove outliers
df1_no_outliers = df1[(df1["ApplicantIncome"] >= lower_bound) & (df1["ApplicantIncome"] <= upper_bound)]

plt.boxplot(df1_no_outliers["ApplicantIncome"])
plt.title("ApplicantIncome without Outliers")
plt.show()
