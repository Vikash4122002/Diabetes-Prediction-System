import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the CSV (replace with your actual path if needed)
df = pd.read_csv(r"C:\Users\Vinay kumar\Desktop\Diabetic prediction system\loan.csv")  # Make sure to save your data as loan_data.csv

# Fill missing values in categorical columns with mode
for column in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'LoanAmount']:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Fill missing values in numerical columns with median
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Create LabelEncoder object
le = LabelEncoder()

# Columns to be label encoded
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                    'Property_Area', 'Loan_Status']

# Apply Label Encoding to each categorical column
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Show encoded DataFrame
print(df.head())
