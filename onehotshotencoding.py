import pandas as pd

# Step 1: Read CSV
dataset = pd.read_csv(r"C:\Users\Vinay kumar\Desktop\Diabetic prediction system\loan.csv")

# Step 2: Check missing values
print("Missing values:\n", dataset.isnull().sum())

# Step 3: Fill missing values in 'Gender'
dataset["Gender"].fillna(dataset["Gender"].mode()[0], inplace=True)

# Step 4: Fill missing values in 'Married'
dataset["Married"].fillna(dataset["Married"].mode()[0], inplace=True)

# Step 5: Select columns for encoding
en_data = dataset[["Gender", "Married"]]
print("\nSelected columns for encoding:\n", en_data.head())

# Step 6: Get dummies (Pandas one-hot encoding)
print("\nPandas get_dummies info:")
pd.get_dummies(en_data).info()

# Step 7: OneHotEncoding using sklearn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

# Step 8: Fit and transform
ar = ohe.fit_transform(en_data).toarray()

# Step 9: Create DataFrame from encoded array
encoded_df = pd.DataFrame(ar, columns=["Gender_Female", "Gender_Male", "Married_No", "Married_Yes"])
print("\nOneHotEncoded DataFrame:\n", encoded_df.head())
