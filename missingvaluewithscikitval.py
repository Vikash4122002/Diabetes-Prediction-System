# Step 1: Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Step 2: Load the dataset
dataset = pd.read_csv("mydata.csv")
print("Original Data:")
print(dataset.head(3))

# Step 3: Check missing values
print("\nMissing values in each column:")
print(dataset.isnull().sum())

# Step 4: Select numerical columns (float64 type)
num_cols = dataset.select_dtypes(include="float64").columns
print("\nNumerical Columns with Float64 Type:")
print(num_cols)

# Step 5: Apply SimpleImputer with 'mean' strategy
si = SimpleImputer(strategy="mean")
imputed_array = si.fit_transform(dataset[num_cols])

# Step 6: Convert the numpy array back to DataFrame
new_dataset = pd.DataFrame(imputed_array, columns=num_cols)

# Step 7: Replace the original numerical columns with the filled ones
dataset[num_cols] = new_dataset

# Step 8: Final check for missing values
print("\nMissing values after imputation:")
print(dataset.isnull().sum())

# Step 9: Optional – Save the cleaned data
dataset.to_csv("loan_cleaned.csv", index=False)
print("\nCleaned data saved as 'loan_cleaned.csv'")
