# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
diabetes = pd.read_csv("diabetes.csv")

# Show first few rows
print("First 3 rows of original dataset:")
print(diabetes.head(3))

# Check for missing values
print("\nMissing values in each column:")
print(diabetes.isnull().sum())

# Plot original glucose distribution using distplot
plt.figure(figsize=(8, 5))
sns.distplot(diabetes['Glucose'], kde=True)
plt.title("Glucose Distribution Before Scaling")
plt.xlabel("Glucose")
plt.show()

# Initialize StandardScaler
ss = StandardScaler()

# Fit the scaler
ss.fit(diabetes[['Glucose']])

# Transform and convert to DataFrame
glucose_scaled = pd.DataFrame(
    ss.transform(diabetes[['Glucose']]),
    columns=['Glucose_ss']
)

# Add scaled column to dataset
diabetes['Glucose_ss'] = glucose_scaled

# Show first 3 rows
print("\nFirst 3 rows after Standard scaling:")
print(diabetes[['Glucose', 'Glucose_ss']].head(3))

# Show stats
print("\nDescriptive statistics of scaled Glucose:")
print(diabetes['Glucose_ss'].describe())

# Compare original and scaled using distplot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Original Glucose Distribution")
sns.distplot(diabetes['Glucose'])
plt.xlabel("Glucose")

plt.subplot(1, 2, 2)
plt.title("Scaled Glucose (StandardScaler)")
sns.distplot(diabetes['Glucose_ss'])
plt.xlabel("Glucose_ss")

plt.tight_layout()
plt.show()
