import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\Vinay kumar\Desktop\Diabetic prediction system\diabetes.csv")
print(dataset.head(7))
print(dataset.shape)
print(dataset.isnull())
print(dataset.isnull().sum())
print(dataset.isnull().sum().sum())
print((dataset.isnull().sum()/dataset.shape[0])*100)
print((dataset.isnull().sum().sum()/(dataset.shape[0]*dataset.shape[1]))*100)
sns.heatmap(dataset.isnull())
plt.show()
dataset.drop(columns=["BMI"], inplace=True)
