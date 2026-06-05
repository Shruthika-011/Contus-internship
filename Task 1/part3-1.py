import pandas as pd

rd=pd.read_csv("Titanic-Dataset.csv")
print("shape:", rd.shape)
print("columns:", rd.columns)
print("types:",rd.dtypes)
missing_values = rd.isnull().sum()
print(missing_values)