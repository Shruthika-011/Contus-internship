import pandas as pd

rd=pd.read_csv("Titanic-Dataset.csv")
grp=rd.groupby("Pclass")[["Survived", "Age", "Fare"]].mean()
print(grp)