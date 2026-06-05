import pandas as pd

rd=pd.read_csv("Titanic-Dataset.csv")
rd["FamilySize"]=rd["SibSp"]+rd["Parch"]
result=rd.groupby("FamilySize")[["Survived"]].mean

print(result)