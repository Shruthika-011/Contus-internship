import pandas as pd

rd=pd.read_csv("Titanic-Dataset.csv")
filter=rd[ (rd["Sex"]== "female") & (rd["Age"] >=18) & ( rd["Age"] <=35)]
print(filter)