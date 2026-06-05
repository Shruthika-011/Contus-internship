import csv
import numpy as np
numarr=[]
with open("Titanic-Dataset.csv" ,"r") as file:
    lines=csv.DictReader(file)
    for a in lines:
        if a["Age"]!="":
            numarr.append(np.nan  )

       
age=np.array(numarr)       
age=np.where(np.isnan(age), meanage, age)



meanage=np.mean(age)
medianage=np.median(age)
stdage=np.std(age)
print("mean:", meanage)
print("median:", medianage)
print("stdage:", std)