import csv
import numpy as np
fare=[]
with open("Titanic-Dataset.csv" ,"r") as file:
    lines= csv.DictReader(file)
    for a in lines:
        if a["Fare"]=="":
            fare.append(np.nan)
        else:
            fare.append(float(a["Fare"]))

f=np.array(fare)
                
mini=np.min(f)
maxi=np.max(f)

normalised=(f-mini)/(maxi-mini)
print(normalised[:10])