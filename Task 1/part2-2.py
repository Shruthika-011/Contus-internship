import csv
import numpy as np
ch=[]
ad=[]
se=[]
with open("Titanic-Dataset.csv" ,"r") as file:
    lines=csv.DictReader(file)
    for a in lines:
        if a["Age"]=="":
            continue
        age=float(a["Age"])
        if age<18 :
            ch.append(age)

        if age>=18 and age<=60:
            ad.append(age)

        if age>60 :
            se.append(age)  

child=np.array(ch)
adult=np.array(ad)
senior=np.array(se)   
print("Child:",child.size)
print("Adult:",adult.size)
print("Senior:",senior.size)         