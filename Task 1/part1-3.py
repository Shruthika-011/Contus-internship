import csv
import json
with open("Titanic-Dataset.csv" , "r") as file1:
    lines=csv.DictReader(file1)
    data=list(lines)


with open("tit.json" , "w") as file2:
    json.dump(data,file2,indent=4) 
 

 