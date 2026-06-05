import csv
with open("Titanic-Dataset.csv" ,"r") as file:
    reader=csv.DictReader(file)
    row = list(reader)
    for a in row[5]:
        print(a)