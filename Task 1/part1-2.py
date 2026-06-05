import csv
with open("Titanic-Dataset.csv", "r") as file :
    lines=csv.DictReader(file)
    with open("survior.csv", "w") as file1:
        writer = csv.DictWriter(file1, fieldnames=lines.fieldnames)
        writer.writeheader()

        for a in lines:
            if a["Survived"]=="1":
                writer.writerow(a)