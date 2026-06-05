with open("Titanic-Dataset.csv" ,"r") as file:
    line= file.readlines()
    head = line[0].strip().split(",")
    sur_in = head.index("Survived")
    with open("su.csv" ,"w") as file1:
        for a in line:
            row = a.strip().split(",")
            if row[sur_in]=="1":

                file1.write(a)