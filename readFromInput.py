import csv

xCoord = []
yCoord = []
with open('testData - Sheet1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Coloumn names are {", ".join(row)}')
            line_count += 1
        else:
            xCoord.append(row[1])
            yCoord.append(row[2])
            line_count += 1
    print(f'Processed {line_count} lines.')
    
print(str(xCoord))
print(str(yCoord))