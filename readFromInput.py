import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, delaunay_plot_2d




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
def constructPoints(x,y):
    points = []
    for i in range(len(x)):
        points.append((x[i],y[i]))
    return points

points = constructPoints(xCoord,yCoord)
# print(str(points))
tri = Delaunay(points)
_ = delaunay_plot_2d(tri)
plt.scatter(xCoord,yCoord)

plt.show()
