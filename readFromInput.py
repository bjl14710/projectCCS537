import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, delaunay_plot_2d




xCoord = []
yCoord = []
zCoord = []
ListaDelDistancia = []
# with open('testData - Sheet1.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Coloumn names are {", ".join(row)}')
#             line_count += 1
#         else:
#             xCoord.append(row[1])
#             yCoord.append(row[2])
#             line_count += 1
#     print(f'Processed {line_count} lines.')

with open('data/hygdata_v3.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count < 2:
            # print(f'Coloumn names are {", ".join(row)}')
            line_count += 1
        elif line_count < 10:
            xCoord.append(row[17])
            yCoord.append(row[18])
            zCoord.append(row[19])
            ListaDelDistancia.append(row[9])
            line_count += 1
        else:
            break
    print(f'Processed {line_count} lines.')

# This data base has 119615 coloumns, so need to make this algorithm faster or stronger to perform 
# This triangulation, right now, added line_count to it because want to only get the first 200 items

# print(str(xCoord))
# print(str(yCoord))
def constructPoints(x,y):
    points = []
    for i in range(len(x)):
        points.append((x[i],y[i]))
    return points
def constructArray(x,y,z):
    arr = []
    for i in range(len(x)):
        arr.append((x[i],y[i],z[i]))
    return arr


arr = constructArray(xCoord,yCoord,zCoord)
# print('array is ' + str(arr))
tri = Delaunay(arr) # points: np.array() of 3d points 
indices = tri.simplices
print('3d triangulation is ' + str(tri))
# print(str(indices))
# vertices = arr[indices]

'''
points = constructPoints(xCoord,yCoord)
# print(str(points))
tri = Delaunay(points)
_ = delaunay_plot_2d(tri)
plt.scatter(xCoord,yCoord)
'''
plt.show()
