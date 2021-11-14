import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
# from scipy.spatial import Delaunay, delaunay_plot_2d
from matplotlib.tri import triangulation
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits import mplot3d




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



def plot_tri_simple(ax, points, tri):
    for tr in tri.simplices:
        pts = points[tr,: ]
    ax.plot3D(pts[[0, 1], 0], pts[[0, 1], 1], pts[[0, 1], 2], color = 'g', lw = '0.1')
    ax.plot3D(pts[[0, 2], 0], pts[[0, 2], 1], pts[[0, 2], 2], color = 'g', lw = '0.1')
    ax.plot3D(pts[[0, 3], 0], pts[[0, 3], 1], pts[[0, 3], 2], color = 'g', lw = '0.1')
    ax.plot3D(pts[[1, 2], 0], pts[[1, 2], 1], pts[[1, 2], 2], color = 'g', lw = '0.1')
    ax.plot3D(pts[[1, 3], 0], pts[[1, 3], 1], pts[[1, 3], 2], color = 'g', lw = '0.1')
    ax.plot3D(pts[[2, 3], 0], pts[[2, 3], 1], pts[[2, 3], 2], color = 'g', lw = '0.1')

    ax.scatter(points[: , 0], points[: , 1], points[: , 2], color = 'b')

### calling with random points
np.random.seed(0)
x = 2.0 * np.random.rand(20) - 1.0
y = 2.0 * np.random.rand(20) - 1.0
z = 2.0 * np.random.rand(20) - 1.0
points = np.vstack([x, y, z]).T
tri = Delaunay(points)

fig = plt.figure()
ax = plt.axes(projection = '3d')
plot_tri_simple(ax, points, tri)




### plotting 




### Constructing the graph
arr = constructArray(xCoord,yCoord,zCoord)
# print('array is ' + str(arr))
# tri = Delaunay(arr) # points: np.array() of 3d points 
# tri = triangulation(xCoord,yCoord,zCoord) # points: np.array() of 3d points 
# indices = tri.simplices
# print('3d triangulation is ' + str(tri))
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
