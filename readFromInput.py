import csv
import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
# from scipy.spatial import Delaunay, delaunay_plot_2d
from matplotlib.tri import triangulation
from scipy.spatial import ConvexHull, Delaunay, delaunay_plot_2d
from mpl_toolkits import mplot3d


class Coord:
    def __init__(self,point,size):
        self.point = point
        self.size = size

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
        elif line_count < 11:
            xCoord.append(row[17])
            yCoord.append(row[18])
            zCoord.append(row[19])
            # xCoord[i] = row[17]
            # yCoord[i] = row[18]
            # zCoord[i] = row[19]
            
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

def distance(x1,y1,z1,x2,y2,z2):
    deltX = (x2-x1)
    deltY = (y2-y1)
    deltZ = (z2-z1)
    return math.sqrt(deltX*deltX + deltY*deltY + deltZ*deltZ)

# def plot_tri_simple(ax, points, tri):
#     for tr in tri.simplices:
#         pts = points[tr,: ]
#     ax.plot3D(pts[[0, 1], 0], pts[[0, 1], 1], pts[[0, 1], 2], color = 'g', lw = '0.1')
#     ax.plot3D(pts[[0, 2], 0], pts[[0, 2], 1], pts[[0, 2], 2], color = 'g', lw = '0.1')
#     ax.plot3D(pts[[0, 3], 0], pts[[0, 3], 1], pts[[0, 3], 2], color = 'g', lw = '0.1')
#     ax.plot3D(pts[[1, 2], 0], pts[[1, 2], 1], pts[[1, 2], 2], color = 'g', lw = '0.1')
#     ax.plot3D(pts[[1, 3], 0], pts[[1, 3], 1], pts[[1, 3], 2], color = 'g', lw = '0.1')
#     ax.plot3D(pts[[2, 3], 0], pts[[2, 3], 1], pts[[2, 3], 2], color = 'g', lw = '0.1')

#     ax.scatter(points[: , 0], points[: , 1], points[: , 2], color = 'b')

# ### calling with random points
# np.random.seed(0)
# # x = 2.0 * np.random.rand(20) - 1.0
# # y = 2.0 * np.random.rand(20) - 1.0
# # z = 2.0 * np.random.rand(20) - 1.0
# x = 2.0 * xCoord - 1
# y = 2.0 * yCoord - 1
# z = 2.0 * zCoord - 1

# points = np.vstack([x, y, z]).T
# tri = Delaunay(points)

# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# plot_tri_simple(ax, points, tri)




### second attempt




# fig = plt.figure()

# fig = plt.figure(figsize = (10,10))

# ax = fig.add_subplot(projection='3d')
# ax = plt.axes(projection='3d')

# ax.plot3D(xCoord,yCoord,zCoord,'gray')

# ax.plot_trisurf(xCoord, yCoord, zCoord, color='white', edgecolors='grey', alpha=0.5)
# ax.scatter3D(xCoord, yCoord, zCoord, color = 'b')

# ax.plot(xCoord, yCoord, zCoord, color='r')
### Constructing the graph
arr = constructArray(xCoord,yCoord,zCoord)
# print('array is ' + str(arr))
# tri = Delaunay(arr) # points: np.array() of 3d points 
# tri = triangulation(xCoord,yCoord,zCoord) # points: np.array() of 3d points 
# indices = tri.simplices
# print('3d triangulation is ' + str(tri))
# print(str(indices))
# vertices = arr[indices]


# points = constructPoints(xCoord,yCoord)
# # print(str(points))
# tri = Delaunay(points)
# _ = delaunay_plot_2d(tri)
# plt.scatter(xCoord,yCoord)

# z = np.random.randint(100, size =(50))
# x = np.random.randint(80, size =(50))
# y = np.random.randint(60, size =(50))
 
# x = [] * len(xCoord)
# y = [] * len(yCoord)
# z = [] * len(zCoord)



# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")


# ax.scatter3D(xCoord,yCoord,zCoord, color = "green")
# ax.scatter3D(xCoord,yCoord,zCoord)

# plt.show()


# fig = plt.figure(figsize=(4,4))

# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(xCoord)):
#     ax.scatter(xCoord[i],yCoord[i],zCoord[i])

# plt.show()



# z_line = np.linspace(0, 15, 1000)
# x_line = np.cos(z_line)
# y_line = np.sin(z_line)
# ax.plot3D(x_line, y_line, z_line, 'gray')

z_points = 15 * np.random.random(len(zCoord))
# x_points = np.cos(z_points) + 0.1 * np.random.randn(100)
# y_points = np.sin(z_points) + 0.1 * np.random.randn(100)

xCoord = np.array(xCoord,dtype = float)

yCoord = np.array(yCoord,dtype = float)

zCoord = np.array(zCoord,dtype = float)


distanceList = []
def plot_triangulation(ax,points,tri):
    for tri in tri.simplices:
        pts = points[tri, :]
        ax.plot3D(pts[[0,1],0], pts[[0,1],1], pts[[0,1],2], color='g', lw='0.1')
        ax.plot3D(pts[[0,2],0], pts[[0,2],1], pts[[0,2],2], color='g', lw='0.1')
        distanceList.append(distance(pts[[0,1],0], pts[[0,1],1], pts[[0,1],2],pts[[0,2],0], pts[[0,2],1], pts[[0,2],2]))
        ax.plot3D(pts[[0,3],0], pts[[0,3],1], pts[[0,3],2], color='g', lw='0.1')
        ax.plot3D(pts[[1,2],0], pts[[1,2],1], pts[[1,2],2], color='g', lw='0.1')
        distanceList.append(distance(pts[[0,3],0], pts[[0,3],1], pts[[0,3],2],pts[[1,2],0], pts[[1,2],1], pts[[1,2],2]))        
        ax.plot3D(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2], color='g', lw='0.1')
        ax.plot3D(pts[[2,3],0], pts[[2,3],1], pts[[2,3],2], color='g', lw='0.1')
        distanceList.append(distance(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2],pts[[2,3],0], pts[[2,3],1], pts[[2,3],2]))    
    ax.scatter(points[:,0], points[:,1], points[:,2], color='b')


# points_init = np.array(xCoord,yCoord, dtype = float)

# points = np.array((xCoord,yCoord,zCoord), dtype = float)

points = np.vstack([xCoord,yCoord,zCoord]).T
tri = Delaunay(points)

fig = plt.figure()
ax = plt.axes(projection="3d")

plot_triangulation(ax, points,tri)

# ax.scatter3D(xCoord, yCoord, zCoord, c=z_points, cmap='hsv')

plt.show()

# https://stackoverflow.com/questions/64530316/euclidean-distance-of-delaney-triangulation-scipy




