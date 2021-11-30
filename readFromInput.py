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
distList = []
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
#             distList.append(row[2])
#             line_count += 1
#     print(f'Processed {line_count} lines.')

with open('data/hygdata_v3.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count < 2:
            # print(f'Coloumn names are {", ".join(row)}')
            line_count += 1
        elif line_count < 14:
            xCoord.append(row[17])
            distList.append(row[9])
            zCoord.append(row[19])
            # xCoord[i] = row[17]
            # distList[i] = row[18]
            # zCoord[i] = row[19]
            
            ListaDelDistancia.append(row[9])
            line_count += 1
        else:
            break
    print(f'Processed {line_count} lines.')

# This data base has 119615 coloumns, so need to make this algorithm faster or stronger to perform 
# This triangulation, right now, added line_count to it because want to only get the first 200 items

# print(str(xCoord))
# print(str(distList))
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
# y = 2.0 * distList - 1
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

# ax.plot3D(xCoord,distList,zCoord,'gray')

# ax.plot_trisurf(xCoord, distList, zCoord, color='white', edgecolors='grey', alpha=0.5)
# ax.scatter3D(xCoord, distList, zCoord, color = 'b')

# ax.plot(xCoord, distList, zCoord, color='r')
### Constructing the graph
arr = constructArray(xCoord,distList,zCoord)
# print('array is ' + str(arr))
# tri = Delaunay(arr) # points: np.array() of 3d points 
# tri = triangulation(xCoord,distList,zCoord) # points: np.array() of 3d points 
# indices = tri.simplices
# print('3d triangulation is ' + str(tri))
# print(str(indices))
# vertices = arr[indices]


# points = constructPoints(xCoord,distList)
# # print(str(points))
# tri = Delaunay(points)
# _ = delaunay_plot_2d(tri)
# plt.scatter(xCoord,distList)

# z = np.random.randint(100, size =(50))
# x = np.random.randint(80, size =(50))
# y = np.random.randint(60, size =(50))

# x = [] * len(xCoord)
# y = [] * len(distList)
# z = [] * len(zCoord)



# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")


# ax.scatter3D(xCoord,distList,zCoord, color = "green")
# ax.scatter3D(xCoord,distList,zCoord)

# plt.show()


# fig = plt.figure(figsize=(4,4))

# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(xCoord)):
#     ax.scatter(xCoord[i],distList[i],zCoord[i])

# plt.show()



# z_line = np.linspace(0, 15, 1000)
# x_line = np.cos(z_line)
# y_line = np.sin(z_line)
# ax.plot3D(x_line, y_line, z_line, 'gray')

z_points = 15 * np.random.random(len(zCoord))
# x_points = np.cos(z_points) + 0.1 * np.random.randn(100)
# y_points = np.sin(z_points) + 0.1 * np.random.randn(100)

xCoord = np.array(xCoord,dtype = float)

distList = np.array(distList,dtype = float)

zCoord = np.array(zCoord,dtype = float)


distanceList1 = []
distanceList2 = []
distanceList3 = []
distanceList4 = []
distanceList5 = []
distanceList6 = []

def average(x,y,z):
    return (x+y+z)/3


def plot_triangulation(ax,points,tri):
    for tri in tri.simplices:
        pts = points[tri, :]
        ax.plot3D(pts[[0,1],0], pts[[0,1],1], pts[[0,1],2], color='g', lw='0.1')
        distanceList1.append(distance(pts[0,0], pts[0,1], pts[0,2],pts[1,0], pts[1,1], pts[1,2]))
        ax.plot3D(pts[[0,2],0], pts[[0,2],1], pts[[0,2],2], color='g', lw='0.1')
        distanceList2.append(distance(pts[0,0], pts[0,1], pts[0,2],pts[2,0], pts[2,1], pts[2,2]))
        ax.plot3D(pts[[0,3],0], pts[[0,3],1], pts[[0,3],2], color='g', lw='0.1')
        distanceList3.append(distance(pts[0,0], pts[0,1], pts[0,2],pts[3,0], pts[3,1], pts[3,2]))
        ax.plot3D(pts[[1,2],0], pts[[1,2],1], pts[[1,2],2], color='g', lw='0.1')
        distanceList4.append(distance(pts[1,0], pts[1,1], pts[1,2],pts[2,0], pts[2,1], pts[2,2]))        
        ax.plot3D(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2], color='g', lw='0.1')
        distanceList5.append(distance(pts[1,0], pts[1,1], pts[1,2],pts[3,0], pts[3,1], pts[3,2]))
        ax.plot3D(pts[[2,3],0], pts[[2,3],1], pts[[2,3],2], color='g', lw='0.1')
        distanceList6.append(distance(pts[2,0], pts[2,1], pts[2,2],pts[3,0], pts[3,1], pts[3,2]))
        # distanceList.append(distance(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2],pts[[2,3],0], pts[[2,3],1], pts[[2,3],2]))    
    print(str(distanceList3))

    ax.scatter(points[:,0], points[:,1], points[:,2], color='b')

distAverages = []
def constructAverage():
    # distAverages = []
    # Collecting the averages of the segment lengths for each point.
    for i in range(len(distanceList1)):
        distAverages.append(average(distanceList1[i],distanceList2[i],distanceList3[i]))
    return distAverages

# correspondingPoint = -1
def smallestAverage():
    min = 99999999
    correspondingPoint = -1
    # for i in range(len(distanceList1)):
    #     distAverages.append(average(distanceList1[i],distanceList2[i],distanceList3[i]))
    for i in range(len(distAverages)):
        if distAverages[i] < min:
            min = distAverages[i]
            correspondingPoint = i
    return correspondingPoint

    # points_init = np.array(xCoord,distList, dtype = float)

    # points = np.array((xCoord,distList,zCoord), dtype = float)

def cartesianToSpherical(x,z,distance):
    # x = rcos(theta)
    # y = rsin(theta)
    # WE KNOW R!!!! IT IS THE DISTANCE
    # We don't know r. But the distance applies to the 
    # sphere to some extent.
    # We can make these spherical coordinates using distance
    # as the r, then finding the new spherical coordinates with the
    # next spherical coordinates. finding the z value using the distance
    # and x and y values. 
    # x and y from center of graph in the view of the telescope
    # with a distance given in the csv file. z is new y?????

    # ok. Well we will have to ignore y because it is acting very weird.
    '''
    Z
    |
    |
    |  y
    | /
    |/__________X
    '''
    # This complicates things with our finding of z. or technicially y but here we are
    # treating z as if it is y for simplicity. So we need to find a new z based on the distance
    # From earth. We know x, y, and distance. We can find theta with that and phi with theta and those
    r = distance
    phi = math.acos(z/r)
    theta = math.acos(x/(r*math.sin(phi)))
    # theta = np.arccos(z/r)

    # This should help: https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates
    return r,theta,phi  

def sphericalToCartesian(r,theta,phi):
    x = r*math.cos(theta)*math.sin(phi)
    y = r*math.sin(theta)*math.cos(phi)
    z = r*math.cos(phi)
    return x,y,z

def multipleCartToSphere(x,z,dist,n):
    points = []
    sphere = []
    for i in range(n):
        sphere.append(cartesianToSpherical(x[i],z[i],dist[i])) 
        points.append(sphericalToCartesian(sphere[i][0],sphere[i][1],sphere[i][2]))
    return points

points = np.vstack([xCoord,distList,zCoord]).T
# tri = Delaunay(points)

fig = plt.figure()
ax = plt.axes(projection="3d")

# cartesian to spherical and back for finding y

otherPoints = multipleCartToSphere(xCoord,zCoord,distList,len(xCoord))
otherPoints = np.vstack([otherPoints[0],otherPoints[1],otherPoints[2]]).T
tri = Delaunay(otherPoints)
# Testing with the new found coordinates


plot_triangulation(ax, points,tri)

constructAverage()
point = smallestAverage() 

print("The point of the smallest average is " + str(point))

# ax.scatter3D(xCoord, distList, zCoord, c=z_points, cmap='hsv')

plt.show()

# https://stackoverflow.com/questions/64530316/euclidean-distance-of-delaney-triangulation-scipy




