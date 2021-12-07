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

print(sys.path)

class Coord:
    def __init__(self,point,size):
        self.point = point
        self.size = size

class Triangulation:
    def __init__(self, point):
        self.point = point

    def lenOfSegment(self,point1,point2):
        return(math.sqrt(point1**2 + point2**2))

    def ConstructTriangle(self,points):
        tri = 0
        return tri

    def ConstructSphere(self,points):
        
        return 0

    def average(self,x,y,z):
        return (x+y+z)/3

    def cartesianToSpherical(self,x,z,distance):
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

    def sphericalToCartesian(self,r,theta,phi):
        x = r*math.cos(theta)*math.sin(phi)
        y = r*math.sin(theta)*math.cos(phi)
        z = r*math.cos(phi)
        return x,y,z

    def multipleCartToSphere(self,x,z,dist,n):
        points = []
        sphere = []
        for i in range(n):
            sphere.append(cartesianToSpherical(x[i],z[i],dist[i])) 
            points.append(sphericalToCartesian(sphere[i][0],sphere[i][1],sphere[i][2]))
        return points




    def neighbors(self,point):
# find the neighbors to use for sphere
        return 0

    def isClockwise(self, point1, point2):
        clock = True 
        # if()
        return clock
 
    


xCoord = []
distList = []
zCoord = []
AmountOfPoints = 100
with open('data/hygdata_v3.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count < 2:
            # print(f'Coloumn names are {", ".join(row)}')
            line_count += 1
        elif line_count < AmountOfPoints:
            xCoord.append(row[17])
            distList.append(row[9])
            zCoord.append(row[19])
            # xCoord[i] = row[17]
            # distList[i] = row[18]
            # zCoord[i] = row[19]
            
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

### Constructing the graph
arr = constructArray(xCoord,distList,zCoord)

z_points = 15 * np.random.random(len(zCoord))

xCoord = np.array(xCoord,dtype = float)

distList = np.array(distList,dtype = float)

zCoord = np.array(zCoord,dtype = float)


distanceList1 = []
distanceList2 = []
distanceList3 = []
distanceList4 = []
distanceList5 = []
distanceList6 = []

def average(x,y,z,xx,yy,zz):
    return (x+y+z+xx+yy+zz)/6


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
        distAverages.append(average(distanceList1[i],
                                    distanceList2[i],
                                    distanceList3[i],
                                    distanceList4[i],
                                    distanceList5[i],
                                    distanceList6[i]))
    return distAverages

# correspondingPoint = -1
def biggestAverage():
    max = -99999999
    correspondingPoint = -1
    # for i in range(len(distanceList1)):
    #     distAverages.append(average(distanceList1[i],distanceList2[i],distanceList3[i]))
    for i in range(len(distAverages)):
        if distAverages[i] > max:
            max = distAverages[i]
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
    try:
        r = distance
        phi = math.acos(z/r)
        theta = math.acos(x/(r*math.sin(phi)))
    except:
        # hit a zero, can't run the cosine
        return 0,0,0
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


def WireframeSphere(centre=[0.,0.,0.], radius=1.,
                    n_meridians=20, n_circles_latitude=None):
    """
    Create the arrays of values to plot the wireframe of a sphere.

    Parameters
    ----------
    centre: array like
        A point, defined as an iterable of three numerical values.
    radius: number
        The radius of the sphere.
    n_meridians: int
        The number of meridians to display (circles that pass on both poles).
    n_circles_latitude: int
        The number of horizontal circles (akin to the Equator) to display.
        Notice this includes one for each pole, and defaults to 4 or half
        of the *n_meridians* if the latter is larger.

    Returns
    -------
    sphere_x, sphere_y, sphere_z: arrays
        The arrays with the coordinates of the points to make the wireframe.
        Their shape is (n_meridians, n_circles_latitude).

    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> sphere = ax.plot_wireframe(*WireframeSphere(), color="r", alpha=0.5)
    >>> fig.show()

    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> frame_xs, frame_ys, frame_zs = WireframeSphere()
    >>> sphere = ax.plot_wireframe(frame_xs, frame_ys, frame_zs, color="r", alpha=0.5)
    >>> fig.show()
    """
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians/2, 4)
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    sphere_x = radius * np.cos(u) * np.sin(v) + centre[0]
    sphere_y = radius * np.sin(u) * np.sin(v) + centre[1]
    sphere_z = radius * np.cos(v) + centre[2]

    ax.plot_wireframe(sphere_x, sphere_y, sphere_z)

    return sphere_x, sphere_y, sphere_z

def sphere(center=[0.,0.,0.], radius = 1):
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    
    x = np.outer(center[0] + radius * np.sin(u),  np.sin(v))
    y = np.outer(center[1] + radius * np.sin(u),  np.cos(v))
    z = np.outer(center[2] + radius * np.cos(u),  np.ones_like(v))
    # r = min(distanceList1[point],
    #         distanceList2[point],
    #         distanceList3[point],
    #         distanceList4[point],
    #         distanceList5[point],
    #         distanceList6[point])
    # r^2 = (x-deltX)^2 + (y - deltY)^2 + (z - deltZ)^2
    # x = sqrt((y - deltY)^2 + (z - deltZ)^2) - r^2)
    # r = 10
    # deltX, deltY, deltZ = otherPoints[point][0],otherPoints[point][1],otherPoints[point][2]
    # x, y, z = r*x - deltX, r*y - deltY, r*z - deltZ

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    ax.plot_wireframe(x, y, z)

def listPoints(points):
    List = []
    for p in points:
        List.append(p[1])
    return List

points = np.vstack([xCoord,distList,zCoord]).T
# tri = Delaunay(points)

fig = plt.figure()
ax = plt.axes(projection="3d")

# cartesian to spherical and back for finding y

otherPoints = multipleCartToSphere(xCoord,zCoord,distList,len(xCoord))

yPoints = listPoints(otherPoints)
points = np.vstack([xCoord,yPoints,zCoord]).T


tri = Delaunay(points)
# Testing with the new found coordinates
# newPoints = np.vstack([otherPoints[:][0],otherPoints[:][1],otherPoints[:][2]]).T
# newPoints = np.array([otherPoints[:][0],otherPoints[:][1],otherPoints[:][2]])
newPoints = listPoints(otherPoints)

plot_triangulation(ax, points,tri)


constructAverage()
point = biggestAverage() 


# r = min(distanceList1[point],
#             distanceList2[point],
#             distanceList3[point],
#             distanceList4[point],
#             distanceList5[point],
#             distanceList6[point])

# print("other points is " + str(otherPoints[point]))
WireframeSphere(points[point],50)
print("The point of the biggest average is at " + str(point))


# plt.Circle((otherPoints[point][0],otherPoints[point][1],otherPoints[point][2]),distAverages[point])
# ax.scatter3D(xCoord, distList, zCoord, c=z_points, cmap='hsv')

plt.show()

# https://stackoverflow.com/questions/64530316/euclidean-distance-of-delaney-triangulation-scipy




