import math, random
from tabnanny import verbose
from typing import List, Tuple
from PIL import Image, ImageDraw
import numpy as np
from shapely.geometry import LinearRing, Polygon, Point
import matplotlib.pyplot as plt
from math import atan2
from cv2 import findHomography, perspectiveTransform
import barycentric

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

TWO_PI = 2 * np.pi

def is_convex_polygon(polygon):
    """Return True if the polynomial defined by the sequence of 2D
    points is 'strictly convex': points are valid, side lengths non-
    zero, interior angles are strictly between zero and a straight
    angle, and the polygon does not intersect itself.

    NOTES:  1.  Algorithm: the signed changes of the direction angles
                from one side to the next side must be all positive or
                all negative, and their sum must equal plus-or-minus
                one full turn (2 pi radians). Also check for too few,
                invalid, or repeated points.
            2.  No check is explicitly done for zero internal angles
                (180 degree direction-change angle) as this is covered
                in other ways, including the `n < 3` check.
    """
    try:  # needed for any bad points or direction changes
        # Check for too few points
        if len(polygon) < 3:
            return False
        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = atan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        # Check each point (the side ending there, its angle) and accum. angles
        for ndx, newpoint in enumerate(polygon):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = atan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  # repeated consecutive points
            # Calculate & check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -np.pi:
                angle += TWO_PI  # make it in half-open interval (-Pi, Pi]
            elif angle > np.pi:
                angle -= TWO_PI
            if ndx == 0:  # if first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # not both pos. or both neg.
                    return False
            # Accumulate the direction-change angle
            angle_sum += angle
        # Check that the total number of full turns is plus-or-minus 1
        return abs(round(angle_sum / TWO_PI)) == 1
    except (ArithmeticError, TypeError, ValueError):
        return False  # any exception means not a proper convex polygon


def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points

# this one has 2 fixed points 
def generate_polygon_2(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    points.append([-1,-1])
    points.append([1, -1])
    angle = random.uniform(0, math.pi)
    for i in range(2):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points


def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles


def plot_poly(vertices):

    poly = Polygon(vertices)
    X, Y = poly.exterior.xy
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X, Y, color='#0047AB', alpha=0.7,
            linewidth=3, solid_capstyle='round', zorder=2)
    # pyplot.xlim([-2, 2])
    # pyplot.ylim([-2, 2])
    ax.set_title('Polygon')
    plt.show()


def plot_polys(polys, w, h):

    from shapely.geometry.polygon import LinearRing, Polygon

    fig, ax = plt.subplots(w, h)
    n=0
    for poly in polys:
        n+=1
        poly = Polygon(poly)

        X, Y = poly.exterior.xy

        ax = plt.subplot(w, h, n) #aspect='equal')
        ax.plot(X,Y)
        ax.plot(X[0:2], Y[0:2], color='red')
        #ax.set_title(str(n), fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
            
        # plt.subplots_adjust(left=0.1,
        #                     bottom=0.1, 
        #                     right=0.9, 
        #                     top=0.9, 
        #                     wspace=0.4, 
        #                     hspace=0.4)

    plt.show()


def center_polys(vertices_list):
    centered_polys = []
    for vertices in vertices_list:
        centroid = vertices.mean(axis=0)
        #print('centroid: ' + str(centroid)) 
        vertices_centered = vertices - centroid
        # print(vertices_centered)
        centered_polys.append(vertices_centered)
        # shifted_x = np.array([vertices[:,0] - centroid[0]])
        # shifted_y = np.array([vertices[:,1] - centroid[1]])
        # vertices = np.concatenate((shifted_x.T, shifted_y.T), axis=1)
        new_centroid = vertices_centered.mean(axis=0)
        #print('new centroid: ', new_centroid)
    return np.array(centered_polys)


def rotate_poly(vertices, angle):
    rot_matrix = np.array([ [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)] ])
    #print(rot_matrix)
    rot_poly = []

    for vertex in vertices:
        rot_poly.append(np.matmul(rot_matrix, vertex.T))
    #print(np.array(rot_poly))
    return np.array(rot_poly)


def linear_transform(poly1, poly2, points1):
    h, status = findHomography(poly1, poly2)
    points2 = perspectiveTransform(points1[None, :, :], h)
    return(points2[0])


def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            points.append(random_point)

    list_array = []
    for p in points:
        list_array.append([p.x, p.y])
    
    return np.array(list_array)


######################################################
# # # ## LINEAR TRANSFORM OF RANDOM POINTS INSIDE POLY
######################################################
# # # poly1 = np.array([[6.731380187456560416e-01, 3.113885087097812976e-01], [1.587972751467133126e+00, 7.939554414114522451e-01], [-1.134644575731510230e+00, -5.441359451227235633e-01], [-1.126466194481278826e+00, -5.612080049985100905e-01]])
# # # poly2 = np.array([[8.281889960601291900e-01, 6.182037220283929102e-01], [-1.494848472439794129e+00, -9.091524571767214402e-01], [2.394477529346155653e-01, 8.694824187537493476e-02], [4.272117234450494294e-01, 2.040004932729536646e-01]])


# square = np.array([ [-1,-1], [1, -1], [1,1], [-1, 1]  ], dtype=np.float32)
# # # poly2 = np.array([[6.731380187456560416e-01, 3.113885087097812976e-01], [1.587972751467133126e+00, 7.939554414114522451e-01], [-1.134644575731510230e+00, -5.441359451227235633e-01], [-1.126466194481278826e+00, -5.612080049985100905e-01]])
# # #poly2 =  np.array([ [-1.5,-1.5], [1, -1], [1.5,1.5], [-1, 1]  ], dtype=np.float32)


# dir_path = r'./linearly_transformed_quads_from_regular_square_grid_barycentric/'
# #np.savetxt(dir_path+'initial_square', points1)
# square_grid = np.genfromtxt(r'./test_uniform_square_d500_b200.dat')[:,0:2]
# polys = np.genfromtxt('quads50a_sp01_ir05_train.txt')
# #polys = np.genfromtxt('quads_centered.txt')
# polys = polys.reshape(len(polys), 4, 2)
# r,c = 4,10

# square_grid_barycentric = barycentric.make_barycentric(square, square_grid)
# np.savetxt('barycentric_unit_square_d500_b200.dat', square_grid_barycentric)
# print(square_grid_barycentric.shape)

# fig, ax = plt.subplots(r, c)

# n=0
# # for poly in polys:
# #     if n==0:
# #         points1 = random_points_within(Polygon(poly), 500)
# #         poly1 = np.copy(poly)
# #     else:
# #         poly2 = np.copy(poly)
# #         points2 = linear_transform(poly1, poly2, points1)
# #         poly1 = np.copy(poly2)
# #         points1 = np.copy(points2)

# # ax = plt.subplot(r, c, 1)
# # plt.title(str(0))
# # ax.scatter(square_grid[:,0], square_grid[:, 1], s=0.6)
# # n+=1
    
# for poly in polys:

#     transform_points = np.dot(square_grid_barycentric, poly)
#     ax = plt.subplot(r, c, n+1)
#     plt.title(str(n))
#     poly_plot = Polygon(poly)
#     X, Y = poly_plot.exterior.xy
#     plt.plot(X, Y, '--', color='k', linewidth='0.7')

#     # pyplot.xlim([-2, 2])
#     # pyplot.xlim([-2, 2])
#     plt.scatter(transform_points[:,0], transform_points[:, 1], s=0.6)


#     n+=1
#     # np.savetxt(dir_path+str(n), points1)
# plt.show()

###############################################
####### TRANSFORM USING BARYCENTRIC COORDS
##############################################




###############################################################
# GENERATE RANDOM QUADS
###############################################################

quadrangle_list = []
sp=0.1 # spikiness
ir=0.5 # irregularity
quad_num = 100
while len(quadrangle_list) < quad_num:
    vertices = generate_polygon(center=(0, 0),
                                avg_radius=1,
                                irregularity=ir,
                                spikiness=sp,
                                num_vertices=4)

    # if len(np.unique(vertices, axis=0)) < len(vertices):
    #     continue
    if is_convex_polygon(vertices):
        quadrangle_list.append(vertices)
    else:
        print('Concave quad')
    #print('vertices: ' + str(vertices))

quadrangle_list = np.array(quadrangle_list)
quadrangle_list = center_polys(quadrangle_list)

# plot_polys(quadrangle_list, round(quad_num/5), 5)

# np.savetxt('quads'+ str(quad_num)+ '_sp' + str(round(sp*10))+'_ir' + str(round(ir*10)) + '.txt', quadrangle_list.reshape(quad_num, 8))

#################################################################


# qg = np.genfromtxt('quads_centered.txt')
# qg = qg.reshape(len(qg), 4, 2)
qg = quadrangle_list
rotated = []


for poly in qg:

    angle = -np.arctan2( poly[1,1]-poly[0,1], poly[1,0]-poly[0,0])
    rotated_poly = rotate_poly(poly, angle)

    rotated_poly = rotate_poly(poly, angle)
    rotated.append(rotated_poly)

rotated = np.array(rotated)

plot_polys(rotated[0::5], round(quad_num/(5*5)), 5)
rotated = rotated.reshape(quad_num, 8)
#np.savetxt('quads'+ str(quad_num)+ 'a' + '_sp' + str(round(sp*10))+'_ir' + str(round(ir*10)), rotated)

########################################
## PLOT POLYS 
########################################

# qg = np.genfromtxt('quads50a_sp01_ir05_train.txt')
# qg = qg.reshape(len(qg), 4, 2)

# plot_polys(np.array(qg[:9]), 3, 3)