import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from gcspy import GraphOfConvexPrograms

# triangular mesh for 2d robot link
mesh = np.array([
    [[1, 0], [0, 2], [0, 1]],
    [[1, 0], [0, 2], [1, 3]],
    [[1, 0], [2, 0], [1, 3]],
    [[3, 1], [2, 0], [1, 3]],
    [[3, 1], [3, 3], [1, 3]],
    [[3, 1], [3, 3], [5, 3]],
    [[3, 1], [5, 3], [5, 1]],
    [[7, 1], [5, 3], [5, 1]],
    [[7, 3], [5, 3], [7, 1]],
    [[7, 3], [9, 3], [7, 1]],
    [[9, 3], [9, 1], [7, 1]],
    [[13, 1], [9, 3], [9, 1]],
    [[13, 1], [9, 3], [14, 3]],
    [[10, 5], [9, 3], [14, 3]],
    [[10, 5], [14, 5], [14, 3]],
    [[10, 5], [14, 5], [11, 6]],
    [[13, 6], [14, 5], [11, 6]],
])

# problem parameters
circle_cost = 0 # fixed cost of using a circle
num_circles = 5 # maximum number of circles

# initialize empty graph
gcs = GraphOfConvexPrograms()

# compute bounding box for entire mesh
l = np.min(np.vstack(mesh), axis=0)
u = np.max(np.vstack(mesh), axis=0)

# upper bound for circle radius
max_radius = np.linalg.norm(u - l) / 2

# add all circles (facilities)
circles = []
for i in range(num_circles):
    circle = gcs.add_vertex(f"c{i}")
    center = circle.add_variable(2)
    radius = circle.add_variable(1)
    circle.add_constraints([center >= l, center <= u])
    circle.add_constraints([radius >= 0, radius <= max_radius])
    circle.add_cost(circle_cost + np.pi * radius ** 2)
    circles.append(circle)

# add all triangles (clients)
triangles = []
for i in range(len(mesh)):
    triangle = gcs.add_vertex(f"t{i}")
    slack = triangle.add_variable(1) # necessary to add at least one variable
    triangle.add_constraint(slack == 0)
    triangles.append(triangle)

# add edge from every circle to every triangle
for circle in circles:
    center, radius = circle.variables
    for points, triangle in zip(mesh, triangles):
        edge = gcs.add_edge(circle, triangle)
        for point in points:
            edge.add_constraint(cp.norm2(point - center) <= radius)

# solve problem
prob = gcs.solve_facility_location()
print('Problem status:', prob.status)
print('Optimal value:', prob.value)

# plot solution
plt.figure()
plt.axis("square")
limits = np.array([l - 1, u + 1])
plt.xlim(limits[:, 0])
plt.ylim(limits[:, 1])

# plot the mesh
for triangle in mesh:
    patch = plt.Polygon(triangle[:3], fc="mintcream", ec="k")
    plt.gca().add_patch(patch)

# plot the circle cover
for circle in circles:
    if np.isclose(circle.binary_variable.value, 1):
        center, radius = circle.variables
        patch = plt.Circle(center.value, radius.value, fc="None", ec="b")
        plt.gca().add_patch(patch)

plt.show()