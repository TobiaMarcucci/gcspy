import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from gcs import GraphOfConvexSets

# random points to be clustered
num_clusters = 3
num_points = 20
dim = 2
np.random.seed(0)
points = np.random.rand(num_points, dim)

# initialize empty graph
graph = GraphOfConvexSets()

# one vertex per cluster
clusters = []
for i in range(num_clusters):
    cluster = graph.add_vertex(f"cluster_{i}")
    x = cluster.add_variable(dim)
    cluster.add_constraint(x >= 0)
    cluster.add_constraint(x <= 1)
    clusters.append(cluster)

# one vertex per data point
data_points = []
for i, point in enumerate(points):
    data_point = graph.add_vertex(f"point_{i}")
    x = data_point.add_variable(dim)
    data_point.add_constraint(x == point)
    data_points.append(data_point)
    
# one edge from every cluster to every data point
for cluster in clusters:
    for data_point in data_points:
        edge = graph.add_edge(cluster, data_point)
        distance = cluster.variables[0] - data_point.variables[0]
        edge.add_cost(cp.sum_squares(distance))

# solve problem
prob = graph.solve_facility_location()
print("Problem status:", prob.status)
print("Optimal value:", prob.value)

# plot optimal clustering
plt.figure()
plt.axis("equal")
cluster_colors = ['r', 'g', 'b']
assert len(cluster_colors) == num_clusters
for cluster, color in zip(clusters, cluster_colors):

    # mark center of the cluster with an x
    x = cluster.variables[0].value
    plt.scatter(*x, color=color, marker='x')

    # scatter with same color all the points in the cluster
    for data_point in data_points:
        edge = graph.get_edge(cluster.name, data_point.name)
        if np.isclose(edge.binary_variable.value, 1):
            x = data_point.variables[0].value
            plt.scatter(*x, color=color, marker='o')

plt.show()
