import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gcsopt import GraphOfConvexSets
from gcsopt.gurobipy.plot_utils import plot_optimal_value_bounds

# Generate random rooms.
np.random.seed(0)
sides = np.array([60, 15])
L = np.zeros((*sides, 2))
U = np.zeros((*sides, 2))
for i in range(sides[0]):
    for j in range(sides[1]):
        c = (i, j)
        if (i + j) % 2 == 0:
            box_sides = [2, 1]
        else:
            box_sides = [1, 2]
        diag = np.multiply(np.random.uniform(2/3, 1, 2), box_sides) / 2
        L[i, j] = c - diag
        U[i, j] = c + diag
L = np.vstack(L)
U = np.vstack(U)

# Rooms.
main_room = 0

# Initialize empty graph.
graph = GraphOfConvexSets() # Directed by default.

# Add vertices.
for i, (l, u) in enumerate(zip(L, U)):
    v = graph.add_vertex(i)
    x = v.add_variable(2)
    c = (l + u) / 2
    v.add_constraint(x >= l)
    v.add_constraint(x <= u)
    D = np.diag(1 / (u - l))
    v.add_cost(cp.norm_inf(D @ (x - c)))

# Add edges.
def room_intersect(i, j):
    return np.all(L[i] <= U[j]) and np.all(L[j] <= U[i])
for i, (li, ui) in enumerate(zip(L, U)):
    for j in range(len(L)):
        if i != j and j != main_room and room_intersect(i, j):
            tail = graph.get_vertex(i)
            head = graph.get_vertex(j)
            e = graph.add_edge(tail, head)
            x_head = head.variables[0]
            e.add_constraint(x_head >= li)
            e.add_constraint(x_head <= ui)

# Solve problem with gurobipy (way too big for deafault MSTP method).
import importlib.util
assert importlib.util.find_spec("gurobipy")
from gcsopt.gurobipy.graph_problems.minimum_spanning_tree import minimum_spanning_tree
root = graph.vertices[main_room]
params = {"OutputFlag": 0}
plot_bounds = True
minimum_spanning_tree(graph, root, gurobi_parameters=params, save_bounds=plot_bounds)
print("Problem status:", graph.status)
print("Optimal value:", graph.value)

# Plot upper and lower bounds from gurobi.
if plot_bounds:
    plot_optimal_value_bounds(graph, "surveillance_bounds")

# Plot rooms and optimal spanning tree.
plt.figure(figsize=sides/2)
plt.axis("off")
for l, u in zip(L, U):
    rect = patches.Rectangle(l, *(u - l),
        edgecolor="k", facecolor="mintcream", alpha=.5)
    plt.gca().add_patch(rect)
graph.plot_2d_solution()
plt.xlim([-1, sides[0]])
plt.ylim([-1, sides[1]])
plt.savefig("surveillance.pdf", bbox_inches="tight")
plt.show()