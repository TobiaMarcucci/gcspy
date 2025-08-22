import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gcspy import GraphOfConvexSets
from surveillance_utils import L, U

# Rooms.
main_room = 0
n_rooms = len(L)

# Initialize empty graph.
graph = GraphOfConvexSets() # Directed by default.

# Add vertices.
for i, (l, u) in enumerate(zip(L, U)):
    v = graph.add_vertex(i)
    x = v.add_variable(2)
    c = (l + u) / 2
    v.add_constraint(x >= l)
    v.add_constraint(x <= u)
    # D = np.diag(1 / (u - l))
    # v.add_cost(cp.norm_inf(D @ (x - c)))
    v.add_cost(cp.norm2(x - c))

# Add edges.
def room_intersect(i, j):
    return np.all(L[i] <= U[j]) and np.all(L[j] <= U[i])
for i, (li, ui) in enumerate(zip(L, U)):
    for j in range(n_rooms):
        if i != j and j != main_room and room_intersect(i, j):
            tail = graph.get_vertex(i)
            head = graph.get_vertex(j)
            e = graph.add_edge(tail, head)
            x_head = head.variables[0]
            e.add_constraint(x_head >= li)
            e.add_constraint(x_head <= ui)

# Solve problem with gurobipy.
root = graph.vertices[main_room]
prob = graph.solve_minimum_spanning_tree_gurobipy(root)
print("Problem status:", prob.status)
print("Optimal value:", prob.ObjVal)

# Show graph using graphviz (requires graphviz).
dot = graph.graphviz()
dot.view()

# Plot rooms.
plt.figure()
plt.axis("equal")
for l, u in zip(L, U):
    rect = patches.Rectangle(l, *(u - l),
        edgecolor="k", facecolor="lavender", alpha=.5)
    plt.scatter(*(l + u) / 2, marker="x", color="red")
    plt.gca().add_patch(rect)
graph.plot_2d_solution()
plt.show()