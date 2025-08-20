import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gcspy import GraphOfConvexSets

# Problem data.
party_position = np.array([45, 7])
guest_positions = np.array([
    [47, 9],
    [52, 4],
    [52, 8],
    [47, 5],
    [48, 11],
    [43, 10],
    [42, 6],
    [44, 4]])

# Compute bounding box for all positions.
positions = np.vstack((party_position, guest_positions))
l = np.min(positions, axis=0)
u = np.max(positions, axis=0)

# Initialize empty graph.
graph = GraphOfConvexSets()

# Vertex for every guest.
for i, position in enumerate(guest_positions):
    guest = graph.add_vertex(f"guest_{i}")
    x = guest.add_variable(2)
    guest.add_constraints([x >= l, x <= u])
    guest.add_cost(cp.norm1(x - position)) # Manhattan distance traveled by the guest

# Vertex for the party location.
party = graph.add_vertex("party")
x = party.add_variable(2)
party.add_constraint(x == party_position)

# Edge between every pair of distinct positions.
for tail in graph.vertices:
    for head in graph.vertices:
        if tail != head:
            edge = graph.add_edge(tail, head)
            x_tail = tail.variables[0]
            x_head = head.variables[0]
            edge.add_cost(cp.norm1(x_tail - x_head)) # Manhattan distance traveled by the driver

# Solve problem.
from time import time
tic = time()
prob = graph.solve_traveling_salesman()
print("Problem status:", prob.status)
print("Optimal value:", prob.value)
print(time() - tic)

# Helper function that draws an L1 arrow between two points.
def l1_arrow(tail, head, color):
    arrowstyle = "->, head_width=3, head_length=8"
    options = dict(color=color, zorder=2, arrowstyle=arrowstyle)
    if not np.isclose(tail[0], head[0]):
        arrow = patches.FancyArrowPatch(tail, (head[0], tail[1]), **options)
        plt.gca().add_patch(arrow)
    if not np.isclose(tail[1], head[1]):
        arrow = patches.FancyArrowPatch((head[0], tail[1]), head, **options)
        plt.gca().add_patch(arrow)

# Plot solution,
plt.figure()
plt.grid()

# Path of the taxi driver.
for edge in graph.edges:
    if np.isclose(edge.binary_variable.value, 1):
        tail = edge.tail.variables[0].value
        head = edge.head.variables[0].value
        l1_arrow(tail, head, 'blue')

# Paths of the guests.
for vertex, position in zip(graph.vertices, guest_positions):
    xv = vertex.variables[0].value
    l1_arrow(position, xv, 'black')
    plt.scatter(*xv, c='green', marker='x', zorder=3)
    plt.scatter(*position, fc='white', ec='black', zorder=3)
    
# Party position.
plt.scatter(*party_position, marker='*', fc='yellow', ec='black', zorder=3, s=200, label="party")

# Adds empty plots for clean legend.
nans = np.full((2, 2), np.nan)
plt.scatter(*nans[0], fc='white', ec='black', label='guest initial')
plt.scatter(*nans[0], c='green', marker='x', label='guest optimal')
plt.plot(*nans, c='black', label='guest motion')
plt.plot(*nans, c='blue', label='host motion')

# Additional settings.
plt.xticks(range(l[0] - 1, u[0] + 2))
plt.yticks(range(l[1] - 1, u[1] + 2))
plt.legend()
plt.show()