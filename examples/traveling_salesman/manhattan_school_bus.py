import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gcsopt import GraphOfConvexSets

# Problem data.
np.random.seed(0)
n_kids = 10
school_position = np.array([45, 7])
kid_positions = np.random.randint([38, 0], [53, 15], (n_kids, 2))

# Bounding box for all positions.
positions = np.vstack((school_position, kid_positions))
l = np.min(positions, axis=0)
u = np.max(positions, axis=0)

# Initialize empty graph.
graph = GraphOfConvexSets(directed=False)

# Vertex for every kid.
for i, position in enumerate(kid_positions):
    kid = graph.add_vertex(f"kid_{i}")
    x = kid.add_variable(2)
    d = kid.add_variable(1)[0] # L1 distance traveled by kid i.
    kid.add_constraints([
        x >= l,
        x <= u,
        d >= cp.norm1(x - position),
        d <= cp.norm1(school_position - position)])
    kid.add_cost(d) 

# Vertex for the school location.
school = graph.add_vertex("school")
x = school.add_variable(2)
school.add_constraint(x == school_position)

# Edge between every pair of distinct positions.
for i, tail in enumerate(graph.vertices):
    for head in graph.vertices[i + 1:]:
        edge = graph.add_edge(tail, head)
        x_tail = tail.variables[0]
        x_head = head.variables[0]
        edge.add_cost(cp.norm1(x_tail - x_head)) # L1 distance traveled by bus.

# Solve problem using gurobipy if possible (uses lazy constraints and is much
# faster). Otherwise use exponential formulation and default cvxpy solver.
import importlib.util
if importlib.util.find_spec("gurobipy"):
    prob = graph.solve_traveling_salesman_gurobipy()
else:
    prob = graph.solve_traveling_salesman()

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

# Bus path.
for edge in graph.edges:
    if np.isclose(edge.binary_variable.value, 1):
        tail = edge.tail.variables[0].value
        head = edge.head.variables[0].value
        l1_arrow(tail, head, "blue")

# Kid paths.
for vertex, position in zip(graph.vertices, kid_positions):
    xv = vertex.variables[0].value
    l1_arrow(position, xv, "black")
    plt.scatter(*xv, c="green", marker="x", zorder=3)
    plt.scatter(*position, fc="white", ec="black", zorder=3)
    
# School position.
plt.scatter(*school_position, marker="*", fc="yellow", ec="black", zorder=3,
            s=200, label="school")

# Adds empty plots for clean legend.
nans = np.full((2, 2), np.nan)
plt.scatter(*nans[0], fc="white", ec="black", label="kid's home")
plt.scatter(*nans[0], c="green", marker="x", label="kid's pickup")
plt.plot(*nans, c="black", label="kid motion")
plt.plot(*nans, c="blue", label="bus motion")

# Additional settings.
plt.xticks(range(l[0] - 1, u[0] + 2))
plt.yticks(range(l[1] - 1, u[1] + 2))
plt.legend()
plt.show()
