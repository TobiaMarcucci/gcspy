import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from gcsopt import GraphOfConvexSets

# Problem data.
np.random.seed(0)
n_kids = 15
school_position = np.array([45, 7])
kid_positions = np.random.randint([30, 1], [60, 12], (n_kids, 2))

# Bounding box for all positions.
positions = np.vstack((school_position, kid_positions))
l = np.min(positions, axis=0)
u = np.max(positions, axis=0)

# Initialize empty graph.
graph = GraphOfConvexSets(directed=False)

# Vertex for every kid.
for i, position in enumerate(kid_positions):
    kid = graph.add_vertex(i + 1)
    x = kid.add_variable(2)
    d = kid.add_variable(1)[0] # L1 distance traveled by kid i.
    kid.add_constraints([
        x >= l,
        x <= u,
        d >= cp.norm1(x - position),
        d <= cp.norm1(school_position - position)])
    kid.add_cost(d)

# Vertex for the school.
school = graph.add_vertex(0)
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
has_gurobi = importlib.util.find_spec("gurobipy")
if has_gurobi:
    from gcsopt.gurobipy.graph_problems.traveling_salesman import traveling_salesman
    plot_bounds = True
    parameters = {"OutputFlag": 0}
    traveling_salesman(graph, gurobi_parameters=parameters, save_bounds=plot_bounds)
    print("Solver time:", graph.solver_stats.solve_time)
else:
    graph.solve_traveling_salesman()

# Plot upper and lower bounds from gurobi.
if plot_bounds and has_gurobi:
    bounds = graph.solver_stats.callback_bounds
    bounds[1] = np.where(bounds[1] >= 0, bounds[1], np.nan)
    plt.figure(figsize=(8, 3))
    plt.plot(bounds[0], bounds[2], lw=2, ls='-', label="best upper bound")
    plt.plot(
        [bounds[0, 0], bounds[0, -1]],
        [bounds[2, -1], bounds[2, -1]],
        lw=2, ls='--', label="optimal value")
    plt.plot(bounds[0], bounds[1], lw=2, ls=':', label="best lower bound")
    plt.xlim([bounds[0, 0], bounds[0, -1]])
    plt.xlabel("solver time (s)")
    plt.ylabel("objective value")
    plt.grid()
    plt.legend()
    plt.xlim(xmin=0)
    plt.savefig("bus_bounds.pdf", bbox_inches="tight")
    plt.show()

# Helper function that draws an L1 path between two points.
def l1_path(tail, head, color, ls):
    options = dict(color=color, ls=ls, zorder=2)
    if not np.isclose(tail[0], head[0]):
        plt.plot([tail[0], head[0]], [tail[1], tail[1]], **options)
    if not np.isclose(tail[1], head[1]):
        plt.plot([head[0], head[0]], [tail[1], head[1]], **options)

# Plot solution.
plt.figure(figsize=(8, 3))
plt.grid()

# Bus path.
for edge in graph.edges:
    if np.isclose(edge.binary_variable.value, 1):
        tail = edge.tail.variables[0].value
        head = edge.head.variables[0].value
        l1_path(tail, head, "red", "--")

# Kid walks.
for vertex, position in zip(graph.vertices, kid_positions):
    xv = vertex.variables[0].value
    l1_path(position, xv, "blue", "-")
    plt.scatter(*xv, c="blue", marker="x", zorder=3)
    plt.scatter(*position, fc="white", ec="blue", zorder=3)
    
# School position.
plt.scatter(*school_position, marker="*", fc="yellow", ec="black", zorder=3,
            s=200, label="school")

# Empty plots for clean legend.
nans = np.full((2, 2), np.nan)
plt.scatter(*nans[0], fc="white", ec="blue", label="home")
plt.scatter(*nans[0], c="blue", marker="x", label="pick-up point")
plt.plot(*nans, c="blue", ls="-", label="kid's walk")
plt.plot(*nans, c="red", ls="--", label="bus tour")

# Additional plot settings.
plt.xticks(range(l[0] - 1, u[0] + 2))
plt.yticks(range(l[1] - 1, u[1] + 2))
plt.xlim([l[0] - 1, u[0] + 1])
plt.ylim([l[1] - 1, u[1] + 1])
plt.xlabel('street')
plt.ylabel('avenue')
plt.legend()
plt.savefig("bus_tour.pdf", bbox_inches="tight")
plt.show()