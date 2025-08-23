import cvxpy as cp
import numpy as np
from gcsopt import GraphOfConvexSets

# Initialize empty graph.
graph = GraphOfConvexSets()

# Define source vertex.
s = graph.add_vertex("s")
xs = s.add_variable(2)
cs = np.array([1, 0]) # center of the set
Ds = np.diag([1, 1/2]) # scaling matrix
s.add_constraint(cp.norm2(Ds @ (xs - cs)) <= 1)

# Define target vertex.
t = graph.add_vertex("t")
xt = t.add_variable(2)
ct = np.array([10, 0]) # center of the set
Dt = np.diag([1/2, 1]) # scaling matrix
t.add_constraint(cp.norm2(Dt @ (xt - ct)) <= 1)
t.add_constraint(xt[0] <= ct[0]) # cut right half of the set

# Define vertex 1.
v1 = graph.add_vertex("v1")
x1 = v1.add_variable(2)
c1 = np.array([4, 2]) # center of the set
v1.add_constraint(cp.norm_inf(x1 - c1) <= 1)

# Define vertex 2.
v2 = graph.add_vertex("v2")
x2 = v2.add_variable(2)
c2 = np.array([5.5, -2]) # center of the set
v2.add_constraint(cp.norm1(x2 - c2) <= 1.2)
v2.add_constraint(cp.norm2(x2 - c2) <= 1)

# Define vertex 3.
v3 = graph.add_vertex("v3")
x3 = v3.add_variable(2)
c3 = np.array([7, 2]) # center of the set
v3.add_constraint(cp.norm2(x3 - c3) <= 1)

# Add some edges.
graph.add_edge(s, v1)
graph.add_edge(s, v2)
graph.add_edge(v1, v2)
graph.add_edge(v1, v3)
graph.add_edge(v2, v3)
graph.add_edge(v2, t)
graph.add_edge(v3, t)

# Add same cost to all edges.
for edge in graph.edges:
    x_tail = edge.tail.variables[0]
    x_head = edge.head.variables[0]
    edge.add_cost(cp.norm2(x_head - x_tail))

    # Add constraint that vertical coordinate of continuous variables can only
    # increase along the path.
    edge.add_constraint(x_head[1] >= x_tail[1])

# Following line ensures that rest of the code is run only if this file is
# executed directly, and it is not run if this file is imported by other files.
if __name__ == "__main__":

    # Solve shortest path problem.
    graph.solve_shortest_path(s, t)
    print("Problem status:", graph.status)
    print("Optimal value:", graph.value)

    # Show graph using graphviz (requires graphviz).
    dot = graph.graphviz()
    dot.view()

    # Plot optimal solution (requires matplotlib).
    import matplotlib.pyplot as plt
    plt.figure()
    plt.axis("equal")
    graph.plot_2d()
    graph.plot_2d_solution()
    plt.show()
