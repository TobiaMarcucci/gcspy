import cvxpy as cp
import numpy as np
from gcspy import GraphOfConvexSets

# initialize empty graph
graph = GraphOfConvexSets()

# source vertex
s = graph.add_vertex("s")
xs = s.add_variable(2)
cs = np.array([1, 0]) # center of the set
Ds = np.diag([1, 1/2]) # scaling matrix
s.add_constraint(cp.norm2(Ds @ (xs - cs)) <= 1)

# target vertex
t = graph.add_vertex("t")
xt = t.add_variable(2)
ct = np.array([10, 0]) # center of the set
Dt = np.diag([1/2, 1]) # scaling matrix
t.add_constraint(cp.norm2(Dt @ (xt - ct)) <= 1)
t.add_constraint(xt[0] <= ct[0]) # cut right half of the set

# vertex 1
v1 = graph.add_vertex("v1")
x1 = v1.add_variable(2)
c1 = np.array([4, 2]) # center of the set
v1.add_constraint(cp.norm_inf(x1 - c1) <= 1)

# vertex 2
v2 = graph.add_vertex("v2")
x2 = v2.add_variable(2)
c2 = np.array([5.5, -2]) # center of the set
v2.add_constraint(cp.norm1(x2 - c2) <= 1.2)
v2.add_constraint(cp.norm2(x2 - c2) <= 1)

# vertex 3
v3 = graph.add_vertex("v3")
x3 = v3.add_variable(2)
c3 = np.array([7, 2]) # center of the set
v3.add_constraint(cp.norm2(x3 - c3) <= 1)

# add some edges
graph.add_edge(s, v1)
graph.add_edge(s, v2)
graph.add_edge(v1, v2)
graph.add_edge(v1, v3)
graph.add_edge(v2, v3)
graph.add_edge(v2, t)
graph.add_edge(v3, t)

# add the same cost to all the edges
for edge in graph.edges:
    x_tail = edge.tail.variables[0]
    x_head = edge.head.variables[0]
    edge.add_cost(cp.norm2(x_head - x_tail))

    # add constraint that y variables can only increase along the path
    edge.add_constraint(x_head[1] >= x_tail[1])

# run followin code only if this file is executed directly, and not when it is
# imported by other files
if __name__ == "__main__":

    # solve shortest path problem
    prob = graph.solve_shortest_path(s, t)
    print("Problem status:", prob.status)
    print("Optimal value:", prob.value)

    # show graph using graphviz (requires graphviz)
    dot = graph.graphviz()
    dot.view()

    # plot optimal solution (requires matplotlib)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.axis("equal")
    graph.plot_2d()
    graph.plot_2d_solution()
    plt.show()