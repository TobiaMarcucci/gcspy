import cvxpy as cp
import numpy as np
from gcsopt import GraphOfConvexSets

# Initialize empty graph.
directed = True # Both directed and undirected work.
graph = GraphOfConvexSets(directed)

# Vertex 0.
v0 = graph.add_vertex(0)
x0 = v0.add_variable(2)
c0 = np.array([-3, 0]) # center of the set
v0.add_constraint(cp.norm_inf(x0 - c0) <= 1)

# Vertex 1.
v1 = graph.add_vertex(1)
x1 = v1.add_variable(2)
c1 = np.array([0, 2.5]) # center of the set
D1 = np.diag([.25, 2]) # scaling matrix
v1.add_constraint(cp.norm_inf(D1 @ (x1 - c1)) <= 1)

# Vertex 2.
v2 = graph.add_vertex(2)
x2 = v2.add_variable(2)
c2 = np.array([3, 0]) # center of the set
v2.add_constraint(cp.norm2(x2 - c2) <= 1)
v2.add_constraint(x2 >= c2) # keep only top right part of the set

# Vertex 3.
v3 = graph.add_vertex(3)
x3 = v3.add_variable(2)
c3 = np.array([0, -2.5]) # center of the set
D3 = np.diag([1, 2]) # scaling matrix
v3.add_constraint(cp.norm2(D3 @ (x3 - c3)) <= 1)

# Vertex 4.
v4 = graph.add_vertex(4)
x4 = v4.add_variable(2)
c4 = np.array([.3, .3]) # center of the set
v4.add_constraint(cp.norm1(x4 - c4) <= 1)

# Add an edge between every pair of distinct vertices.
for i, tail in enumerate(graph.vertices):
    heads = graph.vertices[i + 1:]
    if directed:
        heads += graph.vertices[:i]
    for head in heads:
        if tail != head:
            if directed or tail.name < head.name:
                edge = graph.add_edge(tail, head)

                # Edge cost is Euclidean distance.
                x_tail = tail.variables[0]
                x_head = head.variables[0]
                edge.add_cost(cp.norm2(x_head - x_tail))

# Run following code only if this file is executed directly, and not when it is
# imported by other files.
if __name__ == "__main__":

    # Solve traveling salesman problem using Dantzig–Fulkerson–Johnson
    # formulation. All the subtour elimination constraints are included.
    graph.solve_traveling_salesman(subtour_elimination=True)
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
