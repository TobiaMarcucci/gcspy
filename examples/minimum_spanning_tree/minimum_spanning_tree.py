import cvxpy as cp
import numpy as np
from itertools import product
from gcsopt import GraphOfConvexSets

# Initialize empty graph.
directed = True # Both directed and undirected work.
graph = GraphOfConvexSets(directed=directed)

# Create vertices on a grid.
grid_sides = (3, 3)
grid_points = [(i, j) for i in range(grid_sides[0]) for j in range(grid_sides[1])]
radius = .25
for i, j in grid_points:
    v = graph.add_vertex((i, j))
    x = v.add_variable(2)
    center = np.array([i, j])
    v.add_constraint(cp.norm2(x - center) <= radius)

# Add edges between neighboring vertices.
for i, j in grid_points:
    for k, l in grid_points:
        distance = abs(k - i) + abs(l - j)
        if distance > 0 and distance <= 1:
            tail = graph.get_vertex((i, j))
            head = graph.get_vertex((k, l))
            if directed or not graph.has_edge((tail.name, head.name)):
                edge = graph.add_edge(tail, head)

                # Edge cost is Euclidean distance.
                x_tail = tail.variables[0]
                x_head = head.variables[0]
                edge.add_cost(cp.norm2(x_head - x_tail))

# Root of the spanning tree if directed.
root = graph.vertices[0]

# Run following code only if this file is executed directly, and not when it is
# imported by other files.
if __name__ == "__main__":

    # Solve minimum spanning tree problem using exponential-size formulation.
    prob = graph.solve_minimum_spanning_tree(root) # root ignored if undirected.

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
