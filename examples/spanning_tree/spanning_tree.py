import cvxpy as cp
import numpy as np
from itertools import product
from gcspy import GraphOfConvexSets

# initialize empty graph
graph = GraphOfConvexSets()

# create vertices on a grid
grid_sides = (4, 3)
grid_points = [(i, j) for i in range(grid_sides[0]) for j in range(grid_sides[1])]
radius = .25
for i, j in grid_points:
    v = graph.add_vertex((i, j))
    x = v.add_variable(2)
    center = np.array([i, j])
    v.add_constraint(cp.norm2(x - center) <= radius)

# root of the spanning tree
root = graph.vertices[0]

# add edges between neighboring vertices
for i, j in grid_points:
    for k, l in grid_points:
        distance = abs(k - i) + abs(l - j)
        if distance > 0 and distance <= 1:
            tail = graph.get_vertex((i, j))
            head = graph.get_vertex((k, l))
            edge = graph.add_edge(tail, head)

            # edge cost is Euclidean distance
            x_tail = tail.variables[0]
            x_head = head.variables[0]
            edge.add_cost(cp.norm2(x_head - x_tail))

# run followin code only if this file is executed directly, and not when it is
# imported by other files
if __name__ == "__main__":

    # solve spanning tree problem using Dantzig–Fulkerson–Johnson formulation
    # all the subtour elimination constraints are included
    prob = graph.solve_spanning_tree(root)
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