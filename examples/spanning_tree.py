import cvxpy as cp
import numpy as np
from gcspy import GraphOfConvexPrograms

# initialize empty graph
graph = GraphOfConvexPrograms()

# create vertices on a grid
grid_side = 3
radius = .25
for i in range(grid_side):
    for j in range(grid_side):
        v = graph.add_vertex((i, j))
        x = v.add_variable(2)
        center = np.array([i, j])
        v.add_constraint(cp.norm2(x - center) <= radius)

# add edges between neighboring vertices
for i in range(grid_side):
    for j in range(grid_side):
        for k in range(grid_side):
            for l in range(grid_side):
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
if __name__ == '__main__':

    # solve spanning tree problem using Dantzig–Fulkerson–Johnson formulation
    # all the subtour elimination constraints are included
    root = graph.vertices[0]
    prob = graph.solve_spanning_tree(root)
    print('Problem status:', prob.status)
    print('Optimal value:', prob.value)

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