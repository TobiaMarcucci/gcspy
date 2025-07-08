import cvxpy as cp
import numpy as np
from gcspy import GraphOfConvexPrograms

# initialize empty graph
graph = GraphOfConvexPrograms()

# vertex 1
v1 = graph.add_vertex("v1")
x1 = v1.add_variable(2)
c1 = np.array([-3, 0]) # center of the set
v1.add_constraint(cp.norm_inf(x1 - c1) <= 1)

# vertex 2
v2 = graph.add_vertex("v2")
x2 = v2.add_variable(2)
c2 = np.array([0, 2.5]) # center of the set
D2 = np.diag([.25, 2]) # scaling matrix
v2.add_constraint(cp.norm_inf(D2 @ (x2 - c2)) <= 1)

# vertex 3
v3 = graph.add_vertex("v3")
x3 = v3.add_variable(2)
c3 = np.array([3, 0]) # center of the set
v3.add_constraint(cp.norm2(x3 - c3) <= 1)
v3.add_constraint(x3 >= c3) # keep only top right part of the set

# vertex 4
v4 = graph.add_vertex("v4")
x4 = v4.add_variable(2)
c4 = np.array([0, -2.5]) # center of the set
D4 = np.diag([1, 2]) # scaling matrix
v4.add_constraint(cp.norm2(D4 @ (x4 - c4)) <= 1)

# vertex 5
v5 = graph.add_vertex("v5")
x5 = v5.add_variable(2)
c5 = np.array([.3, .3]) # center of the set
v5.add_constraint(cp.norm1(x5 - c5) <= 1)

# add an edges vetween every pair of distinct vertices
for tail in graph.vertices:
    for head in graph.vertices:
        if tail != head:
            edge = graph.add_edge(tail, head)

            # edge cost is Euclidean distance
            x_tail = tail.variables[0]
            x_head = head.variables[0]
            edge.add_cost(cp.norm2(x_head - x_tail))

# run followin code only if this file is executed directly, and not when it is
# imported by other files
if __name__ == '__main__':

    # solve traveling salesman problem using Dantzig–Fulkerson–Johnson
    # formulation, all the subtour elimination constraints are included
    prob = graph.solve_traveling_salesman(subtour_elimination=True)
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