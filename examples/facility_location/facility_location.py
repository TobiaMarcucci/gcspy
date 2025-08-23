import cvxpy as cp
import numpy as np
from gcsopt import GraphOfConvexSets

# initialize empty graph
graph = GraphOfConvexSets()

# facility 1
f1 = graph.add_vertex("f1")
x = f1.add_variable(2)
c = np.array([0, 2.5])
f1.add_constraint(cp.norm_inf(x - c) <= .5)

# facility 2
f2 = graph.add_vertex("f2")
x = f2.add_variable(2)
c = np.array([0, -1])
D = np.diag([2, .5])
f2.add_constraint(cp.norm_inf(D @ (x - c)) <= 1)

# client 1
c1 = graph.add_vertex("c1")
x = c1.add_variable(2)
c = np.array([3, .5])
c1.add_constraint(cp.norm2(x - c) <= .5)

# client 2
c2 = graph.add_vertex("c2")
x = c2.add_variable(2)
c = np.array([3, -1])
c2.add_constraint(cp.norm2(x - c) <= .5)

# client 3
c3 = graph.add_vertex("c3")
x = c3.add_variable(2)
c = np.array([3, -2.5])
c3.add_constraint(cp.norm2(x - c) <= .5)

# add edges
facilities = [f1, f2]
clients = [c1, c2, c3]
for facility in facilities:
    for client in clients:
        edge = graph.add_edge(facility, client)
        x_facility = facility.variables[0]
        x_client = client.variables[0]
        edge.add_cost(cp.norm2(x_client - x_facility))

# run followin code only if this file is executed directly, and not when it is
# imported by other files
if __name__ == "__main__":

    # solve problem
    prob = graph.solve_facility_location()
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
