import cvxpy as cp
import numpy as np
from gcspy import GraphOfConvexPrograms

# initialize empty graph
graph = GraphOfConvexPrograms()

# facility 1
f1 = graph.add_vertex("f1")
xf1 = f1.add_variable(2)
cf1 = np.array([0, 2.5])
f1.add_constraint(cp.norm_inf(xf1 - cf1) <= .5)

# facility 2
f2 = graph.add_vertex("f2")
xf2 = f2.add_variable(2)
cf2 = np.array([0, -1])
Df2 = np.diag([2, .5])
f2.add_constraint(cp.norm_inf(Df2 @ (xf2 - cf2)) <= 1)

# user 1
u1 = graph.add_vertex("u1")
xu1 = u1.add_variable(2)
cu1 = np.array([3, .5])
u1.add_constraint(cp.norm2(xu1 - cu1) <= .5)

# user 2
u2 = graph.add_vertex("u2")
xu2 = u2.add_variable(2)
cu2 = np.array([3, -1])
u2.add_constraint(cp.norm2(xu2 - cu2) <= .5)

# user 3
u3 = graph.add_vertex("u3")
xu3 = u3.add_variable(2)
cu3 = np.array([3, -2.5])
u3.add_constraint(cp.norm2(xu3 - cu3) <= .5)

# add edges
facilities = [f1, f2]
users = [u1, u2, u3]
for facility in facilities:
    for user in users:
        edge = graph.add_edge(facility, user)
        x_facility = facility.variables[0]
        x_user = user.variables[0]
        edge.add_cost(cp.norm2(x_user - x_facility))

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