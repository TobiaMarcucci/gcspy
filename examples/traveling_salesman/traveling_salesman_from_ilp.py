from itertools import combinations
from traveling_salesman import graph

# solve TSP using Dantzig–Fulkerson–Johnson formulation
prob = graph.solve_traveling_salesman()
print("Problem status:", prob.status)
print("Optimal value:", prob.value)

# If the method solve_traveling_salesman was not implemented, we could still
# solve the TSP by passing the constraints of its integer linear program (ILP)
# formulation. Below, we write the constraints for the ILP formulation by DFJ
# explicitly. Note that the lower bound of 0 and upper bound of 1 on the binary
# variables are automatically enforced and we do not have to include them in our
# formulation.

# binary variables
yv = graph.vertex_binaries()
ye = graph.edge_binaries()

# vertex constraints
ilp_constraints = []
for i, vertex in enumerate(graph.vertices):
    inc = graph.incoming_edge_indices(vertex)
    out = graph.outgoing_edge_indices(vertex)
    ilp_constraints += [yv[i] == 1, sum(ye[out]) == 1, sum(ye[inc]) == 1]

# subtour elimnation constraints
for subtour_size in range(2, graph.num_vertices() - 1):
    for vertices in combinations(graph.vertices, subtour_size):
        out = graph.outgoing_edge_indices(vertices)
        ilp_constraints.append(sum(ye[out]) >= 1)

# solve TSP from constraints of the ILP formulation and check that optimal value
# is equal to the one above
prob = graph.solve_from_ilp(ilp_constraints)
print("Problem status from ILP:", prob.status)
print("Optimal value from ILP:", prob.value)