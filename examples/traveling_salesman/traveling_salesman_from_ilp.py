from itertools import combinations
from traveling_salesman import graph

# Solve problem using built-in method.
graph.solve_traveling_salesman()
print("Problem status:", graph.status)
print("Optimal value:", graph.value)

# If the method solve_traveling_salesman was not implemented, we could still solve
# the problem by passing the constraints of its integer linear programming (ILP)
# formulation. Below we write these constraints explicitly. Note that the lower
# bound of 0 and upper bound of 1 on the binary variables are automatically
# enforced and we do not have to include them in our formulation.

# Binary variables.
yv = graph.vertex_binaries()
ye = graph.edge_binaries()

# Vertex constraints.
ilp_constraints = []
for i, vertex in enumerate(graph.vertices):
    inc = graph.incoming_edge_indices(vertex)
    out = graph.outgoing_edge_indices(vertex)
    ilp_constraints += [yv[i] == 1, sum(ye[out]) == 1, sum(ye[inc]) == 1]

# Subtour elimnation constraints.
for subtour_size in range(2, graph.num_vertices() - 1):
    for vertices in combinations(graph.vertices, subtour_size):
        out = graph.outgoing_edge_indices(vertices)
        ilp_constraints.append(sum(ye[out]) >= 1)

# Solve probelm from ILP constraints. Check that optimal value is equal to the
# one above.
graph.solve_from_ilp(ilp_constraints)
print("Problem status from ILP:", graph.status)
print("Optimal value from ILP:", graph.value)