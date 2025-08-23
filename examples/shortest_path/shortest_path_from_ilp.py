from shortest_path import graph, s, t

# Solve problem using built-in method.
graph.solve_shortest_path(s, t)
print("Problem status:", graph.status)
print("Optimal value:", graph.value)

# If the method solve_shortest_path was not implemented, we could still solve
# the problem by passing the constraints of its integer linear programming (ILP)
# formulation. Below we write these constraints explicitly. Note that the lower
# bound of 0 and upper bound of 1 on the binary variables are automatically
# enforced and we do not have to include them in our formulation.

# Binary variables.
yv = graph.vertex_binaries()
ye = graph.edge_binaries()

# Vertex constraints.
ilp_constraints = []
for i, v in enumerate(graph.vertices):
    is_source = 1 if v == s else 0
    is_target = 1 if v == t else 0
    inc = graph.incoming_edge_indices(v)
    out = graph.outgoing_edge_indices(v)
    ilp_constraints += [
        yv[i] == sum(ye[inc]) + is_source,
        yv[i] == sum(ye[out]) + is_target]

# Solve probelm from ILP constraints. Check that optimal value is equal to the
# one above.
graph.solve_from_ilp(ilp_constraints)
print("Problem status from ILP:", graph.status)
print("Optimal value from ILP:", graph.value)