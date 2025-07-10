from shortest_path import graph, s, t

# solve SPP
prob = graph.solve_shortest_path(s, t)
print("Problem status:", prob.status)
print("Optimal value:", prob.value)

# If the method solve_shortest_path was not implemented, we could still solve
# the SPP by passing the constraints of its integer linear program (ILP)
# formulation. Below we write the constraints explicitly. Note that the lower
# bound of 0 and upper bound of 1 on the binary variables are automatically
# enforced and we do not have to include them in our formulation.

# binary variables
yv = graph.vertex_binaries()
ye = graph.edge_binaries()

# vertex constraints
s_index = graph.vertex_index(s)
t_index = graph.vertex_index(t)
ilp_constraints = []
for i, v in enumerate(graph.vertices):
    is_source = 1 if v == s else 0
    is_target = 1 if v == t else 0
    inc = graph.incoming_edge_indices(v)
    out = graph.outgoing_edge_indices(v)
    ilp_constraints += [
        yv[i] == sum(ye[inc]) + is_source,
        sum(ye[inc]) + is_source == sum(ye[out]) + is_target]

# solve SPP from constraints of the ILP formulation and check that optimal value
# is equal to the one above
prob = graph.solve_from_ilp(ilp_constraints)
print("Problem status from ILP:", prob.status)
print("Optimal value from ILP:", prob.value)