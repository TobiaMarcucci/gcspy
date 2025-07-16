from facility_location import graph

# solve FLP
prob = graph.solve_facility_location()
print("Problem status:", prob.status)
print("Optimal value:", prob.value)

# If the method solve_facility_location was not implemented, we could still
# solve the FLP by passing the constraints of its integer linear program (ILP)
# formulation. Below we write the constraints explicitly. Note that the lower
# bound of 0 and upper bound of 1 on the binary variables are automatically
# enforced and we do not have to include them in our formulation.

# binary variables
yv = graph.vertex_binaries()
ye = graph.edge_binaries()

# vertex constraints
ilp_constraints = []
for i, v in enumerate(graph.vertices):
    inc = graph.incoming_edge_indices(v)
    if len(inc) > 0:
        ilp_constraints += [yv[i] == 1, sum(ye[inc]) == 1]
        
# edge constraints
for k, edge in enumerate(graph.edges):
    i = graph.vertex_index(edge.tail)
    ilp_constraints.append(yv[i] >= ye[k])

# solve FLP from constraints of the ILP formulation and check that optimal value
# is equal to the one above
prob = graph.solve_from_ilp(ilp_constraints)
print("Problem status from ILP:", prob.status)
print("Optimal value from ILP:", prob.value)