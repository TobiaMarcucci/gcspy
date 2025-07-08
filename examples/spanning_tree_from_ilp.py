from itertools import combinations
from spanning_tree import graph, root

# solve MSTP using Dantzig–Fulkerson–Johnson formulation
prob = graph.solve_spanning_tree(root)
print("Problem status:", prob.status)
print("Optimal value:", prob.value)

# If the method solve_spanning_tree was not implemented, we could still
# solve the MSTP by passing the constraints of its integer linear program (ILP)
# formulation. Below, we write the constraints for the ILP formulation by DFJ
# explicitly. Note that the lower bound of 0 and upper bound of 1 on the binary
# variables are automatically enforced and we do not have to include them in our
# formulation.

# binary variables
yv = graph.vertex_binaries()
ye = graph.edge_binaries()

# vertex constraints
ilp_constraints = []
root_index = graph.vertex_index(root)
for i, vertex in enumerate(graph.vertices):
    inc = graph.incoming_edge_indices(vertex)
    inc_flow = 0 if i == root_index else 1
    ilp_constraints += [yv[i] == 1, sum(ye[inc]) == inc_flow]

# subtour elimnation constraints
for subtour_size in range(2, graph.num_vertices()):
    for vertices in combinations(graph.vertices[1:], subtour_size):
        inc = graph.incoming_edge_indices(vertices)
        ilp_constraints.append(sum(ye[inc]) >= 1)

# solve MSTP from constraints of the ILP formulation and check that optimal
# value is equal to the one above
prob = graph.solve_from_ilp(ilp_constraints)
print("Problem status from ILP:", prob.status)
print("Optimal value from ILP:", prob.value)