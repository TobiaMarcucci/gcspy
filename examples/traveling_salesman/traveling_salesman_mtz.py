from traveling_salesman import graph

# solve TSP using Dantzig–Fulkerson–Johnson formulation
prob = graph.solve_traveling_salesman()
print("Problem status:", prob.status)
print("Optimal value:", prob.value)

# We can solve the TSP using a strategy inspired by the Miller-Tucker-Zemlin
# formulation.

# for each vertex we add a continuous variable that represents how many vertices
# are visited by the tour befors this one
for i, vertex in enumerate(graph.vertices):
    count = vertex.add_variable(1)
    if i == 0:
        vertex.add_constraint(count == 1)
    else:
        vertex.add_constraint(count >= 1)
    vertex.add_constraint(count <= graph.num_vertices())

# every time we traverse an edge the counter must increase the counter by one
for edge in graph.edges:
      if edge.head.name != "v0":
            count_tail = edge.tail.variables[1]
            count_head = edge.head.variables[1]
            edge.add_constraint(count_head == count_tail + 1)

# now we can solve the TSP withouth the subtour elimination constraints
prob = graph.solve_traveling_salesman(subtour_elimination=False)
print("Problem status MTZ:", prob.status)
print("Optimal value MTZ:", prob.value)