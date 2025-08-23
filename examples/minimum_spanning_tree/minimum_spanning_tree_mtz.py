from minimum_spanning_tree import graph, root

# solve MSTP using subtour-elimination formulation
graph.solve_minimum_spanning_tree(root)
print("Problem status:", graph.status)
print("Optimal value:", graph.value)

# We can solve the MSTP using a strategy inspired by the Miller-Tucker-Zemlin
# formulation.

# for each vertex we add a continuous variable that represents how many vertices
# are visited by the tree before this one
for vertex in graph.vertices:
    count = vertex.add_variable(1)
    if vertex == root:
        vertex.add_constraint(count == 1)
    else:
        vertex.add_constraint(count >= 1)
    vertex.add_constraint(count <= graph.num_vertices())

# every time we traverse an edge the counter must increase the counter by one
for edge in graph.edges:
      if edge.head != root:
            count_tail = edge.tail.variables[1]
            count_head = edge.head.variables[1]
            edge.add_constraint(count_head == count_tail + 1)

# now we can solve the MSTP withouth the subtour elimination constraints
graph.solve_minimum_spanning_tree(root, subtour_elimination=False)
print("Problem status MTZ:", graph.status)
print("Optimal value MTZ:", graph.value)