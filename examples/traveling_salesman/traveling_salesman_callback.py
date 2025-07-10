import numpy as np
from traveling_salesman import graph

# solve TSP using Dantzig–Fulkerson–Johnson formulation, all the subtour
# elimination constraints are included
prob = graph.solve_traveling_salesman(subtour_elimination=True)
print("Problem status:", prob.status)
print("Optimal value:", prob.value)

# Instead of including all the subtour elimination constraints in the initial
# formulation, we can add them as needed. To this end we provide as a
# callback to the solve method the subtour_elimination function defined below. 

# function that computes a cycle with nonzero flow starting from a given vertex
def expand_cycle(graph, vertex, ye):
    cycle = [vertex]
    while True:
        out = graph.outgoing_edge_indices(cycle[-1])
        next_edge_idx = out[np.argmax(ye[out])]
        next_edge = graph.edges[next_edge_idx]
        next_vertex = next_edge.head
        if next_vertex == cycle[0]:
            break
        cycle.append(next_vertex)
    return cycle

# function that computes a subtour of minimum length
def shortest_subtour(graph, ye):
    vertices_left = graph.vertices
    subtour = graph.vertices
    while len(vertices_left) > 0:
        new_subtour = expand_cycle(graph, vertices_left[0], ye)
        if len(new_subtour) < len(subtour):
            subtour = new_subtour
        vertices_left = [vertex for vertex in vertices_left if vertex not in new_subtour]
    return subtour

# function that computes a linear constraint that eliminates a subtour
# inputs are yv and ye as required for a callback function
def subtour_elimination(yv, ye):
    ye_value = np.array([y.value for y in ye])
    subtour = shortest_subtour(graph, ye_value)
    if len(subtour) == graph.num_vertices():
        return []
    else:
        print(f"Eliminated subtour with {len(subtour)} vertices.")
        out = graph.outgoing_edge_indices(subtour)
        return [sum(ye[out]) >= 1]
    
# solve traveling salesman using callback
prob = graph.solve_traveling_salesman(subtour_elimination=False, callback=subtour_elimination)
print("Problem status with callback:", prob.status)
print("Optimal value with callback:", prob.value)