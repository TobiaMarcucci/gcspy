import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from gcsopt import GraphOfConvexSets

# problem data
num_islands = 20 # number of islands
max_radius = .1 # maximum radius of each island
start = np.zeros(2) # start position of the helicopter
goal = np.ones(2) # goal position of the helicopter
speed = 1 # helicopter speed
max_battery = 1 # maximum battery level
discharge_rate = 5 # battery discharge rate when flying
charge_rate = 1 # battery charge rate when resting on an island
max_range = speed * max_battery / discharge_rate # maximum flying range

# generate random islands that do not intersect
np.random.seed(1) # tuned to make the islands look nice
centers = np.full((num_islands, 2), np.inf) # inf ensures no intersection with sampled islands
radii = np.zeros(num_islands)
sampled_islands = 0
while sampled_islands < num_islands:
    center = np.random.uniform(start, goal)
    radius = np.random.uniform(0, max_radius)

    # discard this island if it intersects with a previously sampled island
    if all(np.linalg.norm(center - centers, axis=1) > radius + radii):
        centers[sampled_islands] = center
        radii[sampled_islands] = radius
        sampled_islands += 1

# include start and goal in the island list (facilitates the graph construction)
centers = np.vstack([start, centers, goal])
radii = np.concatenate([[0], radii, [0]])

# initialize empty graph
graph = GraphOfConvexSets()

# add one vertex for every island (including start and goal)
for i, (center, radius) in enumerate(zip(centers, radii)):
    vertex = graph.add_vertex(i)
    q = vertex.add_variable(2) # helicopted landing position on the island
    z = vertex.add_variable(2) # batter level at landing and take off
    t = vertex.add_variable(1) # recharge time
    vertex.add_cost(t)
    vertex.add_constraints([
        cp.norm2(q - center) <= radius,
        z >= 0, z <= max_battery,
        t >= 0, t <= max_battery / charge_rate,
        z[1] == z[0] + charge_rate * t])
    
    # battery is fully charged at the beginning
    if i == 0:
        vertex.add_constraint(z[0] == max_battery)

# add edges between pairs of islands that are close enough
for i, (center_i, radius_i) in enumerate(zip(centers, radii)):
    vertex_i = graph.get_vertex(i)
    qi, zi = vertex_i.variables[:2]
    for j, (center_j, radius_j) in enumerate(zip(centers, radii)):
        if i != j:
            center_dist = np.linalg.norm(center_i - center_j)
            island_dist = center_dist - radius_i - radius_j
            if island_dist < max_range: # necessary condition for flight feasibility
                vertex_j = graph.get_vertex(j)
                qj, zj = vertex_j.variables[:2]
                edge = graph.add_edge(vertex_i, vertex_j)
                t = edge.add_variable(1) # flight time between islands i and j
                edge.add_cost(t)
                edge.add_constraints([
                    t >= cp.norm2(qi - qj) / speed,
                    zi[1] == zj[0] + discharge_rate * t])

# solve shortest path problem from start to goal points
source = graph.vertices[0]
target = graph.vertices[-1]
prob = graph.solve_shortest_path(source, target)
print("Problem status:", prob.status)
print("Optimal value:", prob.value)

# plot optimal flight trajectory
plt.figure()
graph.plot_2d_solution()

# plot ocean
l = start - max_radius
d = (goal - start) + 2 * max_radius
ocean = plt.Rectangle(l, *d, fc="azure")
plt.gca().add_patch(ocean)

# plot islands
for i in range(num_islands):
    island = plt.Circle(centers[i], radii[i], ec="k", fc="lightgreen")
    plt.gca().add_patch(island)

# misc plot settings
plt.gca().set_aspect("equal")
limits = np.array([l, l + d])
plt.xlim(limits[:, 0])
plt.ylim(limits[:, 1])
plt.show()

# reconstruct the battery level as a function of time
battery_levels = []
times = [0]
vertex = source
while vertex != target:
    z, t = vertex.variables[1:]
    battery_levels.extend(z.value)
    times.extend(times[-1] + t.value)
    for edge in graph.outgoing_edges(vertex):
        if np.isclose(edge.binary_variable.value, 1):
            t = edge.variables[0]
            times.extend(times[-1] + t.value)
            vertex = edge.head
            break
battery_levels.append(target.variables[1].value[0])

# plot battery level
plt.figure()
end_times = (times[0], times[-1])
plt.plot(end_times, (0, 0), "r--") # minimum value
plt.plot(end_times, (max_battery, max_battery), "g--") # maximum value
plt.plot(times, battery_levels)
plt.xlabel("Time")
plt.ylabel("Battery level")
plt.xlim(end_times)
plt.grid()
plt.show()
