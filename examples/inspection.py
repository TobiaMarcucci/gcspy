import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gcsopt import GraphOfConvexSets

# all the rooms in the floor plant
# each room is described by a triplet (index, lower corner, upper corner)
rooms = [
    (0, [0, 5], [2, 8]), 
    (1, [0, 3], [4, 5]),
    (2, [0, 0], [2, 3]),
    (3, [2, 5], [4, 8]),
    (4, [2, 0], [4, 3]),
    (5, [4, 0], [6, 8]),
    (6, [6, 7], [8, 8]),
    (7, [6, 1], [8, 7]),
    (8, [6, 0], [8, 1]),
    (9, [8, 0], [10, 10]),
    (10, [10, 5], [12, 10]),
    (11, [10, 4], [12, 5]),
    (12, [10, 0], [12, 4]),
    (13, [12, 0], [16, 2]),
    (14, [12, 2], [14, 10]),
    (15, [14, 7], [16, 10]),
    (17, [0, 8], [4, 10]),
    (18, [4, 8], [8, 10]),
    (19, [14, 2], [16, 7]),
]

# all the doors in the floor plant
# each door is described by
# (first room index, second room index, door lower corner, door upper corner)
doors = [
    (0, 3, [2, 7], [2, 8]),
    (1, 2, [1, 3], [2, 3]),
    (1, 5, [4, 4], [4, 5]),
    (2, 4, [2, 2], [2, 3]),
    (3, 5, [4, 5], [4, 8]),
    (3, 17, [3, 8], [4, 8]),
    (4, 5, [4, 0], [4, 3]),
    (5, 6, [6, 7], [6, 8]),
    (5, 7, [6, 1], [6, 2]),
    (5, 8, [6, 0], [6, 1]),
    (6, 9, [8, 7], [8, 8]),
    (7, 9, [8, 3], [8, 4]),
    (8, 9, [8, 0], [8, 1]),
    (9, 10, [10, 7], [10, 8]),
    (9, 11, [10, 4], [10, 5]),
    (9, 12, [10, 3], [10, 4]),
    (9, 18, [8, 9], [8, 10]),
    (10, 14, [12, 5], [12, 6]),
    (11, 14, [12, 4], [12, 5]),
    (12, 13, [12, 0], [12, 2]),
    (14, 15, [14, 7], [14, 8]),
    (14, 19, [14, 6], [14, 7]),
]

# rooms that must be visited by the minimum-length trajectory
visit_rooms = [0, 2, 7, 10, 13, 15, 18]

# helper class that allows to construct a floor
class Floor(GraphOfConvexSets):

    def __init__(self, rooms, doors, name):
        super().__init__()
        self.rooms = rooms
        self.doors = doors
        self.name = name
        for room in rooms:
            self.add_room(*room)
        for door in doors:
            self.add_door(*door)
        
    def add_room(self, n, l, u):
        v = self.add_vertex(f"{self.name}_{n}")
        x1 = v.add_variable(2)
        x2 = v.add_variable(2)
        v.add_constraints([x1 >= l, x1 <= u])
        v.add_constraints([x2 >= l, x2 <= u])
        v.add_cost(cp.norm2(x2 - x1))
        return v
    
    def add_one_way_door(self, n, m, l, u):
        tail = self.get_vertex(f"{self.name}_{n}")
        head = self.get_vertex(f"{self.name}_{m}")
        e = self.add_edge(tail, head)
        e.add_constraint(tail.variables[1] == head.variables[0])
        e.add_constraint(tail.variables[1] >= l)
        e.add_constraint(tail.variables[1] <= u)
        return e
    
    def add_door(self, n, m, l, u):
        e1 = self.add_one_way_door(n, m, l, u)
        e2 = self.add_one_way_door(m, n, l, u)
        return e1, e2

# initialize empty graph
graph = GraphOfConvexSets()

# adds one copy of the floor plant for each room that we must visit
num_floors = len(visit_rooms)
for floor in range(num_floors):
    graph.add_disjoint_subgraph(Floor(rooms, doors, floor))

# connects copies of the floors on a given room
def connect_floors(floor1, floor2, room):
    tail = graph.get_vertex(f"{floor1}_{room}")
    head = graph.get_vertex(f"{floor2}_{room}")
    edge = graph.add_edge(tail, head)
    edge.add_constraint(tail.variables[1] == head.variables[0])

# connect top floor to ground floor at first visit room
first_room = visit_rooms[0]
first_floor = 0
last_floor = num_floors - 1
connect_floors(last_floor, first_floor, first_room)

# connect each floor to the floor above at the visit room
for floor in range(last_floor):
    for room in visit_rooms[1:]:
        connect_floors(floor, floor + 1, room)

# retrieve binary variables
yv = graph.vertex_binaries()
ye = graph.edge_binaries()

# constraints of the integer programming formulation
ilp_constraints = []
for i, vertex in enumerate(graph.vertices):
    inc_edges = graph.incoming_edge_indices(vertex)
    out_edges = graph.outgoing_edge_indices(vertex)
    ilp_constraints.append(yv[i] == sum(ye[inc_edges]))
    ilp_constraints.append(yv[i] == sum(ye[out_edges]))
    
# returns the edge binary variable that connects two floors through a given room
def get_binary_variable(floor1, floor2, room):
    tail_name = f"{floor1}_{room}"
    head_name = f"{floor2}_{room}"
    edge = graph.get_edge(tail_name, head_name)
    return ye[graph.edge_index(edge)]

# add constraints that force the trajectory to move between floors
ilp_constraints.append(get_binary_variable(last_floor, first_floor, first_room) == 1)
for room in visit_rooms[1:]:
    flow = sum(get_binary_variable(floor, floor + 1, room) for floor in range(last_floor))
    ilp_constraints.append(flow == 1)

# solve problem (this will take a very long time)
prob = graph.solve_from_ilp(ilp_constraints)
print('Problem status:', prob.status)
print('Optimal value:', prob.value)

# plot solution
plt.figure()
plt.axis("equal")

# helper function that plots one room
def plot_room(n, l, u):
    l = np.array(l)
    u = np.array(u)
    d = u - l
    fc = "mistyrose" if n in visit_rooms else "mintcream"
    rect = patches.Rectangle(l, *d, fc=fc, ec="k")
    plt.gca().add_patch(rect)
        
# helper function that plots one door
def plot_door(l, u):
    endpoints =  np.array([l, u]).T
    plt.plot(*endpoints, color="mintcream", solid_capstyle="butt")
    plt.plot(*endpoints, color="grey", linestyle=":")
        
# plot all rooms and doors
for room in rooms:
    plot_room(*room)
for door in doors:
    plot_door(door[2], door[3])

# plot optimal trajectory
for vertex in graph.vertices:
    if np.isclose(vertex.binary_variable.value, 1):
        x1, x2 = vertex.variables
        values = np.array([x1.value, x2.value]).T
        plt.plot(*values, c="b", linestyle="--")
    
plt.show()
