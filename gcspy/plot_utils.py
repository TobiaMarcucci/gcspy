import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import graphviz as gv

def discretize_vertex_2d(vertex, n=50):
    variable = vertex.variables[0]
    value = variable.value
    cost = cp.Parameter(2)
    prob = cp.Problem(cp.Maximize(cost @ variable), vertex.constraints)
    vertices = np.zeros((n, 2))
    for i, angle in enumerate(np.linspace(0, 2 * np.pi, n)):
        cost.value = np.array([np.cos(angle), np.sin(angle)])            
        prob.solve(warm_start=True)
        vertices[i] = variable.value
    variable.value = value
    return vertices


def plot_vertex_2d(vertex, n=50, tol=1e-4, **kwargs):
    vertices = discretize_vertex_2d(vertex, n)
    options = {'fc': 'mintcream', 'ec': 'black'}
    options.update(kwargs)
    vertex_min = np.min(vertices, axis=0)
    vertex_max = np.max(vertices, axis=0)
    vertex_dist = np.linalg.norm(vertex_max - vertex_min)
    if vertex_dist <= tol:
        plt.scatter(*vertices[0], fc='k', ec='k')
    else:
        plt.fill(*vertices.T, **options, zorder=0)
    value = vertex.variables[0].value
    

def plot_edge_2d(edge, endpoints=None, **kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    for variables in [edge.tail.variables, edge.head.variables]:
        if len(variables) != 1 or variables[0].size != 2:
            raise ValueError("Can only plot 2D sets.")
    arrowstyle = "->, head_width=3, head_length=8"
    options = dict(zorder=2, arrowstyle=arrowstyle)
    options.update(kwargs)
    if endpoints is None:
        endpoints = closest_points(edge.tail, edge.head)
    arrow = patches.FancyArrowPatch(*endpoints, **options)
    plt.gca().add_patch(arrow)


def closest_points(vertex1, vertex2):
    variable1 = vertex1.variables[0]
    variable2 = vertex2.variables[0]
    value1 = variable1.value
    value2 = variable2.value
    cost = cp.sum_squares(variable2 - variable1)
    constraints = vertex1.constraints + vertex2.constraints
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    points = [variable1.value, variable2.value]
    variable1.value = value1
    variable2.value = value2
    return points


def plot_gcs_2d(gcs, n=50):
    for vertex in gcs.vertices:
        if len(vertex.variables) != 1:
            raise ValueError("Can only plot sets with one variable.")
        if vertex.variables[0].size != 2:
            raise ValueError("Can only plot 2D sets.")
        plot_vertex_2d(vertex, n)
    for edge in gcs.edges:
        plot_edge_2d(edge, color='grey')


def plot_subgraph_2d(gcs, tol=1e-4):
    for vertex in gcs.vertices:
        if vertex.y.value > tol:
            variable = vertex.variables[0]
            plt.scatter(*variable.value, fc='w', ec='k', zorder=3)
    for edge in gcs.edges:
        if edge.y.value > tol:
            tail = edge.tail.variables[0].value
            head = edge.head.variables[0].value
            endpoints = (tail, head)
            plot_edge_2d(edge, endpoints, color='blue')


def graphviz_gcs(gcs, vertex_labels=None, edge_labels=None):
    if vertex_labels is None:
        vertex_labels = [vertex.name for vertex in gcs.vertices]
    if edge_labels is None:
        edge_labels = [''] * gcs.num_edges()
    digraph = gv.Digraph()
    for label in vertex_labels:
        digraph.node(str(label))
    for edge, label in zip(gcs.edges, edge_labels):
        tail = vertex_labels[gcs.vertices.index(edge.tail)]
        head = vertex_labels[gcs.vertices.index(edge.head)]
        digraph.edge(str(tail), str(head), str(label))
    return digraph
