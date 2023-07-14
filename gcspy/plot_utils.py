import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import graphviz as gv


def discretize_vertex_2d(vertex, n=30):
    if len(vertex.variables) != 1 or vertex.variables[0].size != 2:
        raise ValueError("Can only discretize 2D sets.")
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


def plot_vertex_2d(vertex, n=30, tol=1e-4, **kwargs):
    if len(vertex.variables) != 1 or vertex.variables[0].size != 2:
        raise ValueError("Can only plot 2D sets.")
    options = {'fc': 'lightcyan', 'ec': 'black'}
    options.update(kwargs)
    vertices = discretize_vertex_2d(vertex, n)
    vertex_min = np.min(vertices, axis=0)
    vertex_max = np.max(vertices, axis=0)
    vertex_dist = np.linalg.norm(vertex_max - vertex_min)
    if vertex_dist <= tol:
        plt.scatter(*vertices[0], fc='k', ec='k')
    else:
        plt.fill(*vertices.T, **options, zorder=0)
    value = vertex.variables[0].value
    if value is not None:
        plt.scatter(*value, fc='w', ec='k', zorder=3)


def plot_edge_2d(edge, **kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    for variables in [edge.tail.variables, edge.head.variables]:
        if len(variables) != 1 or variables[0].size != 2:
            raise ValueError("Can only plot 2D sets.")
    arrowstyle = '->, head_width=3, head_length=8'
    options = dict(color='k', zorder=2, arrowstyle=arrowstyle)
    options.update(kwargs)
    tail = edge.tail.variables[0].value
    head = edge.head.variables[0].value
    if tail is None:
        tail = edge.tail.get_feasible_point()[0]
    if head is None:
        head = edge.head.get_feasible_point()[0]
    arrow = patches.FancyArrowPatch(tail, head, **options)
    if edge.value is not None:
        center = (tail + head) / 2
        bbox = dict(facecolor='w', edgecolor='r', boxstyle='round')
        plt.text(*center, str(edge.value),
                 color='r', ha='center', va='center', bbox=bbox)
    plt.gca().add_patch(arrow)


def plot_gcs_2d(gcs):
    for vertex in gcs.vertices:
        plot_vertex_2d(vertex)
    for edge in gcs.edges:
        plot_edge_2d(edge)


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
