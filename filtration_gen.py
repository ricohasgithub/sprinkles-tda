
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import gudhi

# Hx, m x n: maps from faces to edges
# Hz, m x n: maps from vertices to edges

def is_zero(vector):
    return np.all(vector == 0)

class SampleQueue:

    def __init__(self, Hx, Hz_t):

        # Hx: m x n, Hz_t: n x m
        self.m, self.n = Hx.shape[0], Hz_t.shape[0]

        for c in range(self.n):
            if is_zero(Hx[:, c]):
                print("modify")
                # e_c c-th basis vector in R^n
                ec = np.eye(self.n)[c]
                Hx = np.vstack(Hx, ec)
        
        self.Hz_t = Hz_t
        self.Hx = Hx
        self.unsampled_2_simplices = list(range(0, self.m))

    def is_empty(self):
        return len(self.unsampled_2_simplices) == 0

    def sample(self):

        # Uniformly sample a random 2-simplex that hasn't already been sampled
        r_sigma = random.choice(self.unsampled_2_simplices)
        sigma = self.Hx[r_sigma]

        # All dependencies of sigma has already been sampled, so we can return it directly
        if is_zero(sigma):
            self.unsampled_2_simplices.remove(r_sigma)
            return (2, r_sigma)
        
        # Uniformly sample a random 1-simplex face of sigma
        tau_indices = list(np.where(sigma == 1)[0])
        r_tau = np.random.choice(tau_indices)
        tau = self.Hz_t[r_tau]

        # All dependencies of tau have already been sampled, so we can return it directly
        if is_zero(tau):
            # Set column in Hx corresponding to tau to 0
            self.Hx[:, r_tau] = 0
            return (1, r_tau)
        
        # Finally, return 1 unsampled 0-simplex of tau if needed
        x_indices = list(np.where(tau == 1)[0])
        r_x = np.random.choice(x_indices)

        # Set column in Hz_t corresponding to x to 0
        self.Hz_t[:, r_x] = 0
        return (0, r_x)

def TS3(Hx, Hz):

    # Idea: perform a topological sort of the 2-simplex
    # We represent the 2-simplex as a directed graph, whereby vertices have indegree 0
    # edges have indegree 2, and faces have indegree 4
    # The topological sort defines a boundary for which we can perform a filtration

    # Graph building
    graph = defaultdict(list)
    indegrees = defaultdict(int)
    # Iterate over all vertex to edge mappings
    for i in range(len(Hz)):
        # Degree of simplex (0 for vertex, 1 for edge, 2 for face), index of topological feature
        for j in range(len(Hz[0])):
            if Hz[i][j] == 1:
                # The current vertex i is a boundary of edge j
                graph[(0, i)].append((1, j))
                indegrees[(1, j)] += 1

    # Iterate over all face to edge mappings
    for i in range(len(Hx)):
        for j in range(len(Hx[0])):
            # The current face i has boundary edge j
            if Hx[i][j] == 1:
                graph[(1, j)].append((2, i))
                indegrees[(2, i)] += 1

    # Collect all degree 0 nods in topological dependency graph (aka all vertices)
    queue = [(0, i) for i in range(len(Hz))]
    # Permute the queue into a random order
    queue = random.sample(queue, len(queue))

    filtration = []

    # Perform a topological sort
    while len(queue) > 0:
        curr = queue.pop(0)
        filtration.append(curr)
        for next in graph[curr]:
            indegrees[next] -= 1
            if indegrees[next] == 0:
                queue.append(next)

    return filtration

def sample_filtration(Hx, Hz):
    filtration = []
    SQ = SampleQueue(deepcopy(Hx), deepcopy(Hz.T))
    while not SQ.is_empty():
        sample = SQ.sample()
        filtration.append(sample)
    return filtration

def build_simplex_tree_filtration(Hx, Hz, filtration):
    Hz_t = Hz.T
    vertex_stream = []
    for simplex in filtration:
        degree, id = simplex
        if degree == 0:
            vertex_stream.append([id])
        elif degree == 1:
            vertex_stream.append(list(np.where(Hz_t[id] == 1)[0]))
        elif degree == 2:
            # First triangulate cell
            # Compute all vertices of cell
            all_vertices = set()
            for i, edge in enumerate(Hx[id]):
                if edge == 1:
                    vertices = list(np.where(Hz_t[i] == 1)[0])
                    for vertex in vertices:
                        all_vertices.add(vertex)
            all_vertices = list(all_vertices)
            # Flip a random vertex to be 0 such that we have a set of size 3
            # Maybe do n choose 3?
            subset = random.sample(all_vertices, 3)
            vertex_stream.append(subset)
    return filtration
    

Hx = np.array([[1, 1, 0, 0, 1, 0, 1, 0],
               [1, 1, 0, 0, 0, 1, 0, 1],
               [0, 0, 1, 1, 1, 0, 1, 0],
               [0, 0, 1, 1, 0, 1, 0, 1]])
Hz = np.array([[1, 0, 1, 0, 1, 1, 0, 0],
               [0, 1, 0, 1, 1, 1, 0, 0],
               [1, 0, 1, 0, 0, 0, 1, 1],
               [0, 1, 0, 1, 0, 0, 1, 1]])

print(TS3(Hx, Hz))
filtration = sample_filtration(Hx, Hz)
st_filtration = build_simplex_tree_filtration(Hx, Hz, filtration)

st = gudhi.SimplexTree()
t = 0.0
for simplex in st_filtration:
    st.insert(simplex, t)
    t += 1.0

print(st.dimension())
print(st.num_simplices())
print(st.num_vertices())

# Compute persistence
diag = st.persistence()

# Plotting the persistence diagram
gudhi.plot_persistence_diagram(diag)
st.betti_numbers()