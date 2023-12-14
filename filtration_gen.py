
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import gudhi

K = 8

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

def sample_filtration(Hx, Hz):
    filtration = []
    SQ = SampleQueue(deepcopy(Hx), deepcopy(Hz.T))
    while not SQ.is_empty():
        sample = SQ.sample()
        filtration.append(sample)
    return filtration

def build_simplex_tree_filtration(Hx, Hz, filtration, sample=True):
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
            edges = []
            for i, edge_val in enumerate(Hx[id]):
                if edge_val == 1:
                    print(Hz_t[i])
                    vertices = list(np.where(Hz_t[i] == 1)[0])
                    edge = []
                    for vertex in vertices:
                        all_vertices.add(vertex)
                        edge.append(vertex)
                    edges.append(edge)
            all_vertices = list(all_vertices)
            # Flip a random vertex to be 0 such that we have a set of size 3
            # Maybe do n choose 3?
            if sample:
                for _ in range(K):
                    subset = random.sample(all_vertices, 3)
                    vertex_stream.append(subset)
            else:
                edge_path, start = compute_edge_path(edges, len(all_vertices))
                for edge in edge_path:
                    vertex_stream.append([start] + edge)
    return vertex_stream

def compute_edge_path(edges, num_vertices):

    start = edges[0][0]
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)

    # Initialize a stack for DFS
    stack = [(start, [start])]

    # Initialize a set to keep track of visited vertices
    visited = set()

    while stack:
        current_vertex, path = stack.pop()

        # Mark the current vertex as visited
        visited.add(current_vertex)

        if len(path) == num_vertices:
            # If all vertices have been reached
            break

        # Explore adjacent vertices
        for neighbor in graph[current_vertex]:
            if neighbor not in visited:
                # Push the neighbor onto the stack along with the updated path
                stack.append((neighbor, path + [neighbor]))
    
    # Initialize an empty list to store the pairs
    pairs_list = []

    # Iterate through the input list to create pairs
    for i in range(len(path) - 1):
        pairs_list.append([path[i], path[i + 1]])

    # Remove first edge containing the start vertex (the first edge)
    pairs_list.pop(0)
    return pairs_list, start
    

# Hx = np.array([[1, 1, 0, 0, 1, 0, 1, 0],
#                [1, 1, 0, 0, 0, 1, 0, 1],
#                [0, 0, 1, 1, 1, 0, 1, 0],
#                [0, 0, 1, 1, 0, 1, 0, 1]])
# Hz = np.array([[1, 0, 1, 0, 1, 1, 0, 0],
#                [0, 1, 0, 1, 1, 1, 0, 0],
#                [1, 0, 1, 0, 0, 0, 1, 1],
#                [0, 1, 0, 1, 0, 0, 1, 1]])

# print(TS3(Hx, Hz))
# filtration = sample_filtration(Hx, Hz)
# st_filtration = build_simplex_tree_filtration(Hx, Hz, filtration)

# st = gudhi.SimplexTree()
# t = 0.0
# for simplex in st_filtration:
#     st.insert(simplex, t)
#     t += 1.0

# print(st.dimension())
# print(st.num_simplices())
# print(st.num_vertices())

# # Compute persistence
# diag = st.persistence()

# # Plotting the persistence diagram
# gudhi.plot_persistence_diagram(diag)
# st.betti_numbers()