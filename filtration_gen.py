
import random
from collections import defaultdict

import numpy as np

def is_zero(c):
    for x in c:
        if x != 0:
            return False
    return True

class SampleQueue:

    def __init__(self, Hx, Hz_t):
        for c in range(len(Hx[0])):
            if is_zero(Hx[:, c]):
                e_c = [0] * len(Hx[0])
                e_c[c] = 1
                Hx = Hx.append(e_c)
        self.Hz_t = Hz_t
        self.Hx = Hx

    def sample(self):
        # Uniformly sample a random 2-simplex
        r_sigma = random.randint(0, len(self.Hz_t) - 1)
        sigma = self.Hz_t[r_sigma]
        # All dependencies of sigma has already been sampled
        if is_zero(sigma):
            return r_sigma
        # 


Hx = [] # b_2 x b_1: maps from faces to edges
Hz = [] # b_0 x b_1: maps from vertices to edges

b_0, b_1, b_2 = len(Hz), len(Hz[0]), len(Hx)

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