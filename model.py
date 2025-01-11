"""

The entire project is inspired by Törnberg, P. (2022), 
"How Digital Media Drive Affective Polarization Through Partisan Sorting", Proceedings of the National Academy of Sciences.

The simulation of opinion dynamics in this code is credited to Törnberg.

The original work is licensed under the MIT License.
Original repository: https://github.com/cssmodels/tornberg2022pnas

"""

import random
import numpy as np
import collections
import networkx as nx
from collections import defaultdict 
import math


class interaction_model:

    def __init__(self, k=2, m=3, n=6, nragents=1000, c=2, gamma=1, h=32, infection_rate = 0.5, recovery_rate = 0.3, init_prob=[0.8, 0.2], custom_network=None, seed=None):
        self.m = m
        self.k = k
        self.n = n
        self.nragents = nragents 
        self.c = c
        self.gamma = gamma
        self.h = h
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.init_prob = init_prob
        self.seed = seed
        
        # Customised network if specified, otherwise a scale-free network with additional edges
        if custom_network is not None:
            self.network = custom_network
        else:
            self.network = nx.powerlaw_cluster_graph(n=nragents, m=8, p=0.01)
            
        self.nodeid = {nid: nr for nr, nid in enumerate(self.network.nodes)}
        
    def reset(self):
        # Fix the seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        # Initialisation
        self.agent_fixed = np.random.randint(self.k, size=(self.nragents))
        self.agent_flex = np.random.randint(self.m, size=(self.nragents, self.n))
        self.agent_flex[:, -2] = np.random.choice([0, 1], size=self.nragents)
        self.agent_flex[:, -1] = np.random.choice([0, 1], size=self.nragents, p=self.init_prob)

        # List all unique combinations between nodes of same and different groups, preprocessing for faster run
        self.within = [(i1, i2) for i1, n1 in enumerate(self.agent_fixed) for i2, n2 in enumerate(self.agent_fixed) if n1 == n2 and i1 < i2]
        self.between = [(i1, i2) for i1, n1 in enumerate(self.agent_fixed) for i2, n2 in enumerate(self.agent_fixed) if n1 != n2 and i1 < i2]
    
    #Run the model
    def run(self,steps,breakat=10):
        stepssincechange = 0
        for step in range(steps):                
            if self.step():
                stepssincechange = 0 
            else:
                stepssincechange += 1
            if breakat is not None and stepssincechange > breakat*self.nragents:
                break
                
            
    def step(self):
        #Select a node at random to update
        node = random.choice(list(self.network.nodes))
        nodeid = self.nodeid[node]

        #Select interlocutors: neighbors + randomly selected
        neigh = list(self.network.neighbors(node))
        nrrandom = int(self.gamma*len(neigh))
        neighbors = random.sample(neigh,len(neigh)-nrrandom) + random.sample(list(self.network.nodes),nrrandom)

        #Pick the other node based on the similarity
        weights = np.array([self.similarity(nodeid,self.nodeid[neighbor])**self.h for neighbor in neighbors])
        if sum(weights)==0: #No similarities with any neighbor: no change
            return False
        
        weights = weights/sum(weights)
        
        #Randomly pick another node on the basis of weights, urn model
        othernode = np.random.choice(len(neighbors),p=weights)
        othernodeid = self.nodeid[neighbors[othernode]]
                
        # Update the flexible attributes
        diff = np.array(
            [
                1 if self.agent_flex[nodeid][dimension] != self.agent_flex[othernodeid][dimension] else 0
                for dimension in range(self.n - 1)  # The last column is for the infection status
            ]
        )
        if sum(diff) == 0:
            return False

        diff = diff / sum(diff)
        dimension = np.random.choice(self.n - 1, p=diff)
        self.agent_flex[nodeid][dimension] = self.agent_flex[othernodeid][dimension]
        self.infection_spread(nodeid)

        return True


    # def step(self):
    #     """
    #     Simulating a situation where attitude updates occur faster than the infection updates
    #     """
    #     #Select a node at random to update
    #     infection_target_node = random.choice(list(self.network.nodes))
    #     infection_target_nodeid = self.nodeid[infection_target_node]

    #     # 10 attitude updates per step
    #     for _ in range(10):
    #         # Select a node at random to update
    #         node = random.choice(list(self.network.nodes))
    #         nodeid = self.nodeid[node]

    #         # Select interlocutors: neighbors + randomly selected
    #         neigh = list(self.network.neighbors(node))
    #         nrrandom = int(self.gamma * len(neigh))
    #         neighbors = random.sample(neigh, len(neigh) - nrrandom) + random.sample(list(self.network.nodes), nrrandom)

    #         # Pick the other node based on the similarity
    #         weights = np.array([self.similarity(nodeid, self.nodeid[neighbor])**self.h for neighbor in neighbors])
    #         if sum(weights) == 0:  # No similarities with any neighbor: no change
    #             continue

    #         weights = weights / sum(weights)

    #         # Randomly pick another node on the basis of weights, urn model
    #         othernode = np.random.choice(len(neighbors), p=weights)
    #         othernodeid = self.nodeid[neighbors[othernode]]

    #         # Update the flexible attributes
    #         diff = np.array(
    #             [
    #                 1 if self.agent_flex[nodeid][dimension] != self.agent_flex[othernodeid][dimension] else 0
    #                 for dimension in range(self.n - 1)  # The last column is for the infection status
    #             ]
    #         )
    #         if sum(diff) == 0:
    #             continue

    #         diff = diff / sum(diff)
    #         dimension = np.random.choice(self.n - 1, p=diff)
    #         self.agent_flex[nodeid][dimension] = self.agent_flex[othernodeid][dimension]

    #     # Infection update
    #     self.infection_spread(infection_target_nodeid)

    #     return True

     
    def infection_spread(self, nodeid):
        # S -> I
        if self.agent_flex[nodeid, -1] == 0:  
            infected_neighbors = sum(
                self.agent_flex[self.nodeid[neighbor], -1] == 1 for neighbor in self.network.neighbors(nodeid)
            )
            attitude_factor = self.agent_flex[nodeid, -2]
            adjusted_infection_rate = self.infection_rate * attitude_factor

            infection_probability = (1 - (1 - adjusted_infection_rate) ** infected_neighbors) 
            if random.random() < infection_probability:
                self.agent_flex[nodeid, -1] = 1

        # I -> S
        elif self.agent_flex[nodeid, -1] == 1:  
            if random.random() < self.recovery_rate:
                self.agent_flex[nodeid, -1] = 0
                self.agent_flex[nodeid, -2] = 1
        return True
        

    def similarity(self, a1, a2):
        return (
            (self.c if self.agent_fixed[a1] == self.agent_fixed[a2] else 0) +
            sum(1 if self.agent_flex[a1][i] == self.agent_flex[a2][i] else 0 for i in range(self.n - 1))
        ) / (self.c + self.n - 1)

    def calculate_sorting(self):
        within = np.mean([self.fraction_shared_flex(a,b) for a,b in self.within])
        between = np.mean([self.fraction_shared_flex(a,b) for a,b in self.between])
        return within - between

    def fraction_shared_flex(self, a1, a2):
        return sum(self.agent_flex[a1, :-1] == self.agent_flex[a2, :-1]) / (self.n - 1)

    def infected_population(self):
        return sum(self.agent_flex[:, -1])

