import numpy as np
import scipy.stats as st

# Define society class
class Society:
    """
    Must provide:
        N:              population size
        prefs_vac:      proportion of [prefers 1, ambiguous, prefers 2]
        h:              cost of punishment
        k:              punishment cost
        m:              sample size (local obs of proportions)
        T:              number social partners
        c_1:            cost of adopting behavior 1
        c_2:            cost of adopting behavior 2
    """
    def __init__(self, N, prefs_vec, h, k, m, T, c_1, c_2):
        self.N = N
        self.prefs_vec = prefs_vec
        self.h = h
        self.k = k
        self.m = m
        self.T = T
        self.c_1 = c_1
        self.c_2 = c_2

    def generate(self):
        """Generates an array of society members"""
                        # IDs
        IDs = np.arange(self.N) # Generate array of N ids individuals

                        # Preferences
        # Vector of preferences, drawn from using self.prefs_vec
        # 1 denotes 1, 0 denotes ambivalent, 2 denotes behavior 2
        ones = np.repeat(1, int(self.prefs_vec[0]))
        nones = np.repeat(0, int(self.prefs_vec[1]))
        twos = np.repeat(2, int(self.prefs_vec[2]))
        # Stick everything together into single preference vector
        prefs = np.concatenate((ones, nones, twos), axis = None)

                        # Initial Beliefs
        # Each individual samples m individuals, with replacement, and observes their preferred behavior
        rng = np.random.default_rng()
        seen = [np.array([rng.choice(IDs, self.m, replace = False) for i in range(self.N)])]
        # Now have array of each m agents that each agent observes, can index for prefs
        seen_prefs = prefs[seen]
        # Map bincount to each observed set then divide by size of observed set (constant, m)
        seen_beliefs = np.array(list(map(np.bincount, seen_prefs)))/self.m