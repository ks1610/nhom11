import numpy as np
from collections import Counter

def gini(y):
    counts = Counter(y)
    impurity = 1.0
    for lbl in counts:
        prob = counts[lbl] / len(y)
        impurity -= prob**2
    return impurity

class DecisionTree:
    def __init__(self, max_depth=10, min_size=2, n_features=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.tree = None
    ...
    # (same code from your training file)
    
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_size=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.trees = []
    ...
