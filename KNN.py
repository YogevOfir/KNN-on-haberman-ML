from itertools import combinations
import numpy as np


 # calculate distance using Lp norm
def calc_lp_dist(p1: np.ndarray, p2: np.ndarray, p: float = 2) -> float:
    if p == np.inf:
        return np.max(np.abs(p1 - p2))
    elif p == 0:
        return np.sum(p1 != p2)
    else:
        return np.linalg.norm(p1 - p2, ord=p)

class KNN:
        
    def __init__(self, k: int, distance_lp: float = 2):
        self.k = k
        self.T = []
        self.distance_lp = distance_lp

    # train the model
    def train_model(self, data: np.ndarray):
        min_distance = np.inf
    
    # Calculate the minimum distance between different class points
        for p1, p2 in combinations(data, 2):
            if not np.array_equal(p1[:-1], p2[:-1]) and p1[-1] != p2[-1]:
                distance = calc_lp_dist(p1[:-1], p2[:-1], self.distance_lp)
                if distance < min_distance:
                    min_distance = distance

    # Add a random point from data to the set T
        self.T.append(data[0])

    # Add points to set T
        for p in data[1:]:
            for t in self.T:
                if calc_lp_dist(p, t, self.distance_lp) >= min_distance:
                    self.T.append(p)
                    break

    
    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        y_predicted = np.zeros(len(X))
        idx = 0
    
    # Predict labels for each point in X
        for point in X:
            distances = [calc_lp_dist(point, t[:-1], self.distance_lp) for t in self.T]
            distances = np.array(distances)
            # Find k nearest neighbors
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.T[i][-1] for i in k_nearest_indices]
            # Predict label based on majority voting
            y_predicted[idx] = np.argmax(np.bincount(k_nearest_labels))
            idx += 1
    
        return y_predicted


