from typing import Dict

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class InterCameraMatching:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def match(self, query_features, gallery_features):
        """
        Input:
            query_features: List[Vector] (Candidates)
            gallery_features: List[Vector] (Lost Tracks)
        Output:
            matches: List[(query_idx, gallery_idx)]
            unmatched_query: List[query_idx]
        """
        if len(query_features) == 0 or len(gallery_features) == 0:
            return [], list(range(len(query_features)))

        # Tính Distance Matrix (Cosine)
        dist_matrix = cdist(query_features, gallery_features, metric='cosine')
        
        # Hungarian Algorithm
        row_inds, col_inds = linear_sum_assignment(dist_matrix)
        
        matches = []
        matched_query_indices = set()
        
        for row, col in zip(row_inds, col_inds):
            if dist_matrix[row, col] < (1 - self.threshold):
                matches.append((row, col))
                matched_query_indices.add(row)
        
        unmatched_query = [i for i in range(len(query_features)) if i not in matched_query_indices]
        
        return matches, unmatched_query