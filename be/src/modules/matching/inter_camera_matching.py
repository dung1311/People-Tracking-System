from typing import List, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.data_templates.sct_template import TrackInfo, MatchResult

class FeatureMatcher:
    """Matching module using cosine similarity"""
    
    def __init__(self, config: Dict):
        self.threshold = config["distance_threshold"]
    
    def compute_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine distance (1 - cosine similarity)"""
        feat1 = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2 = feat2 / (np.linalg.norm(feat2) + 1e-8)
        similarity = np.dot(feat1, feat2)
        return 1.0 - similarity
    
    def match(self, query_tracks: List[TrackInfo], gallery_tracks: List[TrackInfo]) -> List[MatchResult]:
        """
        Match query tracks against gallery tracks using Hungarian algorithm
        
        Args:
            query_tracks: List of new tracks to match
            gallery_tracks: List of existing tracks in gallery
            
        Returns:
            List of MatchResult for each query track
        """
        if not query_tracks or not gallery_tracks:
            return [MatchResult(i, -1, float('inf'), False) for i in range(len(query_tracks))]
        
        # Build distance matrix
        n_query = len(query_tracks)
        n_gallery = len(gallery_tracks)
        distance_matrix = np.zeros((n_query, n_gallery))
        
        for i, q_track in enumerate(query_tracks):
            q_feat = q_track.get_representative_feature()
            if q_feat is None:
                distance_matrix[i, :] = float('inf')
                continue
            
            for j, g_track in enumerate(gallery_tracks):
                g_feat = g_track.get_representative_feature()
                if g_feat is None:
                    distance_matrix[i, j] = float('inf')
                else:
                    distance_matrix[i, j] = self.compute_distance(q_feat, g_feat)
        
        # Hungarian matching
        row_indices, col_indices = linear_sum_assignment(distance_matrix)
        
        # Build results
        results = []
        matched_queries = set()
        
        for q_idx, g_idx in zip(row_indices, col_indices):
            distance = distance_matrix[q_idx, g_idx]
            is_matched = distance < self.threshold
            
            results.append(MatchResult(
                query_idx=q_idx,
                gallery_idx=g_idx if is_matched else -1,
                distance=distance,
                is_matched=is_matched,
                min_distance=np.min(distance_matrix[q_idx])  # Optional: store min distance for analysis
            ))
            matched_queries.add(q_idx)
        
        # Add unmatched queries
        for i in range(n_query):
            if i not in matched_queries:
                results.append(MatchResult(i, -1, float('inf'), False, min_distance=np.min(distance_matrix[i])))
        
        # Sort by query_idx
        results.sort(key=lambda x: x.query_idx)
        
        return results