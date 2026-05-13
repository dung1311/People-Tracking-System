from typing import Dict, List
from scipy.spatial.distance import cdist

from ..templates.mct_templates import GTrackInfo, MatchResult

METRIC = 'cosine'

class Matching3D:
    def __init__(self, cfg: Dict):
        self.distance_threshold = cfg["threshold"]

    def match_one_id(self, query_track: GTrackInfo, gallery_track: GTrackInfo) -> MatchResult:
        match_result = MatchResult(-1, 0.0, 0.0)

        g_id = gallery_track.track_id
        q_embeds = query_track.embeddings
        g_embeds = gallery_track.embeddings
        
        match_distance, match_frequency = self.calculate_distance(q_embeds, g_embeds)
        match_result.match_id = g_id
        match_result.match_distance = match_distance
        match_result.match_frequency = match_frequency
        print(f'match with ID {g_id} {match_result.match_distance}, {match_result.match_frequency}, {q_embeds.shape}, {g_embeds.shape}')
        return match_result

    def match_all(self, query_track, l_gallery_track) -> List[MatchResult]:
        match_results = [self.match_one_id(query_track, gallery_track) for gallery_track in l_gallery_track]
        match_results.sort(key=lambda x: (x.match_frequency, -x.match_distance), reverse=True)
        
        return match_results

    def calculate_distance(self, query_embeds, gallery_embeds):
        if len(query_embeds) == 0:
            return 0.0, 0.0
        dist_matrix = cdist(query_embeds, gallery_embeds, metric=METRIC)
        match_mask = dist_matrix < self.distance_threshold
        match_frequency = int(match_mask.sum())
        if match_frequency > 0:
            mean_dist = float(dist_matrix[match_mask].mean())
        else:
            mean_dist = float(dist_matrix.mean())
            
        return mean_dist, match_frequency