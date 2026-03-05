from typing import List
import numpy as np

def is_full_body(kpt_score: List[float] | List[List[float]] | np.ndarray, confidence_threshold: float = 0.5) -> bool:
    """Return True when shoulder/hip/knee/ankle keypoints all exceed threshold.

    Supported input formats:
    - [17]: scores for one person
    - [N, 17]: scores for multiple persons
    - [17, 3]: keypoints for one person as (x, y, score)
    - [N, 17, 3]: keypoints for multiple persons as (x, y, score)
    """
    scores_np = np.asarray(kpt_score, dtype=np.float32)

    required_indices = np.array([5, 6, 11, 12, 13, 14, 15, 16], dtype=np.int64)

    if scores_np.ndim == 3:
        if scores_np.shape[1] < 17 or scores_np.shape[2] < 3:
            return False
        body_scores = scores_np[:, required_indices, 2]
        return bool(np.any(np.all(body_scores > confidence_threshold, axis=1)))

    if scores_np.ndim == 2:
        if scores_np.shape[0] >= 17 and scores_np.shape[1] >= 3:
            body_scores = scores_np[required_indices, 2]
            return bool(np.all(body_scores > confidence_threshold))

        if scores_np.shape[1] < 17:
            return False
        body_scores = scores_np[:, required_indices]
        return bool(np.any(np.all(body_scores > confidence_threshold, axis=1)))

    if scores_np.ndim == 1:
        if scores_np.shape[0] < 17:
            return False
        return bool(np.all(scores_np[required_indices] > confidence_threshold))

    return False
