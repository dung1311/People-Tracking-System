"""SingleTrackManager variant that uses MCTGallery for global-ID remapping."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from modules.gallery.mct_gallery import MCTGallery
from modules.track_manager.single_track_manager import SingleTrackManager
from modules.data_templates.sct_template import TrackInfo, TrackState

logger = logging.getLogger(__name__)


class MCTTrackManager(SingleTrackManager):
    """Drop-in replacement for SingleTrackManager inside an MCT pipeline.

    The only behavioural difference is that the gallery is an ``MCTGallery``
    which exposes ``apply_global_ids()`` so the pipeline can push global IDs
    back after cross-camera matching.
    """

    def __init__(self, config: Dict, embedder=None, **kwargs):
        gallery = MCTGallery(config["GALLERY"])
        super().__init__(config, gallery=gallery, embedder=embedder, **kwargs)

    def apply_global_ids(self, pid_map: Dict[int, int]):
        """Remap local person_ids → global IDs inside the gallery."""
        self.gallery.apply_global_ids(pid_map)

    def get_active_tracks(self) -> List[TrackInfo]:
        """Return currently ACTIVE tracks from the gallery."""
        return [
            t for t in self.gallery.tracks.values()
            if t.state == TrackState.ACTIVE
        ]
