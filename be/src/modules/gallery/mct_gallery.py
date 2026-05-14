"""Gallery that supports global ID remapping from MCT pipeline.

Extends InMemGallery with a method to swap local person_ids for global IDs
while keeping all internal mappings (tracker_id -> person_id, person_id -> track)
consistent.
"""

from __future__ import annotations

import logging
from typing import Dict

from modules.gallery.in_mem_gallery import InMemGallery
from modules.data_templates.sct_template import TrackInfo

logger = logging.getLogger(__name__)


class MCTGallery(InMemGallery):

    def apply_global_ids(self, pid_map: Dict[int, int]):
        """Remap person_ids in-place: ``{current_pid: new_global_id}``.

        Updates ``self.tracks``, ``self.map_id``, and each ``track.person_id``.
        Bumps ``self.next_id`` above the highest global ID to prevent future
        local-ID collisions.
        """
        if not pid_map:
            return

        new_tracks: Dict[int, TrackInfo] = {}

        for old_pid in list(self.tracks):
            track = self.tracks[old_pid]
            new_pid = pid_map.get(old_pid, old_pid)

            track.person_id = new_pid
            new_tracks[new_pid] = track
            self.map_id[track.tracker_id] = new_pid

            if old_pid != new_pid:
                logger.debug(
                    "Remap cam=%s tid=%s: pid %s -> gid %s",
                    track.cam_id, track.tracker_id, old_pid, new_pid,
                )

        self.tracks = new_tracks

        all_ids = list(pid_map.values()) + list(new_tracks.keys())
        if all_ids:
            self.next_id = max(self.next_id, max(all_ids) + 1)
