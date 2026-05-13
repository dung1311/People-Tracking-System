from typing import Dict
import torch

from ..templates.mct_templates import Pose3DResult, GTrackInfo, Pose2DResult
from ..embedding.fastreid.embed import Embedding
from ..matching.matching_3d import Matching3D


class GTrackManager:
    def __init__(self, track_manager_config: Dict):
        self.dict_tracks: Dict[int, GTrackInfo] = {}
        self.temp_tracks: Dict[int, GTrackInfo] = {}
        self.config = track_manager_config
        self.select_cfg = track_manager_config["VECTORIZATION"]["selection"]
        self.is_join_track = track_manager_config["join_track"]["enable"]

        vec_config = track_manager_config["VECTORIZATION"]["fast_reid"]
        self.embed = Embedding(vec_config)

        if self.is_join_track:
            self.max_dist_join = track_manager_config["join_track"]["max_dist"]
            self.matching = Matching3D(track_manager_config["join_track"])
            self._joined_tracks: Dict[int, GTrackInfo] = {}
            self.match_number = track_manager_config["join_track"]["match_number"]

        self.num_init = track_manager_config["min_hits"]
        self.min_num_embeds = track_manager_config["min_num_embeds"]
        self.max_num_embeds = track_manager_config["max_num_embeds"]
        self.max_fragment_frame = track_manager_config["max_fragment_frame"]

    def check_initted(self, track_id):
        return track_id in self.dict_tracks

    def add_track_to_temp_tracks(self, track_id, track):
        self.temp_tracks[track_id] = track

    def update_temp_track(
        self,
        track_id,
        pose_3d: Pose3DResult,
        dict_pose_2d: Dict[int, Pose2DResult],
        frame_id,
        timestamp,
    ):
        track_info = self.temp_tracks[track_id]

        # update 3D pose
        track_info.pose_3ds.append(pose_3d)
        track_info.pose_3ds = track_info.pose_3ds[-self.max_num_embeds:]

        # update per-camera 2D info
        for cid, pose_2d in dict_pose_2d.items():
            bbox = pose_2d.box
            pose = pose_2d.kpts
            # embed = torch.tensor(pose_2d.embed).reshape((-1, 512))
            embed = pose_2d.embed
            local_track = pose_2d.box_id

            track_info.bboxes[cid].append(bbox)
            track_info.bboxes[cid] = track_info.bboxes[cid][-self.max_num_embeds:]

            track_info.pose_2ds[cid].append(pose)
            track_info.pose_2ds[cid] = track_info.pose_2ds[cid][-self.max_num_embeds:]

            track_info.local_tracks[cid].append(local_track)
            track_info.local_tracks[cid] = track_info.local_tracks[cid][-self.max_num_embeds:]

            # embeddings là global
            track_info.embeddings = torch.cat((track_info.embeddings, embed), dim=0)
            track_info.embeddings = track_info.embeddings[-self.max_num_embeds:]

        track_info.end_frame = frame_id
        track_info.end_time = timestamp
        track_info.update_time += 1

    def add_new_track_to_dict(self, track_id):
        if self.is_join_track:
            match_id = self.try_reid(self.temp_tracks[track_id])
        else:
            match_id = -1
        
        if match_id == -1:
            self.dict_tracks[track_id] = self.temp_tracks[track_id]
            self.dict_tracks[track_id].is_inited = True
            print(f"INIT new global track {track_id}")
        else:
            self.recover_track(self.temp_tracks[track_id], match_id)

        self.temp_tracks.pop(track_id)
        return track_id if match_id < 0 else match_id

    def update_track(
        self,
        track_id,
        pose_3d: Pose3DResult,
        dict_pose_2d: Dict[int, Pose2DResult],
        frame_id,
        timestamp,
    ):
        track_info = self.dict_tracks[track_id]

        # update 3D pose
        track_info.pose_3ds.append(pose_3d)
        track_info.pose_3ds = track_info.pose_3ds[-self.max_num_embeds:]

        # update per-camera 2D info
        for cid, pose_2d in dict_pose_2d.items():
            bbox = pose_2d.box
            pose = pose_2d.kpts
            # embed = torch.tensor(pose_2d.embed).reshape((-1, 512))
            embed = pose_2d.embed
            local_track = pose_2d.box_id

            track_info.bboxes[cid].append(bbox)
            track_info.bboxes[cid] = track_info.bboxes[cid][-self.max_num_embeds:]

            track_info.pose_2ds[cid].append(pose)
            track_info.pose_2ds[cid] = track_info.pose_2ds[cid][-self.max_num_embeds:]

            track_info.local_tracks[cid].append(local_track)
            track_info.local_tracks[cid] = track_info.local_tracks[cid][-self.max_num_embeds:]

            # embeddings là global
            track_info.embeddings = torch.cat((track_info.embeddings, embed), dim=0)
            track_info.embeddings = track_info.embeddings[-self.max_num_embeds:]

        track_info.end_frame = frame_id
        track_info.end_time = timestamp
        track_info.update_time += 1

    def get_dead_tracks(self):
        dead_tracks = []
        for track_id, track_info in self.dict_tracks.items():
            if track_info.is_dead:
                dead_tracks.append(track_info)
        return dead_tracks

    def try_reid(self, track: GTrackInfo):
        dead_tracks = self.get_dead_tracks()
        candidate_tracks = dead_tracks

        match_id = -1
        match_results = self.matching.match_all(track, candidate_tracks)

        for match_result in match_results:
            if match_result.match_frequency >= self.match_number:
                match_id = match_result.match_id
                return match_id

        return match_id

    def recover_track(self, track: GTrackInfo, match_id: int):
        self._joined_tracks[match_id] = track

        self.dict_tracks[match_id].is_dead = False

        self.dict_tracks[match_id].pose_3ds.extend(track.pose_3ds)
        self.dict_tracks[match_id].pose_3ds = self.dict_tracks[match_id].pose_3ds[
            -self.max_num_embeds :
        ]
        for cid, _ in track.pose_2ds.items():
            self.dict_tracks[match_id].bboxes[cid].extend(track.bboxes[cid])
            self.dict_tracks[match_id].bboxes[cid] = self.dict_tracks[match_id].bboxes[
                cid
            ][-self.max_num_embeds :]

            self.dict_tracks[match_id].pose_2ds[cid].extend(track.pose_2ds[cid])
            self.dict_tracks[match_id].pose_2ds[cid] = self.dict_tracks[match_id].pose_2ds[
                cid
            ][-self.max_num_embeds :]

            self.dict_tracks[match_id].local_tracks[cid].extend(track.local_tracks[cid])
            self.dict_tracks[match_id].local_tracks[cid] = self.dict_tracks[match_id].local_tracks[
                cid
            ][-self.max_num_embeds :]

        self.dict_tracks[match_id].embeddings = torch.cat(
            (self.dict_tracks[match_id].embeddings, track.embeddings), dim=0
        )
        self.dict_tracks[match_id].embeddings = self.dict_tracks[match_id].embeddings[
            -self.max_num_embeds :
        ]

        self.dict_tracks[match_id].end_frame = track.end_frame
        self.dict_tracks[match_id].end_time = track.end_time
        self.dict_tracks[match_id].update_time += track.update_time

        print(
            f"RECOVERING: Track {track.track_id} -> Merged into existing Track {match_id}"
        )
