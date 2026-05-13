from typing import List
import numpy as np
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter

from ..track_manager.global_track_manager import GTrackManager
from ..templates.mct_templates import Pose3DResult, GTrackInfo

np.random.seed(0)

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def dist_batch(points_test, points_gt, dim=3):
  dm = cdist(points_test[:,:dim], points_gt[:,:dim], metric="euclidean")
  return dm

def convert_point_to_z(point:List, dim=3):
  return np.array(point[:dim]).reshape((len(point[:dim]), 1))

def convert_x_to_point(x:np.ndarray, dim=3, score=None):
  x = np.squeeze(x)
  p = x[:dim]
  if(score==None):
    return np.array(p).reshape(1, dim)
  else:
    p = p.tolist()
    p.append(score)
    return np.array(p).reshape(1, dim+1)

class KalmanPointTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as 3D point.
  state = [x, y, z, xdot, ydot, zdot]'
  out = [x, y, z]'
  """
  count = 0
  def __init__(self, point, dim_z=2):
    """
    Initialises a tracker using initial point.
    """
    self.dim_z = dim_z
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=6, dim_z=dim_z) 
    
    dt = 1
    # (dim_x, dim_x)
    self.kf.F = np.array(
        [
          [1, 0, 0, dt, 0, 0],
          [0, 1, 0, 0, dt, 0],
          [0, 0, 1, 0, 0, dt],
          [0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 1],
        ]
    )
    # (dim_z, dim_x)
    self.kf.H = np.array(
        [
          [1, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0],
        ]
    )

    # (dim_z, dim_z)
    # self.kf.R[dim_z:, dim_z:] *= 1.
    
    # (dim_x, dim_x)
    self.kf.P[dim_z:, dim_z:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    
    # (dim_x, dim_x)
    self.kf.Q[dim_z:, dim_z:] *= 0.01
    
    self.kf.x[:dim_z] = convert_point_to_z(point, dim=dim_z)
    self.time_since_update = 0
    self.id = KalmanPointTracker.count
    KalmanPointTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self, point):
    """
    Updates the state vector with observed point.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_point_to_z(point, dim=self.dim_z))

  def predict(self):
    """
    Advances the state vector and returns the predicted point estimate.
    """
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_point(self.kf.x, dim=self.dim_z))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current point estimate.
    """
    return convert_x_to_point(self.kf.x, dim=self.dim_z)


def associate_detections_to_trackers(detections, trackers, det_size=3, dist_thresh=100):
  """
  Assigns detections to tracked object (both represented as points)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,det_size), dtype=int)

  dist_matrix = dist_batch(detections, trackers)

  if min(dist_matrix.shape) > 0:
    a = (dist_matrix < dist_thresh).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(dist_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(dist_matrix[m[0], m[1]]>dist_thresh):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class PointTrack:
  def __init__(self, config):
    """
    Sets key parameters for SORT
    """
    self.max_age = config.get("max_age", 15)
    self.min_hits = config.get("min_hits", 3)
    self.dist_thresh = config.get("dist_thresh", 150)
    self.det_size = config.get("det_size", 4)   # [x, y, z, score]
    self.trackers = []

  def update(self, point_3ds, pose_3ds: List[Pose3DResult], track_manager: GTrackManager, frames, frame_id, timestamp):
    """
    Params:
      dets - a numpy array of detections in the format [[xc,yc,,score], [xc,yc,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, det_size)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), self.det_size))
    to_del = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(point_3ds, trks, det_size=self.det_size, dist_thresh=self.dist_thresh)

    
    # Create new track for unmatched detections
    for det_idx in unmatched_dets:
      new_track = KalmanPointTracker(point_3ds[det_idx,:], dim_z=self.det_size-1)
      self.trackers.append(new_track)
      
      # Get attribute
      track_id = new_track.id
      pose_3d = pose_3ds[det_idx]
      pose_2ds = pose_3ds[det_idx].pose_2ds
      
      new_temp_track = GTrackInfo(track_id, frame_id, timestamp)
      track_manager.add_track_to_temp_tracks(track_id, new_temp_track)
      track_manager.update_temp_track(track_id, pose_3d, pose_2ds, frame_id, timestamp)

    for det_idx, trk_idx in matched:
        self.trackers[trk_idx].update(point_3ds[det_idx, :])
        
        track_id = self.trackers[trk_idx].id
        pose_3d = pose_3ds[det_idx]
        pose_2ds = pose_3ds[det_idx].pose_2ds
        
        # CORE LOGIC
        if track_id in track_manager.temp_tracks:
            track_manager.update_temp_track(track_id, pose_3d, pose_2ds, frame_id, timestamp)
            track_info = track_manager.temp_tracks[track_id]
            if track_info.update_time >= track_manager.num_init:
                final_id = track_manager.add_new_track_to_dict(track_id)
                self.trackers[trk_idx].id = final_id
                
        elif track_id in track_manager.dict_tracks:
          track_manager.update_track(track_id, pose_3d, pose_2ds, frame_id, timestamp)
    
    for i, trk in enumerate(self.trackers):
      if trk.time_since_update > self.max_age:
          track_id = trk.id
          if track_id not in track_manager.dict_tracks:
              continue
          track_manager.dict_tracks[track_id].is_dead = True
          self.trackers.pop(i)

    alive_tracks = []
    for track in track_manager.dict_tracks.values():
        if track.end_time != timestamp:
            continue
        
        alive_tracks.append(track)
    
    return alive_tracks
    

