import numpy as np
import numpy.linalg as la
from scipy.optimize import linear_sum_assignment

from .hypothesis import Hypothesis, HypothesisList, get_believe
from .geometry.stereo import get_fundamental_matrix, get_homography_matrix
from .geometry.camera import AffineCamera


class Pose3D:
    def __init__(self, cfg:dict, calibs:list[AffineCamera]):
        """
        :param self.scale_to_mm: d * self.scale_to_mm = d_in_mm
            that means: if our scale is in [m] we need to set
            self.scale_to_mm = 1000
        :param merge_3dpose_dist: in [mm]
        :param self.epi_threshold:
        :param self.z_axis: some datasets are rotated around one axis
        :param self.correct_limb_size: if True remove limbs that are too long or short
        :param self.get_hypothesis:
        param calib:
        """
        self.scale_to_mm = cfg["scale_to_mm"]
        self.merge_3dpose_dist = cfg["merge_3dpose_dist"]
        self.epi_threshold = cfg["epi_threshold"]
        self.homo_threshold = cfg["homo_threshold"]
        self.pose_conf_threshold = cfg["pose_conf_threshold"]
        self.distance_threshold = cfg["distance_threshold"]
        self.epi_homo_W_matching = cfg["epi_homo_W_matching"]
        self.cosine_d_xview_thresh = cfg["cosine_d_xview_thresh"]
        self.correct_limb_size = cfg["correct_limb_size"]
        self.z_axis = cfg["z_axis"]
        self.get_hypothesis = cfg["get_hypothesis"]
        self.n_kpts = cfg["n_kpts"]
        
        self.calibs = calibs
        assert len(self.calibs) == 2   
        
        self.f_mat = get_fundamental_matrix(calibs[0].P, calibs[1].P) 
        
        # solution 1
        self.h_mat = [np.asmatrix(calibs[0].H), np.asmatrix(calibs[1].H)]
        
        # # solution 2
        # self.h_mat = get_homography_matrix(np.linalg.inv(calibs[0].H), np.linalg.inv(calibs[1].H))

        
            
    def estimate(self, poses):
        """
            poses: list of poses from n camera. Format of poses is [poses, embed, box, box_id, box_iou]
        """
        n_cameras = len(self.calibs)
        # FIX HERE TODO
        assert n_cameras == len(poses)
        
        # cleanup
        poses_ = []
        for cid in range(n_cameras):
            cam_ = []
            for pose, emb, box, box_id, box_iou in poses[cid]:
                if get_believe(pose) > self.pose_conf_threshold:
                    cam_.append((pose, emb, box, box_id, box_iou))
            poses_.append(cam_)
        poses = poses_
        # add all detections in the first frames as hypothesis
        # TODO: handle the case when there is NO pose in 1. cam
        first_cid = 0
        H = [
            Hypothesis(pose, self.calibs[0], self.epi_threshold,
                       self.homo_threshold, self.pose_conf_threshold,
                    scale_to_mm=self.scale_to_mm,
                    distance_threshold=self.distance_threshold,
                    cosine_d_xview_thresh=self.cosine_d_xview_thresh,
                    debug_2d_id=(first_cid, pid))
            for pid, pose in enumerate(poses[first_cid])]
        # print('poses', poses[first_cid])
        for cid in range(1, n_cameras):
            cam = self.calibs[cid]
            all_detections = poses[cid]

            n_hyp = len(H)
            n_det = len(all_detections)

            C = np.zeros((n_hyp, n_det))
            Mask = np.zeros_like(C).astype('int32')

            for pid, person in enumerate(all_detections):
                for hid, h in enumerate(H):
                    epi_cost, epi_veto = h.calculate_epi_cost(person, cam, self.f_mat)
                    homo_cost, homo_W, homo_veto = h.calculate_homo_cost(person, cam, self.h_mat, use='feet')
                                 
                    # print(f"Pair {hid} - {pid} - epi distance:  {round(epi_cost, 2)} {epi_veto}")
                    # print(f"Pair {hid} - {pid} - homo distance:  {round(homo_cost, 2)} {round(homo_W, 2)} {homo_veto}")                              
                    if epi_veto or homo_veto:
                        Mask[hid, pid] = 1
                        C[hid, pid] = 10000000   
 
                    else:
                        C[hid, pid] = self.epi_homo_W_matching * epi_cost + (1-self.epi_homo_W_matching) *homo_cost
            rows, cols = linear_sum_assignment(C) # cross view matching
                    
            handled_pids = set()
            for hid, pid in zip(rows, cols):
                is_masked = Mask[hid, pid] == 1
                handled_pids.add(pid)
                if is_masked: # create a new hyp
                    # even the closest other person is
                    # too far away (> threshold)
                    H.append(Hypothesis(
                        all_detections[pid],
                        cam,
                        self.epi_threshold,
                        self.homo_threshold,
                        self.pose_conf_threshold,
                        scale_to_mm=self.scale_to_mm,
                        distance_threshold=self.distance_threshold,
                        debug_2d_id=(cid, pid)))
                else: # update/add
                    H[hid].merge(all_detections[pid], cam)
                    H[hid].debug_2d_ids.append((cid, pid))

            for pid, person in enumerate(all_detections):
                if pid not in handled_pids:
                    # create a new hyp
                    H.append(Hypothesis(
                        all_detections[pid],
                        cam,
                        self.epi_threshold,
                        self.homo_threshold,
                        self.pose_conf_threshold,
                        scale_to_mm=self.scale_to_mm,
                        distance_threshold=self.distance_threshold,
                        debug_2d_id=(cid, pid)))

        surviving_H = []
        humans = []
        for hyp in H:
            if hyp.size() > 1: # matched with at least one camera
                cam_ids = [cam.cid for cam in hyp.cams]
                humans.append([hyp.get_3d_person(), hyp.kpts_emb_box_bid_iou, cam_ids])
                surviving_H.append(hyp)
        # merge closeby poses
        if self.merge_3dpose_dist > 0:
            distances = []  # (hid1, hid2, distance)
            n = len(humans)
            for i in range(n):
                for j in range(i+1, n):
                    pose1 = humans[i][0]
                    pose2 = humans[j][0]
                    distance = distance_between_poses(pose1, pose2, self.z_axis)
                    distances.append((i, j, distance * self.scale_to_mm))

            # the root merge is always the smallest hid
            # go through all merges and point higher hids
            # towards their smallest merge hid

            mergers_root = {}  # hid -> root
            mergers = {}  # root: [ hid, hid, .. ]
            all_merged_hids = set()
            for hid1, hid2, distance in distances:
                if distance > self.merge_3dpose_dist:
                    continue

                if hid1 in mergers_root and hid2 in mergers_root:
                    continue  # both are already handled

                if hid1 in mergers_root:
                    hid1 = mergers_root[hid1]

                if hid1 not in mergers:
                    mergers[hid1] = [hid1]

                mergers[hid1].append(hid2)
                mergers_root[hid2] = hid1
                all_merged_hids.add(hid1)
                all_merged_hids.add(hid2)

            merged_surviving_H = []
            merged_humans = []

            for hid in range(n):
                if hid in mergers:
                    hyp_list = [surviving_H[hid2] for hid2 in mergers[hid]]
                    hyp = HypothesisList(hyp_list)
                    pose = hyp.get_3d_person()
                    merged_surviving_H.append(hyp)
                    merged_humans.append((pose, hyp.kpts_emb_box_bid_iou, hyp.cam_ids))
                elif hid not in all_merged_hids:
                    merged_surviving_H.append(surviving_H[hid])
                    merged_humans.append(humans[hid])

            humans = merged_humans
            surviving_H = merged_surviving_H

        if self.correct_limb_size:
            # --- remove limbs with bad length ---
            ua_range = [50, 500] # mm
            la_range = [50, 550]
            ul_range = [250, 700]
            ll_range = [200, 600]

            for hid, human in enumerate(humans):
                # check left arm
                if test_distance(human[0], self.scale_to_mm, 5, 7, *ua_range):
                    humans[hid][0][7] = None
                    humans[hid][0][9] = None  # we need to disable hand too
                elif test_distance(human[0], self.scale_to_mm, 7, 9, *la_range):
                    humans[hid][0][9] = None

                # check right arm
                if test_distance(human[0], self.scale_to_mm, 6, 8, *ua_range):
                    humans[hid][0][8] = None
                    humans[hid][0][10] = None  # we need to disable hand too
                elif test_distance(human[0], self.scale_to_mm, 8, 10, *la_range):
                    humans[hid][0][10] = None

                # check left leg
                if test_distance(human[0], self.scale_to_mm, 11, 13, *ul_range):
                    humans[hid][0][13] = None
                    humans[hid][0][15] = None  # we need to disable foot too
                elif test_distance(human[0], self.scale_to_mm, 13, 15, *ll_range):
                    humans[hid][0][15] = None

                # check right leg
                if test_distance(human[0], self.scale_to_mm, 12, 14, *ul_range):
                    humans[hid][0][14] = None
                    humans[hid][0][16] = None  # we need to disable foot too
                elif test_distance(human[0], self.scale_to_mm, 14, 16, *ll_range):
                    humans[hid][0][16] = None
            
        if self.get_hypothesis:
            return humans, surviving_H
        else:
            return humans


def test_distance(human, scale_to_mm, jid1, jid2, lower, higher):
    """
    :param human: [ (x, y, z) ] * J
    :param self.scale_to_mm:
    :param jid1:
    :param jid2:
    :param lower:
    :param higher:
    :return:
    """
    a = human[jid1]
    b = human[jid2]
    if a is None or b is None:
        return False
    distance = la.norm(a - b) * scale_to_mm
    if lower <= distance <= higher:
        return False
    else:
        return True


def distance_between_poses(pose1, pose2, z_axis):
    """
    :param pose1:
    :param pose2:
    :param z_axis: some datasets are rotated around one axis
    :return:
    """
    J = len(pose1)
    assert len(pose2) == J
    distances = []
    for jid in range(J):
        if pose2[jid] is None or pose1[jid] is None:
            continue
        d = la.norm(pose2[jid] - pose1[jid])
        distances.append(d)

    if len(distances) == 0:
        # TODO check this heuristic
        # take the centre distance in x-y coordinates
        valid1 = []
        valid2 = []
        for jid in range(J):
            if pose1[jid] is not None:
                valid1.append(pose1[jid])
            if pose2[jid] is not None:
                valid2.append(pose2[jid])

        assert len(valid1) > 0
        assert len(valid2) > 0
        mean1 = np.mean(valid1, axis=0)
        mean2 = np.mean(valid2, axis=0)
        assert len(mean1) == 3
        assert len(mean2) == 3

        # we only care about xy coordinates
        mean1[z_axis] = 0
        mean2[z_axis] = 0

        return la.norm(mean1 - mean2)
    else:
        return np.mean(distances)  # TODO try different versions