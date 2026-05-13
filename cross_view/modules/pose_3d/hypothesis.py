import cv2
import numpy as np
import numpy.linalg as la
import math
from scipy.spatial import distance as sci_d

from .geometry.stereo import get_fundamental_matrix
from .geometry.geometry import line_to_point_distance, from_homogeneous


def get_single_human3d(humans3d):
    J = len(humans3d[0])
    human3d = [None] * J  # single 3d person
    for jid in range(J):
        pts3d = []
        for person3d in humans3d:
            if person3d[jid] is not None:
                pts3d.append(person3d[jid])

        if len(pts3d) > 0:
            pt3d = np.mean(pts3d, axis=0)
            human3d[jid] = pt3d
    return human3d


def get_distance3d(person1, person2):
    J = len(person1)
    assert len(person2) == J
    result = []
    for jid in range(J):
        if person1[jid] is None or person2[jid] is None:
            continue
        d = la.norm(person1[jid] - person2[jid])
        result.append(d)
    return np.array(result)


def merge3d(persons3d, weights):
    """
    :param person3d:
    :param weights:
    :return:
    """
    assert len(persons3d) == len(weights)
    assert 1.001 > np.sum(weights) > 0.99
    J = len(persons3d[0])
    n = len(persons3d)
    result = [None] * J
    for jid in range(J):
        w_acc = 0
        never_hit = True
        pt3d = np.array([0, 0, 0], np.float32)
        for i in range(n):
            if persons3d[i][jid] is not None:
                w = weights[i]
                w_acc += w
                pt3d += w * persons3d[i][jid]
                never_hit = False
        if not never_hit:
            pt3d = pt3d / w_acc
            result[jid] = pt3d
    return result

def calculate_cost(cam1, person1, cam2, person2):
    """ calculate the epipolar distance between two humans
    :param cam1:
    :param person1:
    :param cam2:
    :param person2:
    :return:
    """
    F = get_fundamental_matrix(cam1.P, cam2.P)
    J = len(person1)
    assert J == len(person2)
    # print('Fundamental matrix: ', F)
    # print('person1', person1)
    # print('person2', person2)
    # drop all points that are -1 -1 (not visible)
    pts1 = []
    pts2 = []
    weights1 = []
    weights2 = []
    for jid in range(J):
        x1, y1, w1 = person1[jid]
        x2, y2, w2 = person2[jid]
        if x1 >= 0 and x2 >= 0:
            pts1.append((x1, y1))
            weights1.append(w1)
            pts2.append((x2, y2))
            weights2.append(w2)
    weights1 = np.clip(weights1, a_min=0, a_max=1)
    weights2 = np.clip(weights2, a_min=0, a_max=1)

    if len(pts1) == 0:
        return np.finfo(np.float32).max

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    epilines_1to2 = np.squeeze(
        cv2.computeCorrespondEpilines(pts1, 1, F))

    epilines_2to1 = np.squeeze(
        cv2.computeCorrespondEpilines(pts2, 2, F))

    total = 0

    n_pairs = len(pts1)
    assert n_pairs == len(pts2)

    if n_pairs == 1:
        epilines_1to2 = np.expand_dims(epilines_1to2, axis=0)
        epilines_2to1 = np.expand_dims(epilines_2to1, axis=0)

    for p1, l1to2, w1, p2, l2to1, w2 in zip(
            pts1, epilines_1to2, weights1,
            pts2, epilines_2to1, weights2):
        d1 = line_to_point_distance(*l1to2, *p2)
        d2 = line_to_point_distance(*l2to1, *p1)
        total += d1 + d2
    return total / n_pairs  # normalize

class Hypothesis:
    def __init__(self, kpts_emb_box_bid_iou, cam, epi_threshold, homo_threshold,
                 pose_conf_thresh,
                 scale_to_mm,
                 distance_threshold,
                 cosine_d_xview_thresh=0.2,
                 debug_2d_id=None):
        """
        :param pts: [ (x, y, w), ... ] * J
        :param cam: ProjectiveCamera
        :param threshold: if cost is larger then this
            value then the 'other' must not be merged
        :param scale_to_mm: d * scale_to_mm = d_in_mm
            that means: if our scale is in [m] we need to set
            scale_to_mm = 1000
        :param variance_threshold: in [mm]. When exceeded we try
            to drop the joint that causes the problem
        """
        self.scale_to_mm = scale_to_mm
        self.distance_threshold = distance_threshold
        self.kpts_emb_box_bid_iou = [kpts_emb_box_bid_iou]
        self.cams = [cam]
        self.epi_threshold = epi_threshold
        self.homo_threshold = homo_threshold
        self.cosine_d_xview_thresh = cosine_d_xview_thresh
        if debug_2d_id is not None:  # only for debugging
            self.debug_2d_ids = [debug_2d_id]
            
        self.pose_conf_thresh = pose_conf_thresh
        self.points = [kpts[0] for kpts in self.kpts_emb_box_bid_iou]

    def size(self):
        return len(self.kpts_emb_box_bid_iou)

    def get_3d_person(self):
        assert self.size() > 1
        distance_threshold = self.distance_threshold
        scale_to_mm = self.scale_to_mm
        humans2d = []
        for cid, (cam, human) in enumerate(zip(self.cams, self.kpts_emb_box_bid_iou)):
            human2d = Person2d(cid, cam, human[0])
            humans2d.append(human2d)

        strong_humans2d = []
        weak_humans2d = []
        for person in humans2d:
            if person.believe > self.pose_conf_thresh:  # .45
                strong_humans2d.append(person)
            else:
                weak_humans2d.append(person)

        if len(strong_humans2d) < 2:
            strong_humans2d = humans2d
            weak_humans2d = []

        strong_humans3d = []
        n = len(strong_humans2d)
        for pid1 in range(n - 1):
            for pid2 in range(pid1 + 1, n):
                person1 = strong_humans2d[pid1]
                person2 = strong_humans2d[pid2]
                person3d, _ = person1.triangulate(person2)
                strong_humans3d.append(person3d)

        strong_human3d = get_single_human3d(strong_humans3d)

        merge_targets = []
        for weak_human2d in weak_humans2d:
            humans3d_normal = []
            for strong_human2d in strong_humans2d:
                person3d, _ = weak_human2d.triangulate(strong_human2d)
                humans3d_normal.append(person3d)
            human3d_normal = get_single_human3d(humans3d_normal)

            humans3d_mirror = []
            weak_human2d_mirror = Person2d.flip_lr(weak_human2d)
            for strong_human2d in strong_humans2d:
                person3d, _ = weak_human2d_mirror.triangulate(strong_human2d)
                humans3d_mirror.append(person3d)
            human3d_mirror = get_single_human3d(humans3d_mirror)

            d_normal = np.mean(
                get_distance3d(strong_human3d, human3d_normal) * scale_to_mm)
            d_mirror = np.mean(
                get_distance3d(strong_human3d, human3d_mirror) * scale_to_mm)

            human3d_select = human3d_normal
            if d_normal > d_mirror:
                human3d_select = human3d_mirror

            if distance_threshold > 0:
                if min(d_normal, d_mirror) < distance_threshold:
                    merge_targets.append(human3d_select)
            else:
                merge_targets.append(human3d_select)  # always choose

        n = len(merge_targets)
        if n > 0:
            weights = [1] + [1/(n+1)] * n
            weights = np.array(weights)
            weights = weights/np.sum(weights)

            human3d = merge3d([strong_human3d] + merge_targets, weights)
        else:
            human3d = strong_human3d

        return human3d

    def calculate_epi_cost(self, o_points, o_cam, f_mat):
        """
        :param o_points: other points * J
        :param o_cam: other camera
        :return:
        """
        veto = False  # if true we cannot join {other} with this
        total_cost = 0
        for person, cam in zip(self.kpts_emb_box_bid_iou, self.cams):
            cost = calculate_cost(cam, person[0],
                                  o_cam, o_points[0])
            total_cost += cost
            if cost > self.epi_threshold and get_believe(person[0]) > self.pose_conf_thresh:
                veto = True

        return total_cost / len(self.kpts_emb_box_bid_iou), veto

    def calculate_homo_cost(self, o_points, o_cam, h_mat, conf_thresh=0.25, use='bbox'):
        """
        :param o_points: other points (Joints)
        :param o_cam: other camera (unused here)
        :param h_mat: list of homographies [H_src_to_world, H_dst_to_world]
        :param use: 'feet', 'bbox', or 'combine'
        :return: (average cost, homo weight, veto)
        """
        veto = False
        total_cost = 0
        homo_W = 0
        for person, cam in zip(self.kpts_emb_box_bid_iou, self.cams):
            kpts1 = person[0]  # (17, 3)
            bbox1 = person[2]  # [x1, y1, x2, y2]

            kpts2 = o_points[0]  # other person keypoints
            bbox2 = o_points[2]  # other person bbox

            p1_ankle = kpts1[15]
            p2_ankle = kpts1[16]
            o1_ankle = kpts2[15]
            o2_ankle = kpts2[16]

            p_foot = np.array([(p1_ankle[0] + p2_ankle[0]) / 2, (p1_ankle[1] + p2_ankle[1]) / 2])
            o_foot = np.array([(o1_ankle[0] + o2_ankle[0]) / 2, (o1_ankle[1] + o2_ankle[1]) / 2])

            p_bbox_bot = np.array([(bbox1[0] + bbox1[2]) / 2, bbox1[3]])
            o_bbox_bot = np.array([(bbox2[0] + bbox2[2]) / 2, bbox2[3]])

            if use == 'feet':
                p_point = p_foot
                o_point = o_foot
            elif use == 'bbox':
                p_point = p_bbox_bot
                o_point = o_bbox_bot
            elif use == 'combine':
                p_point = (p_foot + p_bbox_bot) / 2
                o_point = (o_foot + o_bbox_bot) / 2
            else:
                raise ValueError(f"Invalid 'use' mode: {use}")

            w_p1, w_p2 = p1_ankle[2], p2_ankle[2]
            w_o1, w_o2 = o1_ankle[2], o2_ankle[2]
            # homo_W = 1 - math.sqrt(1 - (w_p1 + w_p2 + w_o1 + w_o2) / 4)

            p_point_h = np.append(p_point, 1).reshape(3, 1)  # Homogeneous coordinates
            p_in_world = h_mat[0] @ p_point_h  # Project to world plane (z=0)
            p_in_world /= p_in_world[2]  # Normalize homogeneous coordinates
            o_point_h = np.append(o_point, 1).reshape(3, 1)  # Homogeneous coordinates  
            o_in_world = h_mat[1] @ o_point_h  # Project to world plane (z=0)
            o_in_world /= o_in_world[2]  # Normalize homogeneous coordinates

            p_world_2d = p_in_world[:3].flatten()  # Lấy tọa độ x,y trên mặt phẳng
            o_world_2d = o_in_world[:3].flatten()  # Lấy tọa độ x,y trên mặt phẳng
            cost = np.linalg.norm(p_world_2d - o_world_2d)*self.scale_to_mm  # Euclidean distance trên mặt phẳng
            total_cost += cost
            if cost > self.homo_threshold and get_believe(kpts1) > self.pose_conf_thresh:
                veto = True

        avg_cost = total_cost / len(self.kpts_emb_box_bid_iou)
        return avg_cost, homo_W, veto

    def calculate_cost(self, o_points, o_cam):
        """
        :param o_points: other points * J
        :param o_cam: other camera
        :return:
        """
        veto = False  # if true we cannot join {other} with this
        total_cost = 0
        for person, cam in zip(self.points, self.cams):
            cost = calculate_cost(cam, person[0],
                                  o_cam, o_points[0])
            total_cost += cost
            if cost > self.epi_threshold and get_believe(person) > 0.5:
                veto = True
            print(cost)
        return total_cost / len(self.points), veto

    def merge(self, o_points, o_cam):
        """ integrate {other} into our hypothesis
        :param o_points:
        :param o_cam:
        :return:
        """
        self.cams.append(o_cam)
        self.kpts_emb_box_bid_iou.append(o_points)

    def calc_cosine_dist_cost(self, o_person):
        veto = False  # if true we cannot join {other} with this
        total_cost = 0
        for person, cam in zip(self.kpts_emb_box_bid_iou, self.cams):
            cost = sci_d.cosine(person[1], o_person[1])
            total_cost += cost
            if cost > self.cosine_d_xview_thresh and get_believe(person[0]) > self.pose_conf_thresh:
                veto = True

        return total_cost / len(self.kpts_emb_box_bid_iou), veto


class HypothesisList:

    def __init__(self, hypothesis_list):
        """
        :param hypothesis_list: list of hypothesis'
        """
        self.hypothesis_list = hypothesis_list

    def get_3d_person(self):
        """
        Get an avg 3d pose from multiple 3d poses
        """
        poses = [] # list of 3d poses
        for hyp in self.hypothesis_list:
            pose = hyp.get_3d_person()
            poses.append(pose)

        J = len(poses[0]) # 17 for COCO
        result = [None] * J
        for jid in range(J):
            valid_points = []
            for pose in poses:
                if pose[jid] is not None:
                    valid_points.append(pose[jid])
            if len(valid_points) > 0:
                result[jid] = np.mean(valid_points, axis=0)
            else:
                result[jid] = None
        return result

    @property
    def kpts_emb_box_bid_iou(self):
        ret = []
        for hyp in self.hypothesis_list:
            kpts_emb_box_bid_iou = hyp.kpts_emb_box_bid_iou
            ret = kpts_emb_box_bid_iou
        return ret
    
    @property
    def cam_ids(self):
        c_ids = []
        for hyp in self.hypothesis_list:
            cam_ids = [cam.cid for cam in hyp.cams]
            c_ids = cam_ids
        return c_ids


def get_believe(points2d):
    believe = []
    J = len(points2d)
    for jid in range(J):
        w = points2d[jid, 2]
        if w >= 0:
            believe.append(w)
    return np.mean(believe)


class Person2d:

    @staticmethod
    def flip_lr(person):
        """ creates a new person with left and right flipped
        :param person: {Person2d}
        :return:
        """
        left  = [1, 3, 5, 7, 9, 11, 13, 15]
        right = [2, 4, 6, 8, 10, 12, 14, 16]
        
        lr = left + right
        rl = right + left

        points2d = person.points2d.copy()
        points2d[lr] = points2d[rl]

        new_person = Person2d(person.cid, person.cam, points2d)
        return new_person

    def __init__(self, cid, cam, points2d, noundistort=False):
        """
        :param cid
        :param cam: {Camera}
        :param points2d: distorted points
        :param noundistort: if True do not undistort
        """
        self.cid = cid
        self.cam = cam
        self.believe = get_believe(points2d)

        if noundistort:
            self.points2d = points2d
        else:
            # ~~~ undistort ~~~
            valid_points2d = []
            jids = []
            for jid, pt2d in enumerate(points2d):
                if pt2d[0] < 0:
                    continue
                jids.append(jid)
                valid_points2d.append(pt2d)
            valid_points2d = np.array(valid_points2d, np.float32)
            points2d_undist = cam.undistort_points(valid_points2d)
            self.points2d = points2d.copy()
            for idx, jid in enumerate(jids):
                self.points2d[jid] = points2d_undist[idx]
            # ~~~~~~~~~~~~~~~~~~~~

    def __len__(self):
        return 17

    def triangulate(self, other):
        """
        :param other: {Person2d}
        :return:
        """
        Pts1 = []
        Pts2 = []
        jids = []
        W1 = []
        W2 = []

        J = len(other)
        assert J == len(self.points2d)
        assert J == len(self)

        for jid in range(J):
            if self.points2d[jid, 2] > 0 and \
                    other.points2d[jid, 2] > 0:
                Pts1.append(self.points2d[jid, 0:2])
                Pts2.append(other.points2d[jid, 0:2])
                jids.append(jid)
                W1.append(self.points2d[jid, 2])
                W2.append(other.points2d[jid, 2])

        Pts1 = np.transpose(Pts1)
        Pts2 = np.transpose(Pts2)

        Points3d = [None] * J
        w = [-1] * J
        if len(Pts1) > 0:
            Pts3d = from_homogeneous(
                np.transpose(cv2.triangulatePoints(
                    self.cam.P, other.cam.P, Pts1, Pts2)))

            for jid, pt3d, w1, w2 in zip(jids, Pts3d, W1, W2):
                Points3d[jid] = pt3d
                w[jid] = min(w1, w2)

        return Points3d, np.array(w)
