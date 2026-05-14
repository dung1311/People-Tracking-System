"""Debug utility for inspecting MCT results from MOT15 txt files.

Usage:
    python -m utils.mct_debug \
        --frame 33 --gid 4 \
        --txt outputs/txt/cam64_mct.txt outputs/txt/cam65_mct.txt outputs/txt/cam66_mct.txt \
        --calib data/cameras/cam64.json data/cameras/cam65.json data/cameras/cam66.json \
        --video data/videos/cam64.mp4 data/videos/cam65.mp4 data/videos/cam66.mp4

If --video is provided, visual features are extracted from the crops and
cosine distances are computed between every camera pair.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.data_templates.mct_template import CameraCalibration

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# MOT15 parser
# -----------------------------------------------------------------------

def parse_mot_file(path: str) -> Dict[int, Dict[int, List[float]]]:
    """Parse a MOT15 txt file.

    Returns:
        ``{frame_id: {global_id: [x1, y1, w, h, ...]}}``.
    """
    result: Dict[int, Dict[int, List[float]]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            gid = int(parts[1])
            vals = [float(p) for p in parts[2:]]
            result.setdefault(frame_id, {})[gid] = vals
    return result


def _bbox_foot(x1: float, y1: float, w: float, h: float) -> np.ndarray:
    """Bottom-centre from MOT15 (x1, y1, w, h)."""
    return np.array([x1 + w / 2.0, y1 + h], dtype=np.float64)


# -----------------------------------------------------------------------
# Core inspection function
# -----------------------------------------------------------------------

def inspect_global_id(
    frame_id: int,
    global_id: int,
    txt_paths: List[str],
    calib_paths: List[str],
    cam_ids: List[int] | None = None,
    video_paths: List[str] | None = None,
    embedder=None,
) -> Dict:
    """Inspect a single global ID at a given frame across cameras.

    Args:
        frame_id: which frame to look at.
        global_id: the global person ID to inspect.
        txt_paths: list of MOT15 txt files (one per camera).
        calib_paths: list of calibration JSON files (same order).
        cam_ids: optional explicit camera IDs (default: extracted from filename).
        video_paths: optional video files to extract crops & visual features.
        embedder: optional pre-initialised embedder; if None and video_paths
                  is given, a FastReID embedder is created from the config.

    Returns:
        Dict with per-camera info and pairwise costs.
    """
    n_cams = len(txt_paths)
    assert len(calib_paths) == n_cams

    if cam_ids is None:
        cam_ids = [_cam_id_from_path(p) for p in txt_paths]

    # Load calibrations
    cals = {cid: CameraCalibration.load_from_json(cp, cid)
            for cid, cp in zip(cam_ids, calib_paths)}

    # Parse MOT files
    per_cam_data: Dict[int, Optional[Dict]] = {}
    for cid, txt in zip(cam_ids, txt_paths):
        mot = parse_mot_file(txt)
        frame_data = mot.get(frame_id, {})
        per_cam_data[cid] = frame_data.get(global_id)

    # Per-camera results
    cam_results: Dict[int, Dict] = {}
    for cid in cam_ids:
        entry = per_cam_data[cid]
        if entry is None:
            cam_results[cid] = {"present": False}
            continue
        x1, y1, w, h = entry[0], entry[1], entry[2], entry[3]
        foot_px = _bbox_foot(x1, y1, w, h)
        foot_world = cals[cid].project_to_world(foot_px)
        cam_results[cid] = {
            "present": True,
            "bbox_xywh": [x1, y1, w, h],
            "foot_pixel": foot_px.tolist(),
            "foot_world": foot_world.tolist(),
        }

    # Visual features (if videos provided)
    features: Dict[int, np.ndarray] = {}
    if video_paths:
        assert len(video_paths) == n_cams
        if embedder is None:
            from modules.embedder.factory import EmbedderFactory
            from utils.load_config import load_config
            cfg = load_config("/home/dungnt/workspaces/HUST/People-Tracking-System/be/configs/sct_config.yaml")
            embedder = EmbedderFactory(cfg["TRACK_MANAGER"]["EMBEDDING"]).get_embedder()

        for cid, vpath in zip(cam_ids, video_paths):
            if not cam_results[cid]["present"]:
                continue
            x1, y1, w, h = cam_results[cid]["bbox_xywh"]
            frame = _read_frame(vpath, frame_id)
            if frame is None:
                continue
            x2, y2 = x1 + w, y1 + h
            ih, iw = frame.shape[:2]
            crop = frame[
                max(0, int(y1)):min(ih, int(y2)),
                max(0, int(x1)):min(iw, int(x2)),
            ]
            if crop.size == 0:
                continue
            feat = embedder.extract_feature([crop])[0]
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            features[cid] = feat
            cam_results[cid]["feature_norm"] = float(np.linalg.norm(feat))

    # Pairwise costs
    present_cams = [cid for cid in cam_ids if cam_results[cid]["present"]]
    pairwise: List[Dict] = []

    for ci, cj in itertools.combinations(present_cams, 2):
        wi = np.array(cam_results[ci]["foot_world"])
        wj = np.array(cam_results[cj]["foot_world"])
        homo_dist = float(np.linalg.norm(wi - wj))

        pair_info = {
            "cam_i": ci,
            "cam_j": cj,
            "homo_distance": homo_dist,
            "world_i": wi.tolist(),
            "world_j": wj.tolist(),
        }

        if ci in features and cj in features:
            cos_sim = float(np.dot(features[ci], features[cj]))
            cos_dist = 1.0 - cos_sim
            pair_info["visual_cosine_dist"] = cos_dist
            pair_info["visual_cosine_sim"] = cos_sim

        pairwise.append(pair_info)

    return {
        "frame_id": frame_id,
        "global_id": global_id,
        "cameras": cam_results,
        "pairwise": pairwise,
    }


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _cam_id_from_path(path: str) -> int:
    """Extract cam id from filename like ``cam64_mct.txt``."""
    stem = Path(path).stem
    for part in stem.split("_"):
        if part.startswith("cam"):
            try:
                return int(part[3:])
            except ValueError:
                pass
    return hash(path) % 1000


def _read_frame(video_path: str, frame_id: int) -> Optional[np.ndarray]:
    """Read the *frame_id*-th frame (1-indexed) from a video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def compare_global_ids(
    frame_id: int,
    gid_cam_pairs: List[Tuple[int, int]],
    txt_paths: List[str],
    calib_paths: List[str],
    cam_ids: List[int] | None = None,
    video_paths: List[str] | None = None,
    embedder=None,
) -> Dict:
    """Compare different global IDs across specified cameras at a given frame.

    Args:
        frame_id: which frame to look at.
        gid_cam_pairs: list of ``(global_id, cam_id)`` to compare.
            E.g. ``[(1, 64), (10, 66)]`` = G1@cam64 vs G10@cam66.
        txt_paths: list of MOT15 txt files (one per camera).
        calib_paths: list of calibration JSON files (same order).
        cam_ids: optional explicit camera IDs.
        video_paths: optional video files for feature extraction.
        embedder: optional pre-initialised embedder.

    Returns:
        Dict with per-entry info and all pairwise costs.
    """
    n_cams = len(txt_paths)
    assert len(calib_paths) == n_cams

    if cam_ids is None:
        cam_ids = [_cam_id_from_path(p) for p in txt_paths]

    cals = {cid: CameraCalibration.load_from_json(cp, cid)
            for cid, cp in zip(cam_ids, calib_paths)}

    # Build cam_id -> txt index
    cam_to_idx = {cid: i for i, cid in enumerate(cam_ids)}

    # Parse all MOT files once
    mot_data: Dict[int, Dict[int, Dict[int, List[float]]]] = {}
    for cid, txt in zip(cam_ids, txt_paths):
        mot_data[cid] = parse_mot_file(txt)

    # Gather info for each (gid, cam) entry
    entries: List[Dict] = []
    features: List[Optional[np.ndarray]] = []

    need_embedder = video_paths is not None
    if need_embedder and embedder is None:
        from modules.embedder.factory import EmbedderFactory
        from utils.load_config import load_config
        cfg = load_config("configs/sct_config.yaml")
        embedder = EmbedderFactory(cfg["TRACK_MANAGER"]["EMBEDDING"]).get_embedder()

    for gid, cid in gid_cam_pairs:
        frame_data = mot_data.get(cid, {}).get(frame_id, {})
        entry_vals = frame_data.get(gid)

        label = f"G{gid}@cam{cid}"
        if entry_vals is None:
            entries.append({"label": label, "gid": gid, "cam_id": cid, "present": False})
            features.append(None)
            continue

        x1, y1, w, h = entry_vals[0], entry_vals[1], entry_vals[2], entry_vals[3]
        foot_px = _bbox_foot(x1, y1, w, h)
        foot_world = cals[cid].project_to_world(foot_px)

        info = {
            "label": label,
            "gid": gid,
            "cam_id": cid,
            "present": True,
            "bbox_xywh": [x1, y1, w, h],
            "foot_pixel": foot_px.tolist(),
            "foot_world": foot_world.tolist(),
        }

        feat = None
        if video_paths:
            vidx = cam_to_idx.get(cid)
            if vidx is not None:
                frame_img = _read_frame(video_paths[vidx], frame_id)
                if frame_img is not None:
                    x2, y2 = x1 + w, y1 + h
                    ih, iw = frame_img.shape[:2]
                    crop = frame_img[
                        max(0, int(y1)):min(ih, int(y2)),
                        max(0, int(x1)):min(iw, int(x2)),
                    ]
                    if crop.size > 0:
                        feat = embedder.extract_feature([crop])[0]
                        feat = feat / (np.linalg.norm(feat) + 1e-8)

        entries.append(info)
        features.append(feat)

    # Pairwise costs between all present entries
    present_idx = [i for i, e in enumerate(entries) if e["present"]]
    pairwise: List[Dict] = []

    for a, b in itertools.combinations(present_idx, 2):
        ea, eb = entries[a], entries[b]
        wa = np.array(ea["foot_world"])
        wb = np.array(eb["foot_world"])
        homo_dist = float(np.linalg.norm(wa - wb))

        pair_info = {
            "entry_a": ea["label"],
            "entry_b": eb["label"],
            "homo_distance": homo_dist,
            "world_a": wa.tolist(),
            "world_b": wb.tolist(),
            "same_camera": ea["cam_id"] == eb["cam_id"],
        }

        fa, fb = features[a], features[b]
        if fa is not None and fb is not None:
            cos_sim = float(np.dot(fa, fb))
            pair_info["visual_cosine_dist"] = 1.0 - cos_sim
            pair_info["visual_cosine_sim"] = cos_sim

        pairwise.append(pair_info)

    return {
        "frame_id": frame_id,
        "entries": entries,
        "pairwise": pairwise,
    }


def pretty_print_compare(result: Dict):
    """Print comparison result."""
    print(f"\n{'='*60}")
    print(f"  Frame {result['frame_id']}  |  Comparing {len(result['entries'])} entries")
    print(f"{'='*60}")

    for e in result["entries"]:
        if not e["present"]:
            print(f"  {e['label']}: NOT PRESENT")
            continue
        bx = e["bbox_xywh"]
        fw = e["foot_world"]
        print(f"  {e['label']}:")
        print(f"    bbox (x,y,w,h) = ({bx[0]:.1f}, {bx[1]:.1f}, {bx[2]:.1f}, {bx[3]:.1f})")
        print(f"    foot world     = ({fw[0]:.2f}, {fw[1]:.2f})")

    print(f"\n  Pairwise costs:")
    print(f"  {'Pair':<28} {'Homo dist':>10} {'Vis cos_d':>10} {'Vis cos_s':>10}  Note")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}  {'-'*10}")
    for p in result["pairwise"]:
        pair = f"{p['entry_a']} vs {p['entry_b']}"
        hd = f"{p['homo_distance']:.2f}"
        if isinstance(p.get("visual_cosine_dist"), float):
            vd = f"{p['visual_cosine_dist']:.4f}"
            vs = f"{p['visual_cosine_sim']:.4f}"
        else:
            vd, vs = "N/A".rjust(10), "N/A".rjust(10)
        note = "same-cam" if p["same_camera"] else ""
        print(f"  {pair:<28} {hd:>10} {vd:>10} {vs:>10}  {note}")
    print()


def pretty_print(result: Dict):
    """Print inspection result in a readable table."""
    print(f"\n{'='*60}")
    print(f"  Frame {result['frame_id']}  |  Global ID {result['global_id']}")
    print(f"{'='*60}")

    for cid, info in result["cameras"].items():
        if not info["present"]:
            print(f"  Cam {cid}: NOT PRESENT")
            continue
        bx = info["bbox_xywh"]
        fp = info["foot_pixel"]
        fw = info["foot_world"]
        print(f"  Cam {cid}:")
        print(f"    bbox (x,y,w,h) = ({bx[0]:.1f}, {bx[1]:.1f}, {bx[2]:.1f}, {bx[3]:.1f})")
        print(f"    foot pixel     = ({fp[0]:.1f}, {fp[1]:.1f})")
        print(f"    foot world     = ({fw[0]:.2f}, {fw[1]:.2f})")

    print(f"\n  Pairwise costs:")
    print(f"  {'Pair':<14} {'Homo dist':>10} {'Vis cos_d':>10} {'Vis cos_s':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
    for p in result["pairwise"]:
        pair = f"cam{p['cam_i']}-cam{p['cam_j']}"
        hd = f"{p['homo_distance']:.2f}"
        vd = f"{p.get('visual_cosine_dist', 'N/A'):>10}" if isinstance(p.get("visual_cosine_dist"), float) else "N/A".rjust(10)
        vs = f"{p.get('visual_cosine_sim', 'N/A'):>10}" if isinstance(p.get("visual_cosine_sim"), float) else "N/A".rjust(10)
        if isinstance(p.get("visual_cosine_dist"), float):
            vd = f"{p['visual_cosine_dist']:.4f}"
            vs = f"{p['visual_cosine_sim']:.4f}"
        print(f"  {pair:<14} {hd:>10} {vd:>10} {vs:>10}")
    print()


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def check_full_body(
    frame_id: int,
    gid_cam_pairs: List[Tuple[int, int]],
    txt_paths: List[str],
    calib_paths: List[str],
    video_paths: List[str],
    cam_ids: List[int] | None = None,
    confidence_threshold: float = 0.5,
) -> Dict:
    """Check whether each (global_id, cam_id) is a full-body detection.

    Uses RTMPose to estimate keypoints and ``is_full_body`` to check
    shoulder/hip/knee/ankle visibility.

    Returns:
        Dict with per-entry pose info.
    """
    from modules.pose_estimator.factory import PoseEstimatorFactory
    from utils.load_config import load_config
    from utils.pose import is_full_body

    n_cams = len(txt_paths)
    if cam_ids is None:
        cam_ids = [_cam_id_from_path(p) for p in txt_paths]

    cam_to_idx = {cid: i for i, cid in enumerate(cam_ids)}

    cals = {cid: CameraCalibration.load_from_json(cp, cid)
            for cid, cp in zip(cam_ids, calib_paths)}

    mot_data = {}
    for cid, txt in zip(cam_ids, txt_paths):
        mot_data[cid] = parse_mot_file(txt)

    cfg = load_config("configs/sct_config.yaml")
    pe = PoseEstimatorFactory(cfg["POSE_ESTIMATION"]).get_pose_estimator()

    entries: List[Dict] = []
    for gid, cid in gid_cam_pairs:
        label = f"G{gid}@cam{cid}"
        frame_data = mot_data.get(cid, {}).get(frame_id, {})
        vals = frame_data.get(gid)

        if vals is None:
            entries.append({"label": label, "gid": gid, "cam_id": cid, "present": False})
            continue

        x1, y1, w, h = vals[0], vals[1], vals[2], vals[3]
        x2, y2 = x1 + w, y1 + h
        bbox = [x1, y1, x2, y2]

        vidx = cam_to_idx.get(cid)
        if vidx is None:
            entries.append({"label": label, "gid": gid, "cam_id": cid, "present": False})
            continue

        frame_img = _read_frame(video_paths[vidx], frame_id)
        if frame_img is None:
            entries.append({"label": label, "gid": gid, "cam_id": cid, "present": False})
            continue

        kpts_scores = pe.detect(frame_img, [bbox])
        full_body = is_full_body(kpts_scores[0], confidence_threshold) if len(kpts_scores) > 0 else False

        # Required keypoint names for readability
        KPT_NAMES = [
            "nose", "L_eye", "R_eye", "L_ear", "R_ear",
            "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
            "L_wrist", "R_wrist", "L_hip", "R_hip",
            "L_knee", "R_knee", "L_ankle", "R_ankle",
        ]
        REQUIRED = [5, 6, 11, 12, 13, 14, 15, 16]

        kpt = kpts_scores[0] if len(kpts_scores) > 0 else None
        kpt_detail = {}
        if kpt is not None:
            kpt_np = np.asarray(kpt)
            if kpt_np.ndim == 2 and kpt_np.shape[1] >= 3:
                for idx in REQUIRED:
                    score = float(kpt_np[idx, 2])
                    kpt_detail[KPT_NAMES[idx]] = {
                        "score": score,
                        "pass": score > confidence_threshold,
                    }

        foot_px = _bbox_foot(x1, y1, w, h)
        foot_world = cals[cid].project_to_world(foot_px)

        entries.append({
            "label": label,
            "gid": gid,
            "cam_id": cid,
            "present": True,
            "bbox_xywh": [x1, y1, w, h],
            "foot_world": foot_world.tolist(),
            "is_full_body": full_body,
            "keypoints": kpt_detail,
        })

    return {"frame_id": frame_id, "confidence_threshold": confidence_threshold, "entries": entries}


def pretty_print_fullbody(result: Dict):
    """Print full-body check results."""
    print(f"\n{'='*60}")
    print(f"  Frame {result['frame_id']}  |  Full-body check (conf > {result['confidence_threshold']})")
    print(f"{'='*60}")

    for e in result["entries"]:
        if not e["present"]:
            print(f"  {e['label']}: NOT PRESENT")
            continue

        status = "FULL BODY" if e["is_full_body"] else "NOT FULL BODY"
        bx = e["bbox_xywh"]
        print(f"\n  {e['label']}:  [{status}]")
        print(f"    bbox (x,y,w,h) = ({bx[0]:.1f}, {bx[1]:.1f}, {bx[2]:.1f}, {bx[3]:.1f})")
        print(f"    foot world     = ({e['foot_world'][0]:.2f}, {e['foot_world'][1]:.2f})")

        if e["keypoints"]:
            print(f"    {'Keypoint':<14} {'Score':>7} {'Pass':>6}")
            print(f"    {'-'*14} {'-'*7} {'-'*6}")
            for name, info in e["keypoints"].items():
                mark = "  Y" if info["pass"] else "  X"
                print(f"    {name:<14} {info['score']:>7.3f} {mark:>6}")
    print()


def _parse_gid_cam(s: str) -> Tuple[int, int]:
    """Parse ``'GID:CAMID'`` string, e.g. ``'1:64'`` → ``(1, 64)``."""
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected GID:CAMID (e.g. 1:64), got '{s}'"
        )
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="MCT debug: inspect or compare global IDs at a given frame",
    )
    sub = parser.add_subparsers(dest="cmd")

    # --- inspect: single global ID across all cameras ---
    p_insp = sub.add_parser("inspect", help="Inspect one global ID across all cameras")
    p_insp.add_argument("--frame", type=int, required=True)
    p_insp.add_argument("--gid", type=int, required=True)
    p_insp.add_argument("--txt", nargs="+", required=True)
    p_insp.add_argument("--calib", nargs="+", required=True)
    p_insp.add_argument("--video", nargs="*", default=None)

    # --- compare: N arbitrary (gid, cam) pairs ---
    p_cmp = sub.add_parser(
        "compare",
        help="Compare multiple GID:CAM pairs (e.g. 1:64 10:66)",
    )
    p_cmp.add_argument("--frame", type=int, required=True)
    p_cmp.add_argument("--pairs", nargs="+", type=_parse_gid_cam, required=True,
                        help="GID:CAMID pairs, e.g. 1:64 10:66 4:65")
    p_cmp.add_argument("--txt", nargs="+", required=True)
    p_cmp.add_argument("--calib", nargs="+", required=True)
    p_cmp.add_argument("--video", nargs="*", default=None)

    # --- fullbody: check if GID:CAM entries are full-body ---
    p_fb = sub.add_parser(
        "fullbody",
        help="Check if GID:CAM pairs are full-body detections",
    )
    p_fb.add_argument("--frame", type=int, required=True)
    p_fb.add_argument("--pairs", nargs="+", type=_parse_gid_cam, required=True,
                       help="GID:CAMID pairs, e.g. 4:64 9:65")
    p_fb.add_argument("--txt", nargs="+", required=True)
    p_fb.add_argument("--calib", nargs="+", required=True)
    p_fb.add_argument("--video", nargs="+", required=True)
    p_fb.add_argument("--conf", type=float, default=0.5, help="Keypoint confidence threshold")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.cmd == "inspect":
        result = inspect_global_id(
            frame_id=args.frame,
            global_id=args.gid,
            txt_paths=args.txt,
            calib_paths=args.calib,
            video_paths=args.video,
        )
        pretty_print(result)

    elif args.cmd == "compare":
        result = compare_global_ids(
            frame_id=args.frame,
            gid_cam_pairs=args.pairs,
            txt_paths=args.txt,
            calib_paths=args.calib,
            video_paths=args.video,
        )
        pretty_print_compare(result)

    elif args.cmd == "fullbody":
        result = check_full_body(
            frame_id=args.frame,
            gid_cam_pairs=args.pairs,
            txt_paths=args.txt,
            calib_paths=args.calib,
            video_paths=args.video,
            confidence_threshold=args.conf,
        )
        pretty_print_fullbody(result)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
