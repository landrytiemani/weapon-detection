import os
import glob
import time
import csv
import cv2
import yaml
import pytz
import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import re

from stages.stage_2_persondetection import PersonDetectionStage
from stages.stage_3_weapondetection import WeaponDetectionStage


# ---------- utils ----------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_yolo_labels(path_txt):
    """Read YOLO txt -> list[[cls,cx,cy,w,h], ...]; [] if file missing/empty."""
    if not os.path.exists(path_txt):
        return []
    out = []
    with open(path_txt, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                out.append([cls, cx, cy, w, h])
    return out

def write_yolo_labels(path_txt, boxes):
    """Write YOLO txt from list[[cls,cx,cy,w,h], ...]."""
    if not boxes:
        open(path_txt, "w").close()
        return
    with open(path_txt, "w") as f:
        for cls, cx, cy, w, h in boxes:
            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def make_dataset_yaml(dataset_root, names):
    """Ultralytics requires train/val/test keys even for test-only eval."""
    data = {
        "path": os.path.abspath(dataset_root),
        "train": "test/images",
        "val":   "test/images",
        "test":  "test/images",
        "names": {i: n for i, n in enumerate(names)},
        "nc":    len(names),
    }
    out = os.path.join(dataset_root, "data.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out

def square_scale_clip_xyxy(x1, y1, x2, y2, img_w, img_h, scale):
    """SQUARE crop centered on bbox center; side = scale * max(w,h); clipped to image."""
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    side = max(w, h) * float(scale)

    nx1 = max(0.0, cx - side / 2.0)
    ny1 = max(0.0, cy - side / 2.0)
    nx2 = min(float(img_w - 1), cx + side / 2.0)
    ny2 = min(float(img_h - 1), cy + side / 2.0)

    if nx2 <= nx1 + 1 or ny2 <= ny1 + 1:
        nx1 = max(0.0, min(nx1, img_w - 2))
        ny1 = max(0.0, min(ny1, img_h - 2))
        nx2 = min(float(img_w - 1), nx1 + 2)
        ny2 = min(float(img_h - 1), ny1 + 2)
    return [nx1, ny1, nx2, ny2]

def remap_frame_labels_to_crop(orig_boxes, img_w, img_h, crop_xyxy, allowed_cls=None):
    """Map full-frame YOLO boxes → crop coords (normalized), only classes in allowed_cls if provided."""
    x1, y1, x2, y2 = map(float, crop_xyxy)
    cw = max(1.0, x2 - x1)
    ch = max(1.0, y2 - y1)
    out = []
    for cls, cx, cy, w, h in orig_boxes:
        if allowed_cls is not None and int(cls) not in allowed_cls:
            continue
        ax = cx * img_w; ay = cy * img_h
        aw = w * img_w;  ah = h * img_h
        bx1 = ax - aw / 2; by1 = ay - ah / 2
        bx2 = ax + aw / 2; by2 = ay + ah / 2

        ix1 = max(bx1, x1); iy1 = max(by1, y1)
        ix2 = min(bx2, x2); iy2 = min(by2, y2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        nw = ix2 - ix1; nh = iy2 - iy1
        ncx = (ix1 + ix2) / 2 - x1
        ncy = (iy1 + iy2) / 2 - y1

        nx = min(max(ncx / cw, 0.0), 1.0)
        ny = min(max(ncy / ch, 0.0), 1.0)
        nnw = min(max(nw / cw, 0.0), 1.0)
        nnh = min(max(nh / ch, 0.0), 1.0)
        if nnw > 0 and nnh > 0:
            out.append([cls, nx, ny, nnw, nnh])
    return out


# ---------- Single experiment pipeline ----------
class SingleExperiment:
    def __init__(self, config_path, use_tracker, frame_gap, experiment_name, output_dir):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        # Override tracking settings for this experiment
        self.cfg["stage_2"]["use_tracker"] = use_tracker
        self.cfg["stage_2"]["frame_gap"] = frame_gap
        self.experiment_name = experiment_name
        self.base_output_dir = output_dir

        p_cfg = self.cfg["pipeline"]
        self.frames_dir = p_cfg.get("frames_dir", "testing_data/gun_action/test/images")
        self.labels_dir = p_cfg.get("labels_dir", "testing_data/gun_action/test/labels")

        self.dataset_root = os.path.join(output_dir, f"crops_{experiment_name}")
        self.dataset_split = p_cfg.get("dataset_split", "test")

        # square crops default: 1.8× largest side
        self.crop_scale = float(self.cfg["stage_2"].get("crop_scale", 1.8))

        # FLOPs probe sizes
        s2cfg = self.cfg.get("stage_2", {})
        self.stage2_flops_imgsz = int(s2cfg.get("flops_imgsz", 800))
        self.stage3_flops_imgsz = int(self.cfg.get("evaluation", {}).get("flops_imgsz", 640))

        # Tracking settings for this experiment
        self.use_tracker = use_tracker
        self.frame_gap = frame_gap

        # Stage-2 model - Create fresh instance for each experiment
        self.person_stage = PersonDetectionStage(self.cfg["stage_2"])
        
        # Reset any internal counters in the detector
        if hasattr(self.person_stage, 'detector'):
            if hasattr(self.person_stage.detector, 'reset'):
                self.person_stage.detector.reset()
            # Reset frame counter if it exists
            if hasattr(self.person_stage.detector, 'frame_count'):
                self.person_stage.detector.frame_count = 0
            if hasattr(self.person_stage.detector, 'frame_idx'):
                self.person_stage.detector.frame_idx = 0

        # Stage-3 settings
        s3 = self.cfg["stage_3"]
        method = s3.get("approach", "")
        self.stage3_model_path = s3.get(method, {}).get("model_path", "")
        self.class_names = s3.get("names", ["weapon"])
        self.allowed_cls = set(range(len(self.class_names)))
        self.weapon_stage = WeaponDetectionStage(self.cfg["stage_3"])
        

        # Eval
        e_cfg = self.cfg.get("evaluation", {})
        self.eval_split = e_cfg.get("split", "test")
        self.enable_eval = bool(e_cfg.get("enabled", True))

        # Coverage counters (by class)
        self.gt_total_by_cls = defaultdict(int)
        self.gt_covered_by_s2_by_cls = defaultdict(int)

        # Tracking metrics
        self.stage2_times_ms = []
        self.detection_times_ms = []
        self.tracking_times_ms = []
        self.detection_count = 0
        self.tracking_count = 0
        
        # Per-frame crop counts for accurate latency calculation
        self.crops_per_frame = []

        # Model names
        s2_method = s2cfg.get("approach", "")
        self.stage2_model_path = s2cfg.get(s2_method, {}).get("model_path", "")
        self.stage2_model_name = os.path.basename(self.stage2_model_path) if self.stage2_model_path else "N/A"
        self.stage3_model_name = os.path.basename(self.stage3_model_path) if self.stage3_model_path else "N/A"

    def _strip_thop_buffers(self, module):
        """Remove THOP's leftover buffers from a module tree."""
        for mod in module.modules():
            for name in ("total_ops", "total_params"):
                if hasattr(mod, name):
                    try:
                        delattr(mod, name)
                    except Exception:
                        pass
                if hasattr(mod, "_buffers") and isinstance(mod._buffers, dict):
                    mod._buffers.pop(name, None)
        return module

    def _stage2_module_handle(self):
        """Return ('ultralytics', YOLO_instance) or ('torch_module', nn.Module) for Stage-2."""
        det = getattr(self.person_stage, "detector", None)
        if det is None:
            return None, None

        yolo_obj = getattr(det, "model", None)
        if isinstance(yolo_obj, YOLO):
            return "ultralytics", yolo_obj

        for owner in (det, getattr(det, "predictor", None)):
            if owner is None:
                continue
            for attr in ("net", "model", "_net", "_model"):
                mod = getattr(owner, attr, None)
                if isinstance(mod, torch.nn.Module):
                    return "torch_module", mod
                if hasattr(mod, "forward"):
                    return "torch_module", mod
        return None, None

    def _compute_flops_gflops(self, model_or_yolo, imgsz=640, device=0):
        """Compute GFLOPs for models."""
        if isinstance(model_or_yolo, YOLO):
            try:
                from ultralytics.utils.torch_utils import get_flops_with_torch_profiler
                return get_flops_with_torch_profiler(model_or_yolo.model, imgsz=imgsz)
            except Exception:
                return None
        
        try:
            from fvcore.nn import FlopCountAnalysis
            m = model_or_yolo.eval()
            dummy = torch.randn(1, 3, imgsz, imgsz)
            if isinstance(device, int) and device >= 0 and torch.cuda.is_available():
                m = m.cuda()
                dummy = dummy.cuda()
            flops = FlopCountAnalysis(m, dummy).total()
            return float(flops / 1e9)
        except Exception:
            pass
        
        try:
            from thop import profile
            m = deepcopy(model_or_yolo).eval()
            self._strip_thop_buffers(m)
            dummy = torch.randn(1, 3, imgsz, imgsz)
            if isinstance(device, int) and device >= 0 and torch.cuda.is_available():
                m = m.cuda()
                dummy = dummy.cuda()
            macs, _ = profile(m, inputs=(dummy,), verbose=False)
            return float(2.0 * macs / 1e9)
        except Exception:
            return None

    def _prepare_out_dirs(self):
        img_out = ensure_dir(os.path.join(self.dataset_root, self.dataset_split, "images"))
        lab_out = ensure_dir(os.path.join(self.dataset_root, self.dataset_split, "labels"))
        return img_out, lab_out

    def _build_cropped_testset(self):
        img_out, lab_out = self._prepare_out_dirs()
        image_paths = sorted(glob.glob(os.path.join(self.frames_dir, "*.*")))
        kept = 0

        # --- infer a video id from filename stem ---
        def _infer_video_id(stem: str) -> str:
            """
            Tries common patterns, e.g.:
            videoA_frame000123.jpg  -> videoA
            clip-07_img0042.png     -> clip-07
            scene42_000015.jpeg     -> scene42
            Falls back to the first token before '_' or '-'.
            """
            m = re.match(r'^(.+?)_(?:frame|img|f)\d+', stem, flags=re.IGNORECASE)
            if m:
                return m.group(1)
            m = re.match(r'^(.+?)[_-]\d{3,}$', stem)
            if m:
                return m.group(1)
            return re.split(r'[_-]', stem, maxsplit=1)[0] or stem

        # --- group frames by video id ---
        video_groups = defaultdict(list)
        for ip in image_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            vid = _infer_video_id(stem)
            video_groups[vid].append(ip)

        # Optional: clear any previous experiment state
        if hasattr(self.person_stage, 'reset'):
            self.person_stage.reset()
        if hasattr(self.person_stage, 'track_history'):
            self.person_stage.track_history.clear()

        # --- iterate video groups; reset tracker per video ---
        for video_id, video_frames in sorted(video_groups.items()):
            print(f"[INFO] Processing video group: {video_id} ({len(video_frames)} frames)")

            # Hard reset so IDs/state never bleed across videos
            if hasattr(self.person_stage, 'reset'):
                self.person_stage.reset()
            if hasattr(self.person_stage, 'track_history'):
                self.person_stage.track_history.clear()
            # Also reset detector counters if present
            if hasattr(self.person_stage, 'detector'):
                det = self.person_stage.detector
                for attr in ('frame_count', 'frame_idx'):
                    if hasattr(det, attr):
                        setattr(det, attr, 0)

            for frame_idx, ip in enumerate(sorted(video_frames)):  # local index restarts at 0 per video
                img = cv2.imread(ip)
                if img is None:
                    continue
                h, w = img.shape[:2]
                stem = os.path.splitext(os.path.basename(ip))[0]

                # Load full-frame GT for this frame (keeps your allowed_cls filter)
                gt_path = os.path.join(self.labels_dir, stem + ".txt")
                full_gt = load_yolo_labels(gt_path)
                full_gt = [b for b in full_gt if int(b[0]) in self.allowed_cls]

                # Prepare GT objects (for coverage)
                gt_objs = []
                for cls, cx, cy, gw, gh in full_gt:
                    ax = cx * w; ay = cy * h
                    aw = gw * w; ah = gh * h
                    bx1 = ax - aw/2; by1 = ay - ah/2
                    bx2 = ax + aw/2; by2 = ay + ah/2
                    gt_objs.append({'cls': int(cls), 'bbox': (bx1, by1, bx2, by2), 'covered': False})
                for g in gt_objs:
                    self.gt_total_by_cls[g['cls']] += 1

                # Stage-2: detect every frame_gap, track between (metrics bookkeeping only—
                # the actual switch happens inside PersonDetectionStage.run based on frame_idx)
                t0 = time.perf_counter()
                run_detection = (frame_idx % self.frame_gap == 0) or not self.use_tracker
                _, persons, _ = self.person_stage.run(img.copy(), frame_idx)

                # --- Stage-3 Weapon Detection (only if persons exist) ---
                img_for_privacy = img.copy()
                if persons:
                    img_for_privacy, weapon_stats = self.weapon_stage.run(img_for_privacy, persons)
                    weapon_ids = set(weapon_stats.get("weapon_person_ids", []))
                else:
                    weapon_stats = {"weapon_person_ids": []}
                    weapon_ids = set()

                # --- Privacy anonymization ---
                if self.cfg.get("privacy", {}).get("enabled", False):
                    from utils.privacy import blur_faces_non_targets, silhouette_non_targets
                    img_for_privacy = blur_faces_non_targets(
                        img_for_privacy, persons, weapon_ids, self.cfg["privacy"]
                    )
                    img_for_privacy = silhouette_non_targets(
                        img_for_privacy, persons, weapon_ids, self.cfg["privacy"]
                    )
                    # --- Save anonymized preview every 10 frames ---
                    if frame_idx % 10 == 0:
                        preview_dir = ensure_dir(os.path.join(self.base_output_dir, f"privacy_previews_{self.experiment_name}"))
                        cv2.imwrite(os.path.join(preview_dir, f"{video_id}_{frame_idx:06d}.jpg"), img_for_privacy)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.stage2_times_ms.append(elapsed_ms)
                if run_detection:
                    self.detection_times_ms.append(elapsed_ms)
                    self.detection_count += 1
                elif self.use_tracker:
                    self.tracking_times_ms.append(elapsed_ms)
                    self.tracking_count += 1

                # Cropping + label remap (unchanged)
                frame_crop_count = 0
                if persons:
                    for p_i, person in enumerate(persons):
                        bbox = person.get("bbox")
                        if not bbox or len(bbox) < 4:
                            continue
                        x1, y1, x2, y2 = bbox
                        cx1, cy1, cx2, cy2 = square_scale_clip_xyxy(x1, y1, x2, y2, w, h, self.crop_scale)
                        ix1, iy1, ix2, iy2 = map(int, [cx1, cy1, cx2, cy2])
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue

                        crop = img_for_privacy[iy1:iy2, ix1:ix2].copy()
                        if crop.size == 0:
                            continue
                        frame_crop_count += 1

                        # coverage mark
                        for g in gt_objs:
                            if g['covered']:
                                continue
                            gx1, gy1, gx2, gy2 = g['bbox']
                            ix1c = max(gx1, cx1); iy1c = max(gy1, cy1)
                            ix2c = min(gx2, cx2); iy2c = min(gy2, cy2)
                            if ix2c > ix1c and iy2c > iy1c:
                                g['covered'] = True

                        remapped = remap_frame_labels_to_crop(
                            full_gt, w, h, [cx1, cy1, cx2, cy2], allowed_cls=self.allowed_cls
                        )
                        out_stem = f"{stem}_p{p_i:02d}"
                        cv2.imwrite(os.path.join(img_out, out_stem + ".jpg"), crop)
                        write_yolo_labels(os.path.join(lab_out, out_stem + ".txt"), remapped)
                        kept += 1

                self.crops_per_frame.append(frame_crop_count)
                for g in gt_objs:
                    if g['covered']:
                        self.gt_covered_by_s2_by_cls[g['cls']] += 1

        yaml_path = make_dataset_yaml(self.dataset_root, self.class_names)
        return yaml_path, kept


    def _eval_stage3_ultralytics(self, dataset_yaml):
        if not self.stage3_model_path or not os.path.exists(self.stage3_model_path):
            raise FileNotFoundError("Stage 3 model_path missing or not found in config.")
        
        model = YOLO(self.stage3_model_path)
        
        try:
            from ultralytics.utils.torch_utils import get_flops_with_torch_profiler
            stage3_gflops = get_flops_with_torch_profiler(model.model, imgsz=self.stage3_flops_imgsz)
        except Exception:
            stage3_gflops = None
        
        # Run validation with explicit output settings
        val_output_dir = os.path.join(self.base_output_dir, f"val_{self.experiment_name}")
        
        metrics = model.val(
            data=dataset_yaml,
            split=self.eval_split,
            imgsz=640,
            device=0,
            conf=0.25,
            project=self.base_output_dir,
            name=f"val_{self.experiment_name}",
            exist_ok=True,  # Allow overwriting if folder exists
            save_json=True,  # Save results in JSON format
            save_txt=True,   # Save detection results
            save_conf=True,  # Save confidences
            plots=True,      # Generate plots
            verbose=False    # Reduce output verbosity
        )
        
        # Ensure validation outputs are saved
        print(f"[VAL] Validation outputs saved to: {val_output_dir}")
        
        return metrics, stage3_gflops

    def run(self):
        """Run single experiment and return results dict."""
        print(f"\n{'='*60}")
        print(f"Running experiment: {self.experiment_name}")
        print(f"Tracker: {self.use_tracker}, Frame Gap: {self.frame_gap}")
        print('='*60)
        
        # Get Stage-2 FLOPs
        kind, stage2_handle = self._stage2_module_handle()
        stage2_gflops = None
        if stage2_handle is not None:
            stage2_gflops = self._compute_flops_gflops(stage2_handle, imgsz=self.stage2_flops_imgsz, device=0)
        
        # Build cropped dataset
        dataset_yaml, kept = self._build_cropped_testset()
        print(f"[CROPS] Generated {kept} crops")
        
        # Check if crops were actually generated
        if kept == 0:
            print(f"[WARN] No crops generated for {self.experiment_name}, skipping validation")
            return {
                'tracker': self.use_tracker,
                'frame_gap': self.frame_gap,
                'coverage': 0.0,
                'pipeline_map50': 0.0,
                'pipeline_recall': 0.0,
                'pipeline_f1': 0.0,
                'stage3_map50': 0.0,
                'stage3_recall': 0.0,
                'stage3_precision': 0.0,
                'stage2_latency_ms': 0.0,
                'stage3_latency_ms': 0.0,
                'latency_ms': 0.0,
                'fps': 0.0,
                'gflops': 0.0,
                'crops_generated': 0,
                'avg_crops_per_frame': 0.0
            }
        
        # Calculate timing stats
        stage2_avg_ms = sum(self.stage2_times_ms) / max(len(self.stage2_times_ms), 1) if self.stage2_times_ms else 0.0
        avg_crops = sum(self.crops_per_frame) / max(len(self.crops_per_frame), 1) if self.crops_per_frame else 0.0
        
        print(f"[INFO] Average crops per frame: {avg_crops:.2f}")
        print(f"[INFO] Stage-2 avg latency: {stage2_avg_ms:.2f} ms")
        
        # Run Stage-3 evaluation
        results = {}
        if self.enable_eval and kept > 0:
            print(f"[EVAL] Running Stage-3 validation for {self.experiment_name}")
            metrics, stage3_gflops = self._eval_stage3_ultralytics(dataset_yaml)
            
            # Coverage
            gt_total = sum(self.gt_total_by_cls.values())
            gt_covered = sum(self.gt_covered_by_s2_by_cls.values())
            coverage = (gt_covered / gt_total) if gt_total > 0 else 0.0
            
            # Stage-3 metrics
            mp = getattr(metrics.box, "mp", 0.0)
            mr = getattr(metrics.box, "mr", 0.0)
            map50 = getattr(metrics.box, "map50", 0.0)
            
            # Pipeline metrics
            pipeline_recall = coverage * mr
            pipeline_map50 = coverage * map50
            pipeline_f1 = (2 * mp * pipeline_recall / (mp + pipeline_recall)) if (mp + pipeline_recall) > 0 else 0.0
            
            # Latency
            spd = getattr(metrics, "speed", {}) or {}
            stage3_inf_ms = float(spd.get("inference", 0.0))
            pipeline_latency_ms = stage2_avg_ms + (avg_crops * stage3_inf_ms)
            
            # Total FLOPs
            stage3_total_gflops = (stage3_gflops or 0.0) * avg_crops if stage3_gflops else 0.0
            total_gflops = (stage2_gflops or 0.0) + stage3_total_gflops
            
            results = {
                'tracker': self.use_tracker,
                'frame_gap': self.frame_gap,
                'coverage': coverage,
                'pipeline_map50': pipeline_map50,
                'pipeline_recall': pipeline_recall,
                'pipeline_f1': pipeline_f1,
                'stage3_map50': map50,
                'stage3_recall': mr,
                'stage3_precision': mp,
                'stage2_latency_ms': stage2_avg_ms,
                'stage3_latency_ms': stage3_inf_ms,
                'latency_ms': pipeline_latency_ms,
                'fps': 1000.0 / pipeline_latency_ms if pipeline_latency_ms > 0 else 0,
                'gflops': total_gflops,
                'crops_generated': kept,
                'avg_crops_per_frame': avg_crops
            }
            
            print(f"[RESULT] Pipeline mAP50: {pipeline_map50:.4f}")
            print(f"[RESULT] Pipeline Latency: {pipeline_latency_ms:.2f} ms")
            print(f"[RESULT] FPS: {results['fps']:.1f}")
        else:
            # Return empty results if no evaluation
            results = {
                'tracker': self.use_tracker,
                'frame_gap': self.frame_gap,
                'coverage': 0.0,
                'pipeline_map50': 0.0,
                'pipeline_recall': 0.0,
                'pipeline_f1': 0.0,
                'stage3_map50': 0.0,
                'stage3_recall': 0.0,
                'stage3_precision': 0.0,
                'stage2_latency_ms': stage2_avg_ms,
                'stage3_latency_ms': 0.0,
                'latency_ms': stage2_avg_ms,
                'fps': 1000.0 / stage2_avg_ms if stage2_avg_ms > 0 else 0,
                'gflops': stage2_gflops or 0.0,
                'crops_generated': kept,
                'avg_crops_per_frame': avg_crops
            }
        
        return results


# ---------- Multi-experiment orchestrator ----------
class MultiExperimentPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.tz = pytz.timezone("US/Pacific")
        self.stamp = datetime.now(self.tz).strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("processed_videos", f"{self.stamp}_multi_experiments")
        ensure_dir(self.output_dir)
        
        # Define experiments to run
        self.experiments = [
            (False, 1, "no_tracker"),  # No tracker
        ]
        # Add tracker experiments with frame gaps 1-10
        for gap in range(1, 11):
            self.experiments.append((True, gap, f"tracker_gap{gap}"))
    
    def run_all_experiments(self):
        """Run all experiments and collect results."""
        all_results = []
        
        for use_tracker, frame_gap, exp_name in self.experiments:
            exp = SingleExperiment(
                self.config_path, 
                use_tracker, 
                frame_gap, 
                exp_name,
                self.output_dir
            )
            results = exp.run()
            all_results.append(results)
        
        return all_results
    
    def write_combined_reports(self, results):
        """Write combined report.txt and table_combined.txt."""
        report_path = os.path.join(self.output_dir, "report.txt")
        table_path = os.path.join(self.output_dir, "table_combined.txt")
        
        # Get model names from first result (same for all experiments)
        if results:
            first_exp = SingleExperiment(self.config_path, False, 1, "temp", self.output_dir)
            stage2_full = first_exp.stage2_model_name.replace('.pt', '').replace('.pth', '').replace('.onnx', '')
            stage3_full = first_exp.stage3_model_name.replace('.pt', '').replace('.pth', '').replace('.onnx', '')
            
            # Get part before first underscore
            stage2_short = stage2_full.split('_')[0] if '_' in stage2_full else stage2_full
            stage3_short = stage3_full.split('_')[0] if '_' in stage3_full else stage3_full
            model_name = f"{stage2_short}_{stage3_short}"
        else:
            model_name = "Unknown"
        
        # Write detailed report with stage metrics
        with open(report_path, "w") as f:
            f.write("=== MULTI-CONFIGURATION PIPELINE EVALUATION ===\n")
            f.write(f"Timestamp: {self.stamp}\n")
            f.write(f"Pipeline Model: {model_name}\n")
            f.write(f"Total experiments: {len(results)}\n\n")
            
            for i, r in enumerate(results):
                f.write(f"\n{'='*60}\n")
                f.write(f"--- Experiment {i+1} ---\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Tracker: {'Yes' if r['tracker'] else 'No'}\n")
                f.write(f"Frame Gap: {r['frame_gap']}\n")
                f.write(f"Coverage: {r['coverage']:.4f}\n")
                f.write(f"Crops: {r['crops_generated']}\n")
                f.write(f"Avg crops/frame: {r['avg_crops_per_frame']:.2f}\n\n")
                
                # Stage-2 Metrics
                f.write("Stage-2 (Person Detection):\n")
                f.write(f"  Coverage: {r['coverage']:.4f}\n")
                f.write(f"  Latency: {r.get('stage2_latency_ms', 0):.2f} ms\n\n")
                
                # Stage-3 Metrics
                f.write("Stage-3 (Weapon Detection on Crops):\n")
                f.write(f"  mAP50: {r.get('stage3_map50', 0):.4f}\n")
                f.write(f"  Recall: {r.get('stage3_recall', 0):.4f}\n")
                f.write(f"  Precision: {r.get('stage3_precision', 0):.4f}\n")
                f.write(f"  Latency: {r.get('stage3_latency_ms', 0):.2f} ms/crop\n\n")
                
                # Pipeline Metrics
                f.write("Pipeline (End-to-End):\n")
                f.write(f"  mAP50: {r['pipeline_map50']:.4f}\n")
                f.write(f"  Recall: {r['pipeline_recall']:.4f}\n")
                f.write(f"  F1: {r['pipeline_f1']:.4f}\n")
                f.write(f"  Latency: {r['latency_ms']:.2f} ms\n")
                f.write(f"  FPS: {r['fps']:.1f}\n")
                f.write(f"  GFLOPs: {r['gflops']:.2f}\n")
        
        # Write properly formatted table
        with open(table_path, "w") as f:
            # Define column widths for proper alignment
            headers = ["Model", "Tracker", "Frame_Gap", "mAP50", "Recall", "F1", "Latency_ms", "FPS", "GFLOPs"]
            
            # Calculate column widths based on content
            col_widths = []
            for i, h in enumerate(headers):
                max_width = len(h)
                for r in results:
                    if i == 0:  # Model
                        max_width = max(max_width, len(model_name))
                    elif i == 1:  # Tracker
                        max_width = max(max_width, 3)  # "Yes" or "No"
                    elif i == 2:  # Frame_Gap
                        max_width = max(max_width, len(str(r['frame_gap'])))
                    elif i == 3:  # mAP50
                        max_width = max(max_width, 6)  # "0.xxxx"
                    elif i == 4:  # Recall
                        max_width = max(max_width, 6)
                    elif i == 5:  # F1
                        max_width = max(max_width, 6)
                    elif i == 6:  # Latency_ms
                        max_width = max(max_width, len(f"{r['latency_ms']:.2f}"))
                    elif i == 7:  # FPS
                        max_width = max(max_width, len(f"{r['fps']:.1f}"))
                    elif i == 8:  # GFLOPs
                        max_width = max(max_width, len(f"{r['gflops']:.2f}"))
                col_widths.append(max_width + 2)  # Add padding
            
            # Write header
            header_line = ""
            for i, h in enumerate(headers):
                header_line += h.ljust(col_widths[i])
            f.write(header_line.rstrip() + "\n")
            
            # Write separator
            f.write("-" * sum(col_widths) + "\n")
            
            # Write data rows
            for r in results:
                row_data = [
                    model_name,
                    "Yes" if r['tracker'] else "No",
                    str(r['frame_gap']),
                    f"{r['pipeline_map50']:.4f}",
                    f"{r['pipeline_recall']:.4f}",
                    f"{r['pipeline_f1']:.4f}",
                    f"{r['latency_ms']:.2f}",
                    f"{r['fps']:.1f}",
                    f"{r['gflops']:.2f}"
                ]
                
                row_line = ""
                for i, val in enumerate(row_data):
                    row_line += val.ljust(col_widths[i])
                f.write(row_line.rstrip() + "\n")
        
        print(f"\n[LOG] Reports written to:")
        print(f"  - {report_path}")
        print(f"  - {table_path}")
    
    def plot_results(self, results):
        """Generate plots for mAP vs Speed vs Frame Gap."""
        plot_dir = os.path.join(self.output_dir, "plots")
        ensure_dir(plot_dir)
        
        # Extract data for plotting
        tracker_results = [r for r in results if r['tracker']]
        no_tracker = [r for r in results if not r['tracker']][0] if results else None
        
        if tracker_results:
            frame_gaps = [r['frame_gap'] for r in tracker_results]
            maps = [r['pipeline_map50'] for r in tracker_results]
            fps_values = [r['fps'] for r in tracker_results]
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: mAP50 vs Frame Gap
            ax1 = axes[0, 0]
            ax1.plot(frame_gaps, maps, 'b.-', markersize=8)
            if no_tracker:
                ax1.axhline(y=no_tracker['pipeline_map50'], color='r', linestyle='--', label='No Tracker')
            ax1.set_xlabel('Frame Gap')
            ax1.set_ylabel('Pipeline mAP50')
            ax1.set_title('mAP50 vs Frame Gap')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: FPS vs Frame Gap
            ax2 = axes[0, 1]
            ax2.plot(frame_gaps, fps_values, 'g.-', markersize=8)
            if no_tracker:
                ax2.axhline(y=no_tracker['fps'], color='r', linestyle='--', label='No Tracker')
            ax2.set_xlabel('Frame Gap')
            ax2.set_ylabel('FPS')
            ax2.set_title('Speed (FPS) vs Frame Gap')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: mAP50 vs FPS (Speed-Accuracy Tradeoff)
            ax3 = axes[1, 0]
            ax3.plot(fps_values, maps, 'mo-', markersize=8)
            for i, gap in enumerate(frame_gaps):
                ax3.annotate(f'gap={gap}', (fps_values[i], maps[i]), 
                           textcoords="offset points", xytext=(5,5), fontsize=8)
            if no_tracker:
                ax3.plot(no_tracker['fps'], no_tracker['pipeline_map50'], 
                        'r*', markersize=15, label='No Tracker')
            ax3.set_xlabel('FPS')
            ax3.set_ylabel('Pipeline mAP50')
            ax3.set_title('Speed-Accuracy Tradeoff')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Plot 4: Combined metrics
            ax4 = axes[1, 1]
            ax4_2 = ax4.twinx()
            
            l1 = ax4.plot(frame_gaps, maps, 'b.-', label='mAP50', markersize=8)
            l2 = ax4_2.plot(frame_gaps, fps_values, 'g.-', label='FPS', markersize=8)
            
            ax4.set_xlabel('Frame Gap')
            ax4.set_ylabel('mAP50', color='b')
            ax4_2.set_ylabel('FPS', color='g')
            ax4.tick_params(axis='y', labelcolor='b')
            ax4_2.tick_params(axis='y', labelcolor='g')
            ax4.set_title('mAP50 and FPS vs Frame Gap')
            ax4.grid(True, alpha=0.3)
            
            # Combined legend
            lns = l1 + l2
            labs = [l.get_label() for l in lns]
            ax4.legend(lns, labs, loc='center right')
            
            plt.suptitle('Pipeline Performance Analysis with Tracking', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(plot_dir, 'performance_analysis.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[PLOT] Saved performance analysis to: {plot_path}")
            
            # Additional plot: 3D surface for mAP, FPS, and Frame Gap
            if len(tracker_results) > 3:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Create scatter plot
                scatter = ax.scatter(frame_gaps, fps_values, maps, 
                                   c=maps, cmap='coolwarm', s=100, alpha=0.8)
                
                # Add no-tracker point if available
                if no_tracker:
                    ax.scatter([0], [no_tracker['fps']], [no_tracker['pipeline_map50']], 
                             color='red', s=200, marker='*', label='No Tracker')
                
                ax.set_xlabel('Frame Gap')
                ax.set_ylabel('FPS')
                ax.set_zlabel('mAP50')
                ax.set_title('3D Performance Space')
                plt.colorbar(scatter, label='mAP50')
                
                # Save 3D plot
                plot3d_path = os.path.join(plot_dir, 'performance_3d.png')
                plt.savefig(plot3d_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"[PLOT] Saved 3D analysis to: {plot3d_path}")
    
    def run(self):
        """Main execution function."""
        print(f"Starting multi-configuration pipeline evaluation")
        print(f"Output directory: {self.output_dir}")
        print(f"Experiments to run: {len(self.experiments)}")
        
        # Run all experiments
        results = self.run_all_experiments()
        
        # Write reports
        self.write_combined_reports(results)
        
        # Generate plots
        self.plot_results(results)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print("\nSummary of Results:")
        print("-"*40)
        
        # Find best configurations
        best_map = max(results, key=lambda x: x['pipeline_map50'])
        best_fps = max(results, key=lambda x: x['fps'])
        
        print(f"\nBest mAP50: {best_map['pipeline_map50']:.4f}")
        print(f"  - Tracker: {best_map['tracker']}, Frame Gap: {best_map['frame_gap']}")
        
        print(f"\nBest FPS: {best_fps['fps']:.1f}")
        print(f"  - Tracker: {best_fps['tracker']}, Frame Gap: {best_fps['frame_gap']}")
        
        # Find optimal balance (using F1 score of normalized mAP and FPS)
        maps = [r['pipeline_map50'] for r in results]
        fps_vals = [r['fps'] for r in results]
        
        # Normalize
        max_map = max(maps)
        max_fps = max(fps_vals)
        
        best_balance = None
        best_balance_score = 0
        
        for r in results:
            norm_map = r['pipeline_map50'] / max_map if max_map > 0 else 0
            norm_fps = r['fps'] / max_fps if max_fps > 0 else 0
            balance_score = 2 * norm_map * norm_fps / (norm_map + norm_fps) if (norm_map + norm_fps) > 0 else 0
            
            if balance_score > best_balance_score:
                best_balance_score = balance_score
                best_balance = r
        
        if best_balance:
            print(f"\nBest Balance (mAP-FPS tradeoff):")
            print(f"  - Tracker: {best_balance['tracker']}, Frame Gap: {best_balance['frame_gap']}")
            print(f"  - mAP50: {best_balance['pipeline_map50']:.4f}, FPS: {best_balance['fps']:.1f}")
        
        print(f"\nAll outputs saved to: {self.output_dir}")


if __name__ == "__main__":
    # Run multi-experiment pipeline
    pipeline = MultiExperimentPipeline("config.yaml")
    pipeline.run()