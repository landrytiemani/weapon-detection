import os
import time
import pytz
import yaml
import cv2
import json
import numpy as np
from datetime import datetime
from stages.stage_2_persondetection import PersonDetectionStage
from stages.stage_3_weapondetection import WeaponDetectionStage
from weapon_detection_pipeline.evaluation import evaluate_predictions

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]

def _get_model_name(stage_cfg):
    approach = stage_cfg.get("approach", "N_A")
    method_cfg = stage_cfg.get(approach, {})
    model_path = method_cfg.get("model_path", "N_A")
    return os.path.basename(model_path).replace('.pt', '').replace('.pth', '')

class Pipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.person_detection_stage = PersonDetectionStage(self.config['stage_2'])
        self.weapon_detection_stage = WeaponDetectionStage(self.config['stage_3'])
        self.enable_eval = self.config.get('evaluation', {}).get('enabled', False)
        self.gt_path = self.config.get('evaluation', {}).get('ground_truth_path', "")
        self.predictions = {}
        self.log_writer = None
        self.log_data = []
        self.unique_person_ids = set()

        self.pacific = pytz.timezone('US/Pacific')
        self.timestamp_str = datetime.now(self.pacific).strftime('%Y%m%d_%H%M%S')
        self.video_source = self.config['pipeline']['video_source']

    def _setup_writers(self, cap):
        save_video = self.config['pipeline'].get('save_video', False)
        if not save_video:
            return None

        output_dir = self.config['pipeline'].get('output_dir', 'processed_videos')
        os.makedirs(output_dir, exist_ok=True)

        stage2_model = _get_model_name(self.config['stage_2'])
        stage3_model = _get_model_name(self.config['stage_3'])
        base_filename = f"{self.timestamp_str}_PST_{stage2_model}_{stage3_model}"

        video_path = os.path.join(output_dir, f"{base_filename}.mp4")
        log_path = os.path.join(output_dir, f"{base_filename}.txt")

        self.log_writer = open(log_path, 'w')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (640, 640))

        print(f"> Saving processed video to: {video_path}")
        print(f"> Saving log file to: {log_path}")
        return video_writer

    def _write_log_header(self):
        if not self.log_writer:
            return

        self.log_writer.write("--- Weapon Detection Pipeline Log ---\n")
        self.log_writer.write(f"Timestamp: {datetime.now(self.pacific).strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_writer.write(f"Video Source: {self.video_source}\n\n")

        p_stats = self.person_detection_stage.model_stats
        self.log_writer.write("--- Stage 2: Person Detection Model ---\n")
        self.log_writer.write(f"Model: {p_stats.get('Model Name', 'N/A')}\n")
        self.log_writer.write(f"Parameters: {p_stats.get('Total Params', 'N/A')}\n")
        self.log_writer.write(f"FLOPs: {p_stats.get('GFLOPs', 'N/A')}\n\n")

        w_stats = self.weapon_detection_stage.model_stats
        self.log_writer.write("--- Stage 3: Weapon Detection Model ---\n")
        self.log_writer.write(f"Model: {w_stats.get('Model Name', 'N/A')}\n")
        self.log_writer.write(f"Parameters: {w_stats.get('Total Params', 'N/A')}\n")
        self.log_writer.write(f"FLOPs: {w_stats.get('GFLOPs', 'N/A')}\n\n")

        self.log_writer.write("             *--- Frame Data ---*\n")
        self.log_writer.write("Frame,      FPS,       Persons,    Avg_Person_Conf,   Weapons,    Avg_Weapon_Conf\n")
        self.log_writer.flush()

    def _write_log_summary(self):
        if not self.log_writer or not self.log_data:
            return

        log_array = np.array(self.log_data)
        avg_fps = np.mean(log_array[:, 1])
        total_persons = len(self.unique_person_ids)
        weapon_matched_persons = getattr(self.weapon_detection_stage, 'persons_with_weapons', set())
        num_persons_with_weapon = len(weapon_matched_persons)

        person_conf_values = log_array[log_array[:, 2] > 0, 3]
        avg_person_conf = np.mean(person_conf_values) if len(person_conf_values) > 0 else 0.0
        weapon_conf_values = log_array[log_array[:, 4] > 0, 5]
        avg_weapon_conf = np.mean(weapon_conf_values) if len(weapon_conf_values) > 0 else 0.0

        self.log_writer.write("\n--- Data Summary ---\n")
        self.log_writer.write(f"Total Frames Processed: {len(self.log_data)}\n")
        self.log_writer.write(f"Average FPS: {avg_fps:.2f}\n")
        self.log_writer.write(f"Total Unique Persons Detected: {total_persons}\n")
        self.log_writer.write(f"Number of Persons Detected With Weapon: {num_persons_with_weapon}\n")
        self.log_writer.write(f"Overall Avg Person Confidence: {avg_person_conf:.4f}\n")
        self.log_writer.write(f"Overall Avg Weapon Confidence: {avg_weapon_conf:.4f}\n")

        stage2_cfg = self.config['stage_2']
        detector_cfg = stage2_cfg.get(stage2_cfg.get("approach", ""), {})
        use_tracker = detector_cfg.get("use_tracker", False)
        self.log_writer.write(f"use_tracker: {use_tracker}\n")
        if use_tracker:
            frame_gap = stage2_cfg.get("frame_gap", 1)
            self.log_writer.write(f"frame_gap: {frame_gap}\n")

        if self.enable_eval:
            self.log_writer.write("\n--- Evaluation Summary ---\n")
            eval_result = evaluate_predictions(
                self.predictions,
                self.gt_path,
                original_size=(self.orig_w, self.orig_h),
                target_size=(self.proc_w, self.proc_h),
            )
            self.log_writer.write(
                f"Total Frames Evaluated: {eval_result['total']}\n"
                f"TP: {eval_result['tp']}, FP: {eval_result['fp']}, FN: {eval_result['fn']}\n"
                f"Precision: {eval_result['precision']:.4f}, Recall: {eval_result['recall']:.4f}, "
                f"F1 Score: {eval_result['f1']:.4f}\n"
            )
            if 'mAP@0.5' in eval_result['mAPs']:
                self.log_writer.write(f"mAP@0.5: {eval_result['mAPs']['mAP@0.5']:.4f}\n")
            if 'mAP@0.5:0.9' in eval_result['mAPs']:
                self.log_writer.write(f"mAP@0.5:0.9: {eval_result['mAPs']['mAP@0.5:0.9']:.4f}\n")

        self.log_writer.flush()

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video source")
        
        # real/original video dimensions (testing data)
        self.orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # model/process dimensions
        self.proc_w, self.proc_h = 640, 640

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        video_writer = self._setup_writers(cap)
        self._write_log_header()

        frame_count = 0
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame.shape[:2] != (self.proc_h, self.proc_w):
                frame = cv2.resize(frame, (self.proc_w, self.proc_h))

            frame, person_data, person_stats = self.person_detection_stage.run(frame, frame_count)
            frame, weapon_stats = self.weapon_detection_stage.run(frame, person_data)

            for person in person_data:
                self.unique_person_ids.add(person['id'])

            fps = 1 / (time.time() - start_time)
            self.log_data.append([
                frame_count, fps, person_stats['count'], person_stats['avg_confidence'],
                weapon_stats['count'], weapon_stats['avg_confidence']
            ])

            if self.log_writer:
                self.log_writer.write(
                    f"{frame_count:<10}{fps:<10.2f}{person_stats['count']:<10}{person_stats['avg_confidence']:<17.4f}"
                    f"{weapon_stats['count']:<10}{weapon_stats['avg_confidence']:<17.4f}\n"
                )

            if self.enable_eval:
                if not hasattr(self, '_gt_data'):
                    with open(self.gt_path, 'r') as f:
                        self._gt_data = json.load(f)

                raw_boxes = weapon_stats.get("bboxes", [])
                converted = [xyxy_to_xywh(box) for box in raw_boxes]
                self.predictions[frame_count] = converted

                gt_boxes = [
                    ann['bbox'] for ann in self._gt_data['annotations']
                    if ann['image_id'] == frame_count
                ]

                # Draw GT (green)
                sx = self.proc_w / max(1, self.orig_w)
                sy = self.proc_h / max(1, self.orig_h)

                for x, y, w, h in gt_boxes:  # GT in original video coords
                    x1 = int(x * sx)
                    y1 = int(y * sy)
                    x2 = int((x + w) * sx)
                    y2 = int((y + h) * sy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw predictions (red)
                for x1, y1, x2, y2 in raw_boxes:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            if video_writer:
                video_writer.write(frame)

        cap.release()
        if video_writer:
            video_writer.release()

        self._write_log_summary()
        if self.log_writer:
            self.log_writer.close()

if __name__ == "__main__":
    config_path = "config.yaml"
    pipeline = Pipeline(config_path)
    pipeline.run()
