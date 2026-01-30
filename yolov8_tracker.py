import torch
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from supervision.draw.color import Color

class YOLOv8ByteTrack:
    def __init__(self, model_path: str, device: str = None, frame_rate: int = 30):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception as e:
            print(f"Warning: Could not transfer model to {device}: {e}")

        if hasattr(self.model, "names"):
            self.class_names = self.model.names
        elif hasattr(self.model, "model") and hasattr(self.model.model, "names"):
            self.class_names = self.model.model.names
        else:
            self.class_names = {}

        self.tracker = sv.ByteTrack(frame_rate=int(frame_rate))

        # Define fixed colors for each class
        self.class_colors = {
            0: (0, 0, 255),       # red - nohelmet
            1: (255, 0, 0),       # blue - car
            2: (0, 255, 0),       # green - motorcycle
            3: (0, 255, 255),     # yellow - bus
            4: (42, 42, 165),     # brown - bicycle
            5: (0, 0, 0)          # black - truck
        }

    def process_frame(self, frame: np.ndarray):
        results = self.model(frame, device=0)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        for xyxy, class_id, tracker_id in zip(detections.xyxy, detections.class_id, detections.tracker_id):
            class_id = int(class_id)
            track_id = int(tracker_id)
            x1, y1, x2, y2 = map(int, xyxy)
            class_name = self.class_names[class_id] if isinstance(self.class_names, (list, tuple)) else self.class_names.get(class_id, class_id)
            color = self.class_colors.get(class_id, (255, 255, 255))

            label = f"#{track_id} {class_name}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        counts = {}
        if isinstance(self.class_names, (list, tuple)):
            for name in self.class_names:
                counts[name] = 0
        elif isinstance(self.class_names, dict):
            for _, name in sorted(self.class_names.items()):
                counts[name] = 0

        for class_id in detections.class_id:
            class_id = int(class_id)
            class_name = self.class_names[class_id] if isinstance(self.class_names, (list, tuple)) else self.class_names.get(class_id, class_id)
            if class_name in counts:
                counts[class_name] += 1
            else:
                counts[class_name] = 1

        return frame, counts

