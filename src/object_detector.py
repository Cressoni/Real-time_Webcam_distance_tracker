
---

### 3. src/object_detector.py

```python
import torch
import cv2

class ObjectDetector:
    """
    Loads a YOLOv5 model via torch.hub and runs inference.
    """
    def __init__(self, model_name='yolov5s', device='cpu'):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.to(self.device)
        self.names = self.model.names

    def detect(self, frame):
        """
        Runs object detection on a single frame (BGR numpy array).
        Returns a list of detections:
          {'box': (x1, y1, x2, y2), 'confidence': float, 'class': str}
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(img)
        dets = []
        for *box, conf, cls in results.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, box)
            label = self.names[int(cls)]
            dets.append({
                'box': (x1, y1, x2, y2),
                'confidence': conf,
                'class': label
            })
        return dets
