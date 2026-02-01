from ultralytics import YOLO

class YoloPose:
    def __init__(self, model_path="yolov8n-pose.pt"):
        self.model = YOLO(model_path)

    def infer(self, frame):
        results = self.model(frame, verbose=False)
        people = []

        for r in results:
            if r.keypoints is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            keypoints = r.keypoints.xy.cpu().numpy()  # (N, 17, 2)

            for box, kpts in zip(boxes, keypoints):
                people.append({
                    "box": box.astype(int),
                    "keypoints": kpts
                })
        return people
