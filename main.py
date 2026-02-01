import cv2
from core.camera import Camera
from core.yolo_pose import YoloPose
from core.visualizer import Visualizer
from pipeline.perception_pipeline import PerceptionPipeline

camera = Camera(0)
pose_model = YoloPose()
visualizer = Visualizer()

pipeline = PerceptionPipeline(pose_model, visualizer)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    output, people = pipeline.process(frame)
    cv2.imshow("YOLOv8 Pose Pipeline", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
