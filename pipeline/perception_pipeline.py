class PerceptionPipeline:
    def __init__(self, pose_model, visualizer):
        self.pose_model = pose_model
        self.visualizer = visualizer

    def process(self, frame):
        people = self.pose_model.infer(frame)

        for person in people:
            self.visualizer.draw(frame, person)

        return frame, people
