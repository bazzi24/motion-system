import cv2

# COCO skeleton
SKELETON = [
    (5,7),(7,9),     # left arm
    (6,8),(8,10),    # right arm
    (5,6),           # shoulders
    (5,11),(6,12),   # torso
    (11,12),
    (11,13),(13,15), # left leg
    (12,14),(14,16)  # right leg
]

class Visualizer:
    def draw(self, frame, person):
        x1, y1, x2, y2 = person["box"]
        kpts = person["keypoints"]

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        for x, y in kpts:
            cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)

        for a, b in SKELETON:
            xa, ya = kpts[a]
            xb, yb = kpts[b]
            cv2.line(frame,
                     (int(xa),int(ya)),
                     (int(xb),int(yb)),
                     (255,0,0), 2)
