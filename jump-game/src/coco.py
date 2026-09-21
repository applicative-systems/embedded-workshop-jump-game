"""COCO-17 keypoint layout, as ultralytics numbers it.

Shared by the takeoff detector and both renderers (the video overlay and the
avatar), so the indices and the skeleton are defined exactly once.
"""

NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

N = 17

# Full skeleton, including the face, for the debug overlay on the video.
LIMBS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]

# The avatar does not use an edge list: it collapses the shoulder and hip
# pairs to midpoints and hangs limbs off those, so its topology is built in
# StickFigure.draw() rather than enumerated here.

LEFT_SIDE = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT_SIDE = {2, 4, 6, 8, 10, 12, 14, 16}
