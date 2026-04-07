"""
Hand Tracker — MediaPipe Tasks API wrapper
------------------------------------------
Handles landmark detection and draws a colored skeletal overlay
on the frame. BGR frames in, BGR frames out.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# ── Skeleton connection groups ────────────────────────────────────────────────
_PALM_CONNECTIONS = [(0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
_FINGER_CONNECTIONS = {
    'thumb':  [(1, 2), (2, 3), (3, 4)],
    'index':  [(5, 6), (6, 7), (7, 8)],
    'middle': [(9, 10), (10, 11), (11, 12)],
    'ring':   [(13, 14), (14, 15), (15, 16)],
    'pinky':  [(17, 18), (18, 19), (19, 20)],
}
_FINGER_COLORS = {
    'thumb':  (60,  180, 255),   # amber-orange in BGR
    'index':  (255, 220, 80),    # cyan-blue
    'middle': (80,  255, 160),   # green
    'ring':   (220, 80,  255),   # purple
    'pinky':  (80,  120, 255),   # red
}
_TIP_INDICES = {4, 8, 12, 16, 20}


class HandTracker:
    def __init__(self,
                 model_path="backend/models/hand_landmarker.task",
                 max_hands=1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe hand-landmarker model not found: {model_path}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.55,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._last_results = None

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, bgr_frame, draw=True):
        """
        Detect hand landmarks in a BGR frame.
        Returns (annotated_bgr_frame, landmarks_or_None).
        landmarks is a list of 21 [x, y, z] normalized floats.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._last_results = self.detector.detect(mp_img)

        landmarks = None
        if self._last_results and self._last_results.hand_landmarks:
            raw = self._last_results.hand_landmarks[0]
            landmarks = [[lm.x, lm.y, lm.z] for lm in raw]
            if draw:
                self._draw_skeleton(bgr_frame, landmarks)

        return bgr_frame, landmarks

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_skeleton(self, img, landmarks):
        h, w = img.shape[:2]
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]

        # Palm base
        for (a, b) in _PALM_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (180, 180, 180), 2, cv2.LINE_AA)

        # Fingers (colored per finger)
        for finger, conns in _FINGER_CONNECTIONS.items():
            color = _FINGER_COLORS[finger]
            for (a, b) in conns:
                cv2.line(img, pts[a], pts[b], color, 3, cv2.LINE_AA)

        # Joints
        for i, pt in enumerate(pts):
            if i == 0:
                # Wrist — large white dot
                cv2.circle(img, pt, 9, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, pt, 6, (140, 140, 160), -1, cv2.LINE_AA)
            elif i in _TIP_INDICES:
                # Fingertips — bright cyan glow
                cv2.circle(img, pt, 8, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, pt, 6, (120, 230, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(img, pt, 5, (200, 200, 200), -1, cv2.LINE_AA)

    # ── Legacy compatibility (used by original app.py) ────────────────────────

    def find_hands(self, img, draw=True):
        img, _ = self.process(img, draw)
        return img

    def get_flattened_landmarks(self, img, hand_no=0):
        if self._last_results and self._last_results.hand_landmarks:
            if len(self._last_results.hand_landmarks) > hand_no:
                raw = self._last_results.hand_landmarks[hand_no]
                lms = [[lm.x, lm.y, lm.z] for lm in raw]
                base = lms[0]
                return [v for lm in lms for v in
                        [lm[0] - base[0], lm[1] - base[1], lm[2] - base[2]]]
        return None
