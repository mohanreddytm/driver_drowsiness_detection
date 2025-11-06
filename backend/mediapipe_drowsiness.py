"""
Driver Drowsiness Detection using MediaPipe Face Mesh and OpenCV

This script replaces Haar/Dlib-based detection with MediaPipe Face Mesh for
robust single-face detection and precise eye landmark extraction. It computes
Eye Aspect Ratio (EAR) from face mesh landmarks, smooths EAR values with a
moving average, and triggers a "Drowsy" state when eyes remain closed past
a configurable threshold.

Compatibility: Python 3.10+, OpenCV, MediaPipe, NumPy

Install prerequisites:
    pip install opencv-python mediapipe numpy

Run:
    python backend/mediapipe_drowsiness.py

"""

from collections import deque
import time
import math
import sys

try:
    import cv2
    import numpy as np
    import mediapipe as mp
except Exception as e:
    print("Missing dependencies or failed to import:", e)
    print("Install with: pip install opencv-python mediapipe numpy")
    raise


# ----------------------------- Configuration -----------------------------
# Frame and preprocessing
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
USE_CLAHE = True
CLAHE_CLIP = 3.0
CLAHE_TILE = (8, 8)

# MediaPipe FaceMesh parameters
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5
MP_MAX_FACES = 1  # Ensure we only process the primary face

# EAR thresholds and smoothing
EAR_THRESHOLD = 0.20         # below this EAR considered "closed"
CONSECUTIVE_FRAMES = 3      # number of consecutive frames EAR must be below threshold
SMOOTHING_WINDOW = 5        # moving average window size (frames) for EAR smoothing

# Visual / debug
SHOW_FPS = True
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ----------------------------- Utility functions -----------------------------

def euclidean(a, b):
    """Euclidean distance between two 2D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_ear(eye_landmarks):
    """
    Compute Eye Aspect Ratio (EAR) for one eye.

    eye_landmarks: list of 6 (x, y) tuples ordered as:
      [p1, p2, p3, p4, p5, p6]
    where p1 and p4 are horizontal endpoints and others are vertical points.
    Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Returns EAR (float). If divisor is zero returns 0.0
    """
    p1, p2, p3, p4, p5, p6 = eye_landmarks
    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    C = euclidean(p1, p4)
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear


# ----------------------------- MediaPipe mapping -----------------------------
# MediaPipe FaceMesh landmark indices for left and right eyes (6-point per eye)
# These indices are commonly used to approximate EAR with FaceMesh.
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def landmarks_to_xy_list(landmarks, indices, image_w, image_h):
    """Convert normalized Mediapipe landmarks to pixel (x, y) tuples for given indices."""
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        x_px = int(min(max(lm.x * image_w, 0), image_w - 1))
        y_px = int(min(max(lm.y * image_h, 0), image_h - 1))
        pts.append((x_px, y_px))
    return pts


# ----------------------------- Detector class -----------------------------
class DrowsinessDetector:
    """Encapsulates MediaPipe FaceMesh, EAR calculation, smoothing, and alert logic."""

    def __init__(self,
                 ear_threshold=EAR_THRESHOLD,
                 consecutive_frames=CONSECUTIVE_FRAMES,
                 smoothing_window=SMOOTHING_WINDOW,
                 frame_width=FRAME_WIDTH,
                 frame_height=FRAME_HEIGHT):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.smoothing_window = smoothing_window
        self.frame_width = frame_width
        self.frame_height = frame_height

        # State
        self.ear_buffer = deque(maxlen=smoothing_window)
        self.closed_frames = 0
        self.drowsy = False

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=MP_MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        )

        # CLAHE
        if USE_CLAHE:
            self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        else:
            self.clahe = None

    def preprocess(self, frame):
        """Resize and optionally enhance contrast with CLAHE. Returns processed frame and gray image."""
        # Resize to desired size for consistent performance
        frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))

        # Apply CLAHE on the grayscale for better contrast where needed
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        if self.clahe is not None:
            gray = self.clahe.apply(gray)

        return frame_resized, gray

    def detect(self, frame):
        """Run face mesh detection and return face landmarks (if found) and bounding box."""
        # MediaPipe expects RGB images
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None, None

        # Use only the first (primary) face
        landmarks = results.multi_face_landmarks[0].landmark

        # Compute face bounding box from landmarks
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x_min = max(0, min(xs))
        x_max = min(1, max(xs))
        y_min = max(0, min(ys))
        y_max = min(1, max(ys))

        # Convert normalized bbox to pixels
        x1 = int(x_min * self.frame_width)
        y1 = int(y_min * self.frame_height)
        x2 = int(x_max * self.frame_width)
        y2 = int(y_max * self.frame_height)
        bbox = (x1, y1, x2 - x1, y2 - y1)

        return landmarks, bbox

    def process_landmarks(self, landmarks, frame):
        """Extract eyes, compute EAR, update smoothing and drowsy state. Returns annotated frame and info."""
        ih, iw = frame.shape[:2]

        # Convert relevant landmarks to pixel coordinates
        left_eye_pts = landmarks_to_xy_list(landmarks, LEFT_EYE_IDX, iw, ih)
        right_eye_pts = landmarks_to_xy_list(landmarks, RIGHT_EYE_IDX, iw, ih)

        # Compute EAR for both eyes
        left_ear = compute_ear(left_eye_pts)
        right_ear = compute_ear(right_eye_pts)
        ear = float((left_ear + right_ear) / 2.0)

        # Smooth EAR with moving average buffer
        self.ear_buffer.append(ear)
        smoothed_ear = float(sum(self.ear_buffer) / len(self.ear_buffer))

        # Update closed frames counter
        if smoothed_ear < self.ear_threshold:
            self.closed_frames += 1
        else:
            self.closed_frames = 0

        # Trigger drowsy only when eyes have been closed for required consecutive frames
        was_drowsy = self.drowsy
        self.drowsy = self.closed_frames >= self.consecutive_frames

        # Draw eye landmarks and EAR on frame for debugging
        for (x, y) in left_eye_pts:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        for (x, y) in right_eye_pts:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # Display EAR and status on the frame
        status_text = "Drowsy" if self.drowsy else "Active"
        cv2.putText(frame, f"EAR: {smoothed_ear:.3f}", (10, 30), FONT, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Status: {status_text}", (10, 60), FONT, 0.8, (0, 255, 0) if not self.drowsy else (0, 0, 255), 2)

        return frame, {
            "ear": smoothed_ear,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "drowsy": self.drowsy,
            "closed_frames": self.closed_frames,
            "was_drowsy": was_drowsy,
        }


# ----------------------------- Main loop -----------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure camera is connected and not used by another app.")
        return

    detector = DrowsinessDetector()
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: empty frame")
                time.sleep(0.05)
                continue

            # Preprocess: resize + CLAHE
            frame_resized, gray = detector.preprocess(frame)

            # Detect primary face and landmarks
            landmarks, bbox = detector.detect(frame_resized)

            if landmarks is None:
                # No face detected: reset counters and show preview
                cv2.putText(frame_resized, "No face detected", (10, 90), FONT, 0.7, (0, 255, 255), 2)
                detector.ear_buffer.clear()
                detector.closed_frames = 0
                detector.drowsy = False
            else:
                # Process landmarks and update EAR/drowsiness
                frame_resized, info = detector.process_landmarks(landmarks, frame_resized)

                # Draw bounding box for the primary face
                x, y, w, h = bbox
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # If drowsy state changed to True, display a prominent alert
                if info["drowsy"] and not info["was_drowsy"]:
                    print(f"ALERT: Drowsy detected at {time.strftime('%H:%M:%S')}")

            # FPS calculation and display
            if SHOW_FPS:
                cur_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, cur_time - prev_time)) if fps else (1.0 / max(1e-6, cur_time - prev_time))
                prev_time = cur_time
                cv2.putText(frame_resized, f"FPS: {fps:.1f}", (frame_resized.shape[1] - 120, 30), FONT, 0.6, (200, 200, 200), 2)

            # Show the preview
            cv2.imshow("Driver Drowsiness Detection (MediaPipe)", frame_resized)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
