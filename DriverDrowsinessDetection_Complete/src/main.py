import os

import cv2
import dlib
import pygame
import argparse
import time
import numpy as np
from collections import deque
from imutils import face_utils
import imutils

from utils import (
    compute_eye_aspect_ratio,
    compute_mouth_aspect_ratio,
    ensure_alarm_sound,
    download_shape_predictor,
    ensure_beep_sound,
)


def parse_args():
	parser = argparse.ArgumentParser(description="Driver Drowsiness Detection")
	parser.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture")
	parser.add_argument("--camera-backend", type=str, default="auto", help="Camera backend: auto|msmf|dshow|any")
	parser.add_argument("--width", type=int, default=720, help="Resize width for processing/display")
	parser.add_argument("--ear-threshold", type=float, default=0.25, help="EAR threshold below which eyes are considered closed")
	parser.add_argument("--ear-consec-frames", type=int, default=20, help="Consecutive frames below EAR threshold to trigger drowsiness")
	parser.add_argument("--enable-yawn", action="store_true", default=True, help="Enable yawning detection via MAR")
	parser.add_argument("--disable-yawn", dest="enable_yawn", action="store_false", help="Disable yawning detection")
	parser.add_argument("--mar-threshold", type=float, default=0.60, help="MAR threshold above which yawning is suspected")
	parser.add_argument("--yawn-consec-frames", type=int, default=15, help="Consecutive frames above MAR threshold to flag yawn")
	parser.add_argument("--off-frames", type=int, default=10, help="Consecutive recovery frames to stop alarm (hysteresis) [deprecated by --off-duration-sec]")
	parser.add_argument("--ear-duration-sec", type=float, default=5.0, help="Time eyes must stay below threshold to trigger drowsiness (seconds). Overrides frame-based setting if > 0.")
	parser.add_argument("--off-duration-sec", type=float, default=2.0, help="Time eyes must stay above threshold to stop alarm (seconds). Overrides frame-based setting if > 0.")
	parser.add_argument("--no-audio", action="store_true", help="Disable audio alarm even if pygame is available")
	parser.add_argument("--volume", type=float, default=0.9, help="Alarm volume (0.0 - 1.0)")
	parser.add_argument("--no-overlay", action="store_true", help="Disable drawing face/eye overlays")
	parser.add_argument("--show-fps", action="store_true", help="Show FPS overlay")
	parser.add_argument("--model-dir", type=str, default="models", help="Directory containing shape predictor model")
	parser.add_argument("--predictor", type=str, default=None, help="Path to shape_predictor_68_face_landmarks.dat (overrides model-dir)")
	parser.add_argument("--use-inner-mouth", action="store_true", default=True, help="Use inner mouth landmarks for MAR")
	parser.add_argument("--use-outer-mouth", dest="use_inner_mouth", action="store_false", help="Use outer mouth landmarks for MAR")
	parser.add_argument("--ema-alpha", type=float, default=0.3, help="EMA smoothing factor for EAR (0=no smoothing, closer to 1=more smoothing)")
	parser.add_argument("--min-face-size", type=int, default=100, help="Minimum face width/height in pixels to consider for detection")
	parser.add_argument("--auto-calibrate-seconds", type=float, default=0.0, help="Collect EAR baseline for N seconds at start and set dynamic threshold")
	parser.add_argument("--calibration-scale", type=float, default=0.8, help="Dynamic EAR threshold = baseline * scale (used when auto-calibrating)")
	parser.add_argument("--release-margin", type=float, default=0.03, help="Margin above EAR threshold to consider clearly awake")
	parser.add_argument("--awake-stop-sec", type=float, default=0.5, help="Time EAR must stay above (threshold+margin) to stop alarm immediately")
	parser.add_argument("--detector-interval", type=int, default=3, help="Run face detector every N frames and reuse last face between runs")
	parser.add_argument("--detector-threshold", type=float, default=0.0, help="Detection threshold for HOG detector.run (higher=fewer detections)")
	parser.add_argument("--off-fade-ms", type=int, default=50, help="Fade-out duration in ms when stopping alarm (0=instant)")
	parser.add_argument("--ear-median-window", type=int, default=5, help="Median filter window (frames) applied to EAR before smoothing")
	return parser.parse_args()


def main(args=None):
    """
    Driver Drowsiness Detection main loop.

    Pipeline:
    - Initialize webcam stream
    - Ensure dlib shape predictor model and alarm sound are available
    - Detect faces and facial landmarks
    - Compute EAR for both eyes to detect eye closure across consecutive frames
    - Optionally compute a simple MAR to hint yawning
    - Render overlays and trigger alarm when drowsiness is detected
    - Press 'q' to quit cleanly
    """

    if args is None:
        args = parse_args()

    # Thresholds and runtime configuration
    EAR_THRESHOLD = float(args.ear_threshold)  # Lower EAR indicates closed eyes
    EAR_CONSEC_FRAMES = int(args.ear_consec_frames)  # Consecutive frames below threshold to trigger drowsiness

    # Optional: Simple yawning detection (mouth aspect ratio)
    ENABLE_YAWN = bool(args.enable_yawn)
    MAR_THRESHOLD = float(args.mar_threshold)
    YAWN_CONSEC_FRAMES = int(args.yawn_consec_frames)

    # Alarm hysteresis (recovery frames to stop alarm)
    ALARM_OFF_FRAMES = int(args.off_frames)
    EAR_MIN_DURATION_SEC = float(args.ear_duration_sec)
    ALARM_OFF_DURATION_SEC = float(args.off_duration_sec)

    # Prepare resources (model and alarm)
    if args.predictor:
        shape_predictor_path = args.predictor
    else:
        shape_predictor_path = download_shape_predictor(model_dir=args.model_dir)
    alarm_sound_path = ensure_alarm_sound(alarm_path=os.path.join("alarm", "alert.wav"))
    awake_chime_path = ensure_beep_sound(sound_path=os.path.join("alarm", "awake.wav"), frequency_hz=1200.0, duration_s=0.2)

    # Validate model presence to avoid runtime crash if download failed
    if not os.path.exists(shape_predictor_path):
        print(
            f"[ERROR] Shape predictor missing at '{shape_predictor_path}'.\n"
            "Download failed or was blocked.\n"
            "Please manually place 'shape_predictor_68_face_landmarks.dat' in the 'models' folder and rerun."
        )
        return

    # Initialize dlib's face detector (HOG-based) and the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    # Grab landmark indices for the left/right eye and mouth regions
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    # Initialize alarm (pygame mixer). If audio device not available, continue without sound.
    audio_enabled = not args.no_audio
    alarm_sound = None
    awake_sound = None
    alarm_channel = None
    alarm_active = False
    try:
        if audio_enabled:
            pygame.mixer.init()
            alarm_sound = pygame.mixer.Sound(alarm_sound_path)
            awake_sound = pygame.mixer.Sound(awake_chime_path)
            volume = max(0.0, min(1.0, float(args.volume)))
            alarm_sound.set_volume(volume)
            awake_sound.set_volume(min(1.0, volume))
    except Exception as e:
        print(f"[WARN] Audio initialization failed, continuing without sound: {e}")
        audio_enabled = False

    def start_alarm():
        nonlocal alarm_channel, alarm_active
        if alarm_active or not audio_enabled:
            return
        alarm_active = True
        # Play continuously in a loop until stopped
        if alarm_sound is not None:
            alarm_channel = alarm_sound.play(loops=-1)

    def stop_alarm(immediate: bool = False, play_awake: bool = True):
        nonlocal alarm_channel, alarm_active
        if not alarm_active:
            return
        if audio_enabled and alarm_channel is not None:
            try:
                if immediate or getattr(args, 'off_fade_ms', 300) <= 0:
                    alarm_channel.stop()
                else:
                    alarm_channel.fadeout(int(getattr(args, 'off_fade_ms', 300)))
            except Exception:
                try:
                    alarm_channel.stop()
                except Exception:
                    pass
        alarm_active = False
        # Optional awake chime
        if audio_enabled and awake_sound is not None and play_awake:
            try:
                awake_sound.play(loops=0)
            except Exception:
                pass

    # Start webcam capture with backend fallback
    backend_map = {
        "msmf": cv2.CAP_MSMF,
        "dshow": cv2.CAP_DSHOW,
        "any": cv2.CAP_ANY,
    }
    tried = []
    stream = None
    def try_open(backend_flag):
        s = cv2.VideoCapture(args.camera, backend_flag)
        return s if s.isOpened() else None

    if args.camera_backend.lower() == "auto":
        for name in ["msmf", "dshow", "any"]:
            s = try_open(backend_map[name])
            tried.append(name)
            if s is not None:
                stream = s
                print(f"[INFO] Opened camera with backend: {name}")
                break
    else:
        name = args.camera_backend.lower()
        if name in backend_map:
            stream = try_open(backend_map[name])
            tried.append(name)
            if stream is not None:
                print(f"[INFO] Opened camera with backend: {name}")

    if stream is None or not stream.isOpened():
        print(f"[ERROR] Could not access the webcam (backends tried: {tried}). Ensure it is connected and not used by another application.")
        return

    # Frame counters
    blink_counter = 0
    yawn_counter = 0
    recovery_counter = 0

    # EAR smoothing and timing-based detection
    ear_smooth = None
    ear_window = deque(maxlen=max(1, int(getattr(args, 'ear_median_window', 5))))
    below_start_time = None
    above_start_time = None

    # Calibration
    calibrate_until = None
    calibration_ears = []
    if args.auto_calibrate_seconds and args.auto_calibrate_seconds > 0:
        calibrate_until = time.time() + float(args.auto_calibrate_seconds)

    # FPS tracking
    fps = 0.0
    fps_alpha = 0.9  # EMA smoothing
    prev_time = time.time()
    frame_index = 0
    last_rect = None
    above_margin_start_time = None

    window_name = "Driver Drowsiness Detection"
    cv2.namedWindow(window_name)

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                # Retry a few times in case of transient MSMF issues
                retry_ok = False
                for _ in range(3):
                    time.sleep(0.05)
                    ret, frame = stream.read()
                    if ret and frame is not None:
                        retry_ok = True
                        break
                if not retry_ok:
                    print("[WARN] Failed to grab frame from webcam. Trying to reinitialize...")
                    # Attempt quick reinit
                    stream.release()
                    time.sleep(0.2)
                    # Reopen with same backend
                    if args.camera_backend.lower() == "auto":
                        for name in tried:
                            s = cv2.VideoCapture(args.camera, backend_map.get(name, cv2.CAP_ANY))
                            if s.isOpened():
                                stream = s
                                print(f"[INFO] Reopened camera with backend: {name}")
                                break
                    else:
                        stream = cv2.VideoCapture(args.camera, backend_map.get(args.camera_backend.lower(), cv2.CAP_ANY))
                    # Final attempt to read
                    ret, frame = stream.read()
                    if not ret or frame is None:
                        print("[ERROR] Webcam stream unrecoverable.")
                        break

            # Resize for faster processing and consistent UI
            frame = imutils.resize(frame, width=args.width)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Light denoise to stabilize landmarks
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # Detect faces in the grayscale frame (run at an interval, reuse previous)
            rects = []
            if frame_index % max(1, args.detector_interval) == 0 or last_rect is None:
                try:
                    # detector.run returns (rects, scores, idx)
                    rects_run = detector.run(gray, 0, args.detector_threshold)
                    if isinstance(rects_run, tuple) and len(rects_run) >= 2:
                        rect_candidates, scores = rects_run[0], rects_run[1]
                        if len(rect_candidates) > 0:
                            best_idx = int(max(range(len(rect_candidates)), key=lambda i: scores[i]))
                            last_rect = rect_candidates[best_idx]
                            rects = [last_rect]
                    else:
                        rects = detector(gray, 0)
                        if len(rects) > 0:
                            last_rect = rects[0]
                except Exception:
                    rects = detector(gray, 0)
                    if len(rects) > 0:
                        last_rect = rects[0]
            else:
                if last_rect is not None:
                    rects = [last_rect]

            status_text = "Awake"
            status_color = (0, 255, 0)  # Green for awake
            warning_text = None

            # Process the largest detected face (assume single driver)
            if len(rects) > 0:
                rect = max(rects, key=lambda r: r.width() * r.height())

                # Convert dlib's rectangle to a bounding box and draw it
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                # Skip tiny faces to reduce false triggers
                if w < args.min_face_size or h < args.min_face_size:
                    cv2.putText(
                        frame,
                        "Face too small for reliable detection",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue
                if not args.no_overlay:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 150, 255), 2)

                # Determine facial landmarks and convert to NumPy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Extract eye regions
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]

                # Compute EAR for both eyes and take the average
                leftEAR = compute_eye_aspect_ratio(leftEye)
                rightEAR = compute_eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # Draw eye contours (polylines)
                if not args.no_overlay:
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)

                # Median filter + EMA smoothing for EAR
                ear_window.append(ear)
                ear_med = float(np.median(ear_window)) if len(ear_window) > 0 else float(ear)
                if ear_smooth is None:
                    ear_smooth = ear_med
                else:
                    ear_smooth = args.ema_alpha * ear_smooth + (1.0 - args.ema_alpha) * ear_med

                # Optional calibration in the first N seconds
                now = time.time()
                if calibrate_until is not None and now <= calibrate_until:
                    calibration_ears.append(ear_smooth)
                    if not args.no_overlay:
                        cv2.putText(
                            frame,
                            "Calibrating... keep eyes open",
                            (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                elif calibrate_until is not None and now > calibrate_until:
                    if calibration_ears:
                        baseline = float(sum(calibration_ears) / len(calibration_ears))
                        EAR_THRESHOLD = baseline * float(args.calibration_scale)
                        print(f"[INFO] Auto-calibrated EAR threshold to {EAR_THRESHOLD:.3f} from baseline {baseline:.3f}")
                    calibrate_until = None

                # Check for drowsiness using time-based or frame-based criteria
                is_below = ear_smooth < EAR_THRESHOLD
                if EAR_MIN_DURATION_SEC and EAR_MIN_DURATION_SEC > 0:
                    if is_below:
                        if below_start_time is None:
                            below_start_time = now
                        above_start_time = None
                        above_margin_start_time = None
                    else:
                        if above_start_time is None:
                            above_start_time = now
                        below_start_time = None
                        # Immediate stop if clearly awake by margin for a short time
                        if ear_smooth > (EAR_THRESHOLD + float(getattr(args, 'release_margin', 0.03))):
                            if above_margin_start_time is None:
                                above_margin_start_time = now
                            elif (now - above_margin_start_time) >= float(getattr(args, 'awake_stop_sec', 0.5)):
                                stop_alarm(immediate=True)
                        else:
                            above_margin_start_time = None

                    if below_start_time is not None and (now - below_start_time) >= EAR_MIN_DURATION_SEC:
                        status_text = "Drowsy"
                        status_color = (0, 0, 255)
                        warning_text = "WARNING: Drowsiness Detected!"
                        start_alarm()

                    if alarm_active and above_start_time is not None and (now - above_start_time) >= ALARM_OFF_DURATION_SEC:
                        stop_alarm()
                else:
                    # Fallback to frame-based logic
                    if is_below:
                        blink_counter += 1
                        recovery_counter = 0
                    else:
                        recovery_counter += 1
                        if recovery_counter >= ALARM_OFF_FRAMES:
                            if blink_counter >= EAR_CONSEC_FRAMES:
                                pass
                            blink_counter = 0
                            stop_alarm()

                # If eyes have been closed for sufficient consecutive frames, trigger alarm
                if blink_counter >= EAR_CONSEC_FRAMES:
                    status_text = "Drowsy"
                    status_color = (0, 0, 255)  # Red for drowsy
                    warning_text = "WARNING: Drowsiness Detected!"
                    start_alarm()

                # Optional yawning detection using a simple MAR
                if ENABLE_YAWN:
                    # Choose inner or outer mouth landmarks
                    if args.use_inner_mouth and "inner_mouth" in face_utils.FACIAL_LANDMARKS_IDXS:
                        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
                    else:
                        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
                    mouth = shape[mStart:mEnd]
                    mar = compute_mouth_aspect_ratio(mouth)
                    if mar > MAR_THRESHOLD:
                        yawn_counter += 1
                    else:
                        yawn_counter = 0

                    # Display MAR value for debugging/insight
                    if not args.no_overlay:
                        cv2.putText(
                            frame,
                            f"MAR: {mar:.2f}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2,
                        )

                    if yawn_counter >= YAWN_CONSEC_FRAMES and warning_text is None:
                        warning_text = "Yawning Detected"

                # Display EAR value on the frame
                if not args.no_overlay:
                    below_secs = 0.0
                    if below_start_time is not None:
                        below_secs = max(0.0, time.time() - below_start_time)
                    cv2.putText(
                        frame,
                        f"EAR: {ear_smooth:.2f} Th: {EAR_THRESHOLD:.2f} t<Th: {below_secs:.1f}s",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

            else:
                # No face found; stop any ongoing alarm to avoid nuisance
                stop_alarm()

            # Render status text
            if not args.no_overlay:
                cv2.putText(
                    frame,
                    f"Status: {status_text}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2,
                )

            # Render warning if applicable
            if warning_text is not None:
                cv2.putText(
                    frame,
                    warning_text,
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # FPS overlay (top-right)
            if args.show_fps:
                now = time.time()
                instant_fps = 1.0 / max(1e-6, now - prev_time)
                prev_time = now
                fps = fps_alpha * fps + (1.0 - fps_alpha) * instant_fps if fps > 0 else instant_fps
                if not args.no_overlay:
                    (h, w) = frame.shape[:2]
                    cv2.putText(
                        frame,
                        f"FPS: {fps:.1f}",
                        (w - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            frame_index += 1

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup resources cleanly
        stop_alarm()
        try:
            if audio_enabled:
                pygame.mixer.quit()
        except Exception:
            pass
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)


