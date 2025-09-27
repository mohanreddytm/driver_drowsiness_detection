import os
import urllib.request

import numpy as np


def compute_eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) given 6 eye landmark points.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Lower EAR values indicate closed eyes. Typical threshold ~0.2 - 0.3.

    Parameters
    ----------
    eye_points : np.ndarray
        Array of shape (6, 2) for the eye landmarks.

    Returns
    -------
    float
        The computed EAR value.
    """
    p1, p2, p3, p4, p5, p6 = eye_points
    # Euclidean distances between vertical pairs
    dist_2_6 = np.linalg.norm(p2 - p6)
    dist_3_5 = np.linalg.norm(p3 - p5)
    # Euclidean distance between horizontal pair
    dist_1_4 = np.linalg.norm(p1 - p4)
    ear = (dist_2_6 + dist_3_5) / (2.0 * dist_1_4 + 1e-6)
    return float(ear)


def compute_mouth_aspect_ratio(mouth_points: np.ndarray) -> float:
    """
    Compute a simple Mouth Aspect Ratio (MAR) to detect yawning.

    Uses the inner mouth landmarks: vertical distance divided by horizontal width.

    Parameters
    ----------
    mouth_points : np.ndarray
        Array of shape (~20, 2) covering the mouth region.

    Returns
    -------
    float
        The computed MAR value.
    """
    # If inner-mouth landmarks (8 points: 60..67) are provided, use a standard MAR formula:
    # MAR_inner = (||p62-p66|| + ||p63-p65||) / (2 * ||p60-p64||)
    if mouth_points.shape[0] == 8:
        p60, p61, p62, p63, p64, p65, p66, p67 = mouth_points
        vertical_1 = np.linalg.norm(p62 - p66)
        vertical_2 = np.linalg.norm(p63 - p65)
        horizontal = np.linalg.norm(p60 - p64)
        mar = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)
        return float(mar)

    # Fallback heuristic for outer mouth (or unknown shape): use bounding distances
    y_top = np.min(mouth_points[:, 1])
    y_bottom = np.max(mouth_points[:, 1])
    x_left = np.min(mouth_points[:, 0])
    x_right = np.max(mouth_points[:, 0])

    vertical = abs(y_bottom - y_top)
    horizontal = abs(x_right - x_left)
    mar = vertical / (horizontal + 1e-6)
    return float(mar)


def ensure_alarm_sound(alarm_path: str) -> str:
    """
    Ensure an alarm wav file exists at `alarm_path`.
    If missing, generate a simple WAV using numpy and write it.
    """
    os.makedirs(os.path.dirname(alarm_path), exist_ok=True)
    if os.path.exists(alarm_path):
        return alarm_path

    # Procedurally generate a loud beep WAV
    try:
        import wave
        import struct
        import math

        framerate = 44100
        duration_s = 2.0
        frequency_hz = 800.0
        amplitude = 32767

        with wave.open(alarm_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(framerate)

            num_samples = int(duration_s * framerate)
            for i in range(num_samples):
                t = i / framerate
                sample = int(amplitude * math.sin(2 * math.pi * frequency_hz * t))
                wav_file.writeframes(struct.pack('<h', sample))
    except Exception:
        # Fallback: create an empty file to avoid crashes; playback may be silent
        with open(alarm_path, 'wb') as f:
            f.write(b'')

    return alarm_path


def ensure_beep_sound(sound_path: str, frequency_hz: float = 1200.0, duration_s: float = 0.25) -> str:
    """
    Ensure a short beep WAV exists at `sound_path`.
    Generates a simple sine beep if missing.
    """
    os.makedirs(os.path.dirname(sound_path), exist_ok=True)
    if os.path.exists(sound_path):
        return sound_path

    try:
        import wave
        import struct
        import math

        framerate = 44100
        amplitude = 20000

        with wave.open(sound_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(framerate)

            num_samples = int(duration_s * framerate)
            for i in range(num_samples):
                t = i / framerate
                sample = int(amplitude * math.sin(2 * math.pi * frequency_hz * t))
                wav_file.writeframes(struct.pack('<h', sample))
    except Exception:
        with open(sound_path, 'wb') as f:
            f.write(b'')

    return sound_path

def download_shape_predictor(model_dir: str = "models") -> str:
    """
    Ensure the dlib 68-point shape predictor model is available locally.
    If missing, download the compressed .bz2 and extract to `model_dir`.

    Returns the path to the `.dat` model.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_dat = os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat")
    if os.path.exists(model_dat):
        return model_dat

    # Download from dlib's official source if available; mirror could be used if blocked.
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    bz2_path = model_dat + ".bz2"

    try:
        print("[INFO] Downloading shape predictor (100+ MB). This may take a while...")

        def _reporthook(block_num, block_size, total_size):
            if total_size <= 0:
                return
            downloaded = block_num * block_size
            percent = min(100, int(downloaded * 100 / total_size))
            if percent % 5 == 0:
                print(f"\r[INFO] Download progress: {percent}%", end="")

        urllib.request.urlretrieve(url, bz2_path, _reporthook)
        print("\r[INFO] Download completed.            ")

        # Extract .bz2
        import bz2

        with bz2.BZ2File(bz2_path, 'rb') as f_in, open(model_dat, 'wb') as f_out:
            data = f_in.read()
            f_out.write(data)
        os.remove(bz2_path)
    except Exception as e:
        print(f"[WARN] Could not download model automatically: {e}")
        print("[ACTION] Please manually place 'shape_predictor_68_face_landmarks.dat' in the 'models' directory.")

    return model_dat


