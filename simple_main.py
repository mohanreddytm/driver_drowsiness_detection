import argparse
from types import SimpleNamespace

from main import main as run_app


def parse_simple_args():
	parser = argparse.ArgumentParser(description="Simple Driver Drowsiness Detection runner")
	parser.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture")
	parser.add_argument("--predictor", type=str, default="models/shape_predictor_68_face_landmarks.dat", help="Path to shape predictor model")
	parser.add_argument("--camera-backend", type=str, default="auto", help="Camera backend: auto|msmf|dshow|any")
	parser.add_argument("--show-fps", action="store_true", help="Show FPS overlay")
	parser.add_argument("--no-audio", action="store_true", help="Disable audio alarm")
	return parser.parse_args()


def main():
	args_simple = parse_simple_args()
	# Map to the full argument namespace expected by main.main with sensible defaults
	args = SimpleNamespace(
		camera=args_simple.camera,
		camera_backend=args_simple.camera_backend,
		width=720,
		ear_threshold=0.25,
		ear_consec_frames=20,
		enable_yawn=True,
		yawn_consec_frames=15,
		mar_threshold=0.60,
		off_frames=10,
		ear_duration_sec=5.0,
		off_duration_sec=2.0,
		no_audio=args_simple.no_audio,
		volume=0.9,
		no_overlay=False,
		show_fps=bool(args_simple.show_fps),
		model_dir="models",
		predictor=args_simple.predictor,
		use_inner_mouth=True,
		ema_alpha=0.3,
		min_face_size=100,
		auto_calibrate_seconds=0.0,
		calibration_scale=0.8,
		release_margin=0.03,
		awake_stop_sec=0.5,
		detector_interval=3,
		detector_threshold=0.0,
		off_fade_ms=50,
		ear_median_window=5,
	)
	run_app(args)


if __name__ == "__main__":
	main()
