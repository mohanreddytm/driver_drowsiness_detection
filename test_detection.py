"""
Quick offline test for detection + ViT confidence + EAR.
Place a few test images under samples/ and run:

python test_detection.py
"""

import glob
import os

import cv2

from detector import run_haar_faces, run_nms, select_primary_face, validate_with_vit, get_eye_cascade
from eyes import detect_eyes_haar
from utils import apply_preprocessing


def main():
	paths = glob.glob(os.path.join("samples", "*.jpg")) + glob.glob(os.path.join("samples", "*.png"))
	if not paths:
		print("No images in samples/. Add a few test images.")
		return
	for p in paths:
		img = cv2.imread(p)
		if img is None:
			continue
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_pre = apply_preprocessing(gray)
		cands = run_haar_faces(gray_pre)
		nms_boxes = run_nms(cands)
		primary = select_primary_face(nms_boxes, img.shape)
		conf = 0.0
		if primary is not None:
			conf = validate_with_vit(img, primary)
			eyes = detect_eyes_haar(gray_pre, primary, get_eye_cascade())
			print(f"{os.path.basename(p)}: face={primary}, ViT={conf:.2f}, eyes={len(eyes)}")
		else:
			print(f"{os.path.basename(p)}: no face detected")


if __name__ == "__main__":
	main()
