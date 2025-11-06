"""
detector.py

Face detection helpers:
- run_haar_faces: Haar with specified params
- run_nms: Non-maximum suppression to drop duplicates
- select_primary_face: choose largest acceptable box with head-region constraint
- validate_with_vit: ViT-B/16 classifier-based validation returning confidence
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import vit_b_16, ViT_B_16_Weights

from utils import nms, preprocess_for_vit


def _resolve_cascade_path(filename: str) -> str:
	import os
	local_path = os.path.join("haarcascade", filename)
	if os.path.exists(local_path):
		return local_path
	return cv2.data.haarcascades + filename


_face_cascade = cv2.CascadeClassifier(_resolve_cascade_path("haarcascade_frontalface_default.xml"))
_eye_cascade = cv2.CascadeClassifier(_resolve_cascade_path("haarcascade_eye.xml"))

_vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
_vit.eval()
_transform = T.Compose([
	T.ToPILImage(),
	T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
	T.ToTensor(),
	T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def run_haar_faces(gray_pre: np.ndarray) -> List[Tuple[int, int, int, int]]:
	faces = _face_cascade.detectMultiScale(
		gray_pre, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
	)
	return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces] if faces is not None else []


def run_nms(boxes: List[Tuple[int, int, int, int]], iou_thr: float = 0.3) -> List[Tuple[int, int, int, int]]:
	return nms(boxes, None, iou_thr)


def select_primary_face(boxes: List[Tuple[int, int, int, int]], frame_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
	if not boxes:
		return None
	H, W = frame_shape[0], frame_shape[1]
	min_area = 0.05 * W * H
	max_area = 0.80 * W * H
	best = None
	best_area = -1
	for (x, y, w, h) in boxes:
		area = w * h
		if area < min_area or area > max_area:
			continue
		cx = x + w / 2.0
		cy = y + h / 2.0
		if cy > 0.70 * H:
			# reject boxes centered in lower 30% (likely torso)
			continue
		if w <= 0 or h <= 0 or max(w / h, h / w) > 3.0:
			continue
		if area > best_area:
			best_area = area
			best = (x, y, w, h)
	return best


@torch.inference_mode()
def validate_with_vit(frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> float:
	x, y, w, h = box
	crop = frame_bgr[max(0, y): y + h, max(0, x): x + w]
	if crop.size == 0:
		return 0.0
	rgb = preprocess_for_vit(crop)
	img = _transform(rgb).unsqueeze(0)
	logits = _vit(img)
	probs = torch.softmax(logits, dim=1)
	return float(probs.max().item())


def get_eye_cascade() -> cv2.CascadeClassifier:
	return _eye_cascade
