import bz2
import os

# Paths
bz2_path = "models/shape_predictor_68_face_landmarks.dat.bz2"
dat_path = "models/shape_predictor_68_face_landmarks.dat"

# Check if already extracted
if os.path.exists(dat_path):
    print("[INFO] Model already extracted:", dat_path)
else:
    print("[INFO] Extracting model, please wait...")
    with bz2.BZ2File(bz2_path, 'rb') as f_in, open(dat_path, 'wb') as f_out:
        f_out.write(f_in.read())
    print("[INFO] Extraction completed:", dat_path)
