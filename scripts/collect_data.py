import os
import numpy as np
import cv2
from utils.mediapipe_loader import create_live_landmarker, frame_to_mp_image
from utils.landmarks_util import landmarks_to_array, normalize_frame

GESTURE_LIST = []