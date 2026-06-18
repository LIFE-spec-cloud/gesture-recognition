import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "C:\Projects\gesture-recognition\models\mediapipe\handlandmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def load_landmarker_task(num_hands=2):
    if not os.path.exists(MODEL_PATH):
        print("Downloading Taks model file from google storage")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done downloading")
        
    options = mp_vision.HandLandmarkerOptions(
        base_options = mp_python.BaseOptions(model_asset_path = MODEL_PATH),
        running_mode = mp_vision.RunningMode.VIDEO,
        num_hands = num_hands,
        min_hand_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )
    
    return mp_vision.HandLandmarker.create_from_options(options)

def create_live_landmarker(num_hands=2):
    return load_landmarker_task(num_hands)

def frame_to_mp_image(frame_rgb):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data = frame_rgb)
