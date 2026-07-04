import numpy as np

def landmarks_to_array(landmarks_list):
    coords = np.zeros(63)
    
    for i, lm in enumerate(landmarks_list):
        base = i * 3
        coords[base] = lm.x
        coords[base + 1] = lm.y
        coords[base + 2] = lm.z
        
    return coords

def normalize_frame(raw_frame):
    landmarks = raw_frame.reshape(21, 3).copy()
    
    WRIST_IDX = 0
    MIDDLE_FINGER_TIP_IDX = 12
    
    wrist = landmarks[WRIST_IDX]
    landmarks -= wrist
    
    ref_vec = landmarks[MIDDLE_FINGER_TIP_IDX]
    ref_dist = np.linalg.norm(ref_vec)
    
    if ref_dist > 1e-5:
        landmarks /= ref_dist
        
    return landmarks.reshape(-1)

def process_two_hands(landmarks_all):
    
    if(len(landmarks_all)) > 0:
        hand1_raw = landmarks_to_array(landmarks_all[0])
        hand1_norm = normalize_frame(hand1_raw)
    else:
        hand1_norm = np.zeros(63)
        
    if(len(landmarks_all)) > 1:
        hand2_raw = landmarks_to_array(landmarks_all[1])
        hand2_norm = normalize_frame(hand2_raw)
    else:
        hand2_norm = np.zeros(63)
        
    return np.concatenate([hand1_norm, hand2_norm])