import os
import numpy as np
import cv2
import time
from utils.mediapipe_loader import create_live_landmarker, frame_to_mp_image
from utils.landmarks_util import process_two_hands

GESTURE_LIST = ["Hello", "Yes", "No", "Stop", "Wait", "Come_Here", "Go_Away", "Thank_You", "Help", "Clap"]
SAMPLES_PER_GESTURE = 50

cap = cv2.VideoCapture(0)
landmarker = create_live_landmarker(num_hands=2)

print("------DATA COLLECTION STARTING---------")

for gesture in GESTURE_LIST:
    SAVE_DIR = f"dataset/landmarks/{gesture}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    saved_count = 0
    sequence = []
    
    print(f"-------PREPARE TO RECORD {gesture} --------")  
    start_time = time.time()
    countdown_seconds = 5
    current_countdown = countdown_seconds
    last_tick = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue  
            
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f"GET READY FOR: {gesture}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(frame, f"Starting in: {current_countdown}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 5)
        cv2.imshow("Data Collector", frame)
        cv2.waitKey(1)
        
        if time.time() - last_tick >= 1.0:
            current_countdown -= 1
            last_tick = time.time()
            print(f"Starting in {current_countdown}...")
            
        if current_countdown <= 0:
            break
    print(f"--------NOW COLLECTING FOR : {gesture} --------")
    
    while saved_count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        result = landmarker.detect_for_video(frame_to_mp_image(rgb), timestamp_ms)
        
        feat = process_two_hands(result.hand_landmarks)
        sequence.append(feat)
        
        if(len(sequence) > 30):
            sequence.pop(0)
            
        cv2.putText(frame, f"CURRENT GESTURE: {gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Progress: {saved_count}/{SAMPLES_PER_GESTURE} samples", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Buffer Window: {len(sequence)}/30", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, "Hold gesture & press 'r' to save", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.imshow("Data Collector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') and len(sequence) == 30:
            np.save(os.path.join(SAVE_DIR, f"sample_{saved_count}.npy"), np.array(sequence))
            saved_count += 1
            sequence = []
            while True:
                ret, pause_frame = cap.read()
                if not ret: 
                    break
                pause_frame = cv2.flip(pause_frame, 1)
                
                cv2.rectangle(pause_frame, (10, 20), (620, 150), (0, 0, 0), -1)
                
                cv2.putText(pause_frame, " SAMPLE SAVED!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.putText(pause_frame, "PAUSED: Press 'r' to start next sample...", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("Data Collector", pause_frame)
                
                pause_key = cv2.waitKey(1) & 0xFF
                
                if pause_key == ord('r'): 
                    if saved_count == SAMPLES_PER_GESTURE:
                        break
                    pause_duration = 5
                    pause_start = time.time()
                    
                    while True:
                        elapsed = time.time() - pause_start
                        countdown_remaining = int(pause_duration - elapsed)
                        
                        if countdown_remaining <= 0:
                            break
                            
                        ret, pause_frame = cap.read()
                        if not ret: 
                            break
                        pause_frame = cv2.flip(pause_frame, 1)
                    
                        cv2.rectangle(pause_frame, (10, 20), (620, 160), (0, 0, 0), -1)
                        cv2.putText(pause_frame, "GET READY FOR NEXT SAMPLE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        cv2.putText(pause_frame, f"Recording starts in: {countdown_remaining}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        
                        cv2.imshow("Data Collector", pause_frame)
                        cv2.waitKey(1)
                    break
            
        elif key == ord('q'):
            print("------- COLLECTION BROKEN MANUALLY -------") 
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            exit()
    
cap.release()
cv2.destroyAllWindows()
landmarker.close()
print("-------- DATA COLLECTION COMPLETE -----------")
        
