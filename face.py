import cv2
import os
import numpy as np
import joblib
import threading
import time
from deepface import DeepFace

# Load camera
cap = cv2.VideoCapture(0)

# Load trained models
print("[INFO] Loading SVM models...")
loaded = joblib.load("svm_model/all_loaded_models")
scaler = loaded.get('scaler')
grid_search = loaded.get('grid_search')
print("[INFO] Models loaded successfully.")

# Haarcascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

frame_count = 0
last_prediction = "Unknown"
lock = threading.Lock()
print(f"this is from the top {last_prediction}")

# Control recognition frequency & overlap
last_inference_time = 0
is_processing = False


def recognize_face(face_img):
    global last_prediction, is_processing
    print(f"this is from the recognize_face function {last_prediction,is_processing}")
    print("[THREAD] Started recognize_face")
    try:
        # Resize for speed (ArcFace default input)
        face_resized = cv2.resize(face_img, (112, 112))

        print("[THREAD] Calling DeepFace.represent...")
        embedding = DeepFace.represent(
            img_path=face_resized,
            model_name="ArcFace",   # faster than ArcFace
            detector_backend="skip",   # we already cropped
            enforce_detection=False
        )
        print("[THREAD] Embedding created ✅")

        emb_obj = embedding[0]['embedding']
        embeddings = np.array(emb_obj).reshape(1, -1)
        scaled_emb = scaler.transform(embeddings)
        prediction = grid_search.best_estimator_.predict(scaled_emb)[0]

        print("[THREAD] Prediction:", prediction)

        with lock:
            last_prediction = prediction

    except Exception as e:
        print("[THREAD ERROR]", e)
        with lock:
            last_prediction = "Unknown"

    finally:
        is_processing = False


print("[INFO] Starting video loop...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab the frame")
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=7, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Start recognition every 2 seconds if not already processing
        if (time.time() - last_inference_time > 2) and not is_processing:
            last_inference_time = time.time()
            is_processing = True
            print(f"[LOOP] Starting recognition thread at frame {frame_count}, face shape={face_img.shape}")
            threading.Thread(
                target=recognize_face,
                args=(face_img.copy(),),
                daemon=True
            ).start()

        # Draw last prediction
        with lock:
            label = str(last_prediction)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("[INFO] Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
