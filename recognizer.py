import os
import cv2
import json
import numpy as np

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (200, 200)
THRESHOLD = 70.0  

def load_labels():
    labels_json = os.path.join("models", "labels.json")
    labels_npy = os.path.join("models", "labels.npy")
    if os.path.isfile(labels_json):
        with open(labels_json, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    elif os.path.isfile(labels_npy):
        return np.load(labels_npy, allow_pickle=True).item()
    else:
        return {}

def ensure_model():
    model_path = os.path.join("models", "face_model.yml")
    if not os.path.isfile(model_path):
        print("Model not found. Train first (run trainer.py or collect_faces.py).")
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    return recognizer

def annotate_prediction(frame, x, y, w, h, name, conf, color=(0, 255, 0)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"{name} | conf={conf:.1f}"
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def predict_face(recognizer, gray, bbox, label_map):
    x, y, w, h = bbox
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, FACE_SIZE)
    label_id, confidence = recognizer.predict(roi)
    name = label_map.get(label_id, "Unknown")
    if confidence >= THRESHOLD:
        name = "Unknown"
    return name, confidence

def recognize_from_webcam():
    label_map = load_labels()
    recognizer = ensure_model()
    if recognizer is None:
        return

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            name, conf = predict_face(recognizer, gray, (x, y, w, h), label_map)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            annotate_prediction(frame, x, y, w, h, name, conf, color=color)

        cv2.imshow("Face Recognition - Webcam (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_from_image(image_path):
    label_map = load_labels()
    recognizer = ensure_model()
    if recognizer is None:
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Image not found or invalid path.")
        return

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        print("No faces detected in this image.")
    for (x, y, w, h) in faces:
        name, conf = predict_face(recognizer, gray, (x, y, w, h), label_map)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        annotate_prediction(img, x, y, w, h, name, conf, color=color)

    cv2.imshow("Face Recognition - Image (press any key to close)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("\n--- Recognizer ---")
    print("1) Webcam recognition")
    print("2) Image recognition")
    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        recognize_from_webcam()
    elif choice == "2":
        path = input("Enter image file path: ").strip()
        recognize_from_image(path)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
