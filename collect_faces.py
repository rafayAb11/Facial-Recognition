import os
import cv2
import shutil
import json
import numpy as np

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (200, 200) 

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def detect_largest_face(gray):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (x, y, w, h)

def save_face_crop(gray, bbox, save_path):
    x, y, w, h = bbox
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, FACE_SIZE)
    cv2.imwrite(save_path, face)

def train_model():
    print("\nTraining model...")
    from trainer import main as trainer_main
    trainer_main()
    print("Training completed.\n")

def capture_from_webcam():
    person_name = input("Enter the name of the person: ").strip()
    person_dir = os.path.join("dataset", person_name)
    ensure_dir(person_dir)

    cap = cv2.VideoCapture(0)
    count = 0
    target = 60  

    print("Press 'q' to stop capturing early.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox = detect_largest_face(gray)
        if bbox is not None:
            count += 1
            save_path = os.path.join(person_dir, f"{count}.jpg")
            save_face_crop(gray, bbox, save_path)

            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{person_name}: {count}/{target}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Capturing Faces (webcam)", frame)

        if (cv2.waitKey(1) & 0xFF == ord("q")) or count >= target:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {count} cropped face images for {person_name}")

    train_model()

def import_from_folder():
    person_name = input("Enter the name of the person (e.g., messi, ronaldo): ").strip()
    src_folder = input("Enter the path of the folder containing images: ").strip()

    if not os.path.isdir(src_folder):
        print("The source folder does not exist.")
        return

    person_dir = os.path.join("dataset", person_name)
    ensure_dir(person_dir)

    count = 0
    for fname in os.listdir(src_folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(src_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bbox = detect_largest_face(gray)
        if bbox is None:
            continue
        count += 1
        save_path = os.path.join(person_dir, f"{count}.jpg")
        save_face_crop(gray, bbox, save_path)

    print(f"Imported & cropped {count} face images for {person_name}")
    if count == 0:
        print("No usable faces detected in that folder.")
        return

    train_model()

def main():
    ensure_dir("dataset")
    ensure_dir("models")

    print("\n--- Collect Faces ---")
    print("1) Capture from webcam")
    print("2) Import from image folder (auto-crop faces)")
    choice = input("Enter your choice (1/2): ").strip()

    if choice == "1":
        capture_from_webcam()
    elif choice == "2":
        import_from_folder()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
