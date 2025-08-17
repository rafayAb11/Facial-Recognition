import os
import cv2
import json
import numpy as np

FACE_SIZE = (200, 200)  

def main():
    dataset_path = "dataset"
    models_path = "models"
    os.makedirs(models_path, exist_ok=True)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    if not os.path.isdir(dataset_path):
        print("dataset/ not found. Nothing to train.")
        return

    person_dirs = [d for d in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, d))]
    if not person_dirs:
        print("dataset/ is empty. Add some people first.")
        return

    for person_name in sorted(person_dirs):
        person_path = os.path.join(dataset_path, person_name)
        label_map[current_label] = person_name

        for file in os.listdir(person_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
           
            if img.shape != FACE_SIZE[::-1]:
                img = cv2.resize(img, FACE_SIZE)

            faces.append(img)
            labels.append(current_label)

        current_label += 1

    if not faces:
        print("No images found to train on.")
        return

    labels = np.array(labels, dtype=np.int32)
    recognizer.train(faces, labels)

    # Save model + labels (both JSON + NPY for convenience)
    recognizer.save(os.path.join(models_path, "face_model.yml"))
    with open(os.path.join(models_path, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(models_path, "labels.npy"), label_map)

    print("Training completed.")
    print("Saved:", os.path.join(models_path, "face_model.yml"))
    print("Saved:", os.path.join(models_path, "labels.json"))

if __name__ == "__main__":
    main()
