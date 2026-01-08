import cv2
import numpy as np
from PIL import Image
import os

# ---------------- BASE DIR ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, "dataset")
trainer_path = os.path.join(BASE_DIR, "trainer")
cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

os.makedirs(trainer_path, exist_ok=True)

# ---------------- LOAD MODELS ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cascade_path)

if detector.empty():
    print("❌ Haarcascade not loaded")
    exit()

# ---------------- NAME → ID MAPPING ----------------
name_id_map = {}
current_id = 0

def getImagesAndLabels(path):
    global current_id
    faceSamples = []
    ids = []

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    for imagePath in imagePaths:
        filename = os.path.split(imagePath)[-1]
        parts = filename.split(".")

        if len(parts) < 3:
            continue

        name = parts[1]   # arjun, vig, etc.

        if name not in name_id_map:
            current_id += 1
            name_id_map[name] = current_id

        id = name_id_map[name]

        img = Image.open(imagePath).convert('L')
        img_numpy = np.array(img, 'uint8')

        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

# ---------------- TRAIN ----------------
print("\n[INFO] Training faces. Please wait...")

faces, ids = getImagesAndLabels(dataset_path)

if len(faces) == 0:
    print("❌ No faces found in dataset")
    exit()

recognizer.train(faces, np.array(ids))
recognizer.save(os.path.join(trainer_path, "trainer.yml"))

# Save name mapping
with open(os.path.join(trainer_path, "labels.txt"), "w") as f:
    for name, id in name_id_map.items():
        f.write(f"{id},{name}\n")

print("\n[INFO] Training completed")
print("[INFO] Name-ID mapping:", name_id_map)
print("[INFO] trainer.yml + labels.txt saved")
