import cv2
import os

# ---------------- CAMERA SETUP ----------------
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Camera not accessible")
    exit()

cam.set(3, 640)  # width
cam.set(4, 480)  # height

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cascade_path = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

dataset_path = os.path.join(BASE_DIR, "dataset")

# Create dataset folder if not exists
os.makedirs(dataset_path, exist_ok=True)

# ---------------- LOAD HAARCASCADE ----------------
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("❌ Haarcascade file not loaded")
    exit()
else:
    print("✅ Haarcascade loaded successfully")

# ---------------- USER INPUT ----------------
face_id = input("\nEnter numeric user id and press <return>: ")

print("\nInitializing face capture. Look at the camera...")

count = 0

# ---------------- FACE CAPTURE LOOP ----------------
while True:
    ret, img = cam.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save face image
        file_name = f"User.{face_id}.{count}.jpg"
        cv2.imwrite(
            os.path.join(dataset_path, file_name),
            gray[y:y + h, x:x + w]
        )

        cv2.imshow('Face Dataset', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27 or count >= 30:  # ESC or 30 images
        break

# ---------------- CLEANUP ----------------
print("\nDataset collection completed.")
cam.release()
cv2.destroyAllWindows()
