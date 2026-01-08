import cv2
import os

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

trainer_path = os.path.join(BASE_DIR, "trainer", "trainer.yml")
labels_path = os.path.join(BASE_DIR, "trainer", "labels.txt")
cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# ---------- LOAD MODEL ----------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

# ---------- LOAD HAAR ----------
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("Haarcascade not loaded")
    exit()

# ---------- LOAD LABELS ----------
id_to_name = {}
with open(labels_path, "r") as f:
    for line in f:
        id_, name = line.strip().split(",")
        id_to_name[int(id_)] = name

print("Labels:", id_to_name)

# ---------- CAMERA ----------
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera not working")
    exit()

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font = cv2.FONT_HERSHEY_SIMPLEX

# ---------- MEMORY FOR STABILITY ----------
prev_faces = None

# ---------- LOOP ----------
while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(100, 100)
    )

    # stabilize bounding box
    if len(faces) > 0:
        prev_faces = faces
    elif prev_faces is not None:
        faces = prev_faces
    else:
        faces = []

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        id_, confidence = recognizer.predict(face_img)

        if confidence < 60 and id_ in id_to_name:
            name = id_to_name[id_]
            conf_text = f"{int(100 - confidence)}%"
        else:
            name = "Unknown"
            conf_text = ""

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10),
                    font, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, conf_text, (x, y+h+20),
                    font, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(10) & 0xFF == 27:  # ESC
        break

# ---------- CLEANUP ----------
cam.release()
cv2.destroyAllWindows()
