import cv2
import os

def detect_and_crop(image_path, save_folder="cropped_faces"):
    os.makedirs(save_folder, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(image_path)

    if img is None:
        print(f"[ERROR] Unable to read image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(save_folder, f"face_{i}.jpg"), face)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {len(faces)} cropped faces in '{save_folder}'")
