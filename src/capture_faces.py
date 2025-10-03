import cv2
import os

def capture_faces(person_name="Unknown", save_dir="dataset", max_images=30):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    person_dir = os.path.join(save_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    count = 0

    print(f"[INFO] Capturing images for {person_name}. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y+h, x:x+w]
            file_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(file_path, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capture Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {count} images to {person_dir}")
