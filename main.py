from src.capture_faces import capture_faces
from src.preprocess import preprocess_images
from src.detect_faces import detect_and_crop

def main():
    print("========== Facial Recognition Attendance System ==========")
    print("1. Capture Face Images")
    print("2. Preprocess Dataset")
    print("3. Test Face Detection")
    print("4. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        name = input("Enter person's name: ")
        capture_faces(person_name=name)
    elif choice == "2":
        preprocess_images()
    elif choice == "3":
        img_path = input("Enter image path: ")
        detect_and_crop(img_path)
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
