import cv2
import os
import numpy as np
import glob

def preprocess_images(input_dir="dataset", output_dir="processed", size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)

    for folder in os.listdir(input_dir):
        person_path = os.path.join(input_dir, folder)
        save_path = os.path.join(output_dir, folder)
        os.makedirs(save_path, exist_ok=True)

        for img_file in glob.glob(f"{person_path}/*.jpg"):
            img = cv2.imread(img_file)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, size)
            normalized = resized / 255.0

            save_file = os.path.join(save_path, os.path.basename(img_file))
            cv2.imwrite(save_file, (normalized * 255).astype("uint8"))

    print(f"[INFO] Preprocessed images saved in '{output_dir}'")
