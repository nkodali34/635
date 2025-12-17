import os
import glob
import numpy as np
import cv2

def main():
    base_dir = os.path.dirname(__file__)
    faces_dir = os.path.join(base_dir, "registered_faces")

    if not os.path.exists(faces_dir):
        print("registered_faces folder not found!")
        return

    # Load model
    face_recognizer = cv2.FaceRecognizerSF_create(
        "face_recognizer_fast.onnx", ""
    )

    # Get all JPG images
    image_files = glob.glob(os.path.join(faces_dir, "*.jpg"))

    if len(image_files) == 0:
        print("No face images found in registered_faces/")
        return

    print(f"Found {len(image_files)} face images. Generating features...")

    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read: {img_path}")
            continue

        # Extract feature
        face_feature = face_recognizer.feature(image)

        # Save feature as .npy
        basename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(faces_dir, basename + ".npy")

        np.save(out_path, face_feature)

        print(f"{basename}.npy created")

    print("\n All features generated successfully inside registered_faces/")

if __name__ == "__main__":
    main()
