import cv2
import os
import glob

def main():
    base_dir = os.path.dirname(__file__)
    save_dir = os.path.join(base_dir, "registered_faces")

    # Create folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Ask user for the name
    user_name = input("Enter the name for saving faces: ").strip().lower()

    if user_name == "":
        print("Name cannot be empty")
        return

    print(f"Faces will be saved inside 'registered_faces/' as:")
    # Load YuNet and recognizer
    face_detector = cv2.FaceDetectorYN_create(
        "face_detection_yunet_2023mar.onnx",
        "",
        (320, 320)
    )
    face_recognizer = cv2.FaceRecognizerSF_create(
        "face_recognizer_fast.onnx", ""
    )

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access webcam")
        return

    print("Press 'c' to CAPTURE aligned face")
    print("Press 'q' to QUIT")

    # ---- Auto-detect next image number ----
    existing_files = glob.glob(os.path.join(save_dir, f"{user_name}_*.jpg"))
    count = len(existing_files) + 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape
        face_detector.setInputSize((w, h))

        _, faces = face_detector.detect(frame)
        faces = faces if faces is not None else []

        # Draw bounding boxes
        for face in faces:
            x, y, w_box, h_box = map(int, face[:4])
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1)

        # ---- CAPTURE ----
        if key == ord('c'):
            if len(faces) == 0:
                print("No face detected! Try again.")
                continue

            face = faces[0]
            aligned_face = face_recognizer.alignCrop(frame, face)

            filename = f"{user_name}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, aligned_face)

            print(f"Saved: {save_path}")
            count += 1

            cv2.imshow("Aligned Face", aligned_face)
            cv2.waitKey(200)

        # ---- QUIT ----
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
