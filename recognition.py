import os
import glob
import numpy as np
import cv2

COSINE_THRESHOLD = 0.363

# Match extracted features with saved .npy database
def match(recognizer, feature1, dictionary):
    best_user = None
    best_score = -1

    for user_id, feature2 in dictionary:
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)

        if score > best_score:
            best_score = score
            best_user = user_id

    if best_score > COSINE_THRESHOLD:
        return True, (best_user, best_score)
    else:
        return False, ("unknown", best_score)

def main():
    directory = os.path.dirname(__file__)

    # ---- Load webcam ----
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open webcam")
        return

    # ---- Load feature dictionary (.npy files) ----
    # Load features from registered_faces folder
    faces_dir = os.path.join(directory, "registered_faces")

    dictionary = []
    for file in glob.glob(os.path.join(faces_dir, "*.npy")):
        feature = np.load(file)
        user_id = os.path.splitext(os.path.basename(file))[0]
        dictionary.append((user_id, feature))


    # ---- Load models ----
    face_detector = cv2.FaceDetectorYN_create(
        "face_detection_yunet_2023mar.onnx",
        "",
        (320, 320)
    )
    face_recognizer = cv2.FaceRecognizerSF_create(
         "face_recognizer_fast.onnx", ""
    )

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape
        face_detector.setInputSize((w, h))

        # Detect faces
        _, faces = face_detector.detect(frame)
        faces = faces if faces is not None else []

        for face in faces:
            x, y, w_box, h_box = map(int, face[:4])

            aligned_face = face_recognizer.alignCrop(frame, face)
            feature = face_recognizer.feature(aligned_face)

            ok, (user_id, score) = match(face_recognizer, feature, dictionary)

            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)

            text = f"{user_id} ({score:.2f})"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Real-Time Face Recognition", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
