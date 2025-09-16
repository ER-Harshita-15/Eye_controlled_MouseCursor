
import cv2
import mediapipe as mp
import numpy as np
import autopy
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Get screen size for cursor mapping
screen_w, screen_h = autopy.screen.size()

# EAR calculation function
def eye_aspect_ratio(eye_points, w, h):
    coords = [(int(p.x * w), int(p.y * h)) for p in eye_points]
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))  # vertical 1
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))  # vertical 2
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear, coords

# Thresholds
wink_threshold = 0.22  # tune after testing
cooldown_frames = 20
frame_counter = 0

# Start webcam
cam = cv2.VideoCapture(0)

while True:
    ret, image = cam.read()
    if not ret:
        break
    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # ---------- Cursor Movement (iris landmarks) ----------
        for id, lm in enumerate(landmarks[474:478]):
            x = int(lm.x * w)
            y = int(lm.y * h)

            if id == 1:  # main tracking point
                mouse_x = int(screen_w / w * x)
                mouse_y = int(screen_h / h * y)
                autopy.mouse.move(screen_w - mouse_x, mouse_y)  # flip x for mirror

            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        # ---------- Left Eye Wink Detection ----------
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        left_EAR, left_coords = eye_aspect_ratio([landmarks[i] for i in LEFT_EYE], w, h)
        right_EAR, right_coords = eye_aspect_ratio([landmarks[i] for i in RIGHT_EYE], w, h)

        # Draw eyes
        for (x, y) in left_coords:
            cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
        for (x, y) in right_coords:
            cv2.circle(image, (x, y), 2, (255, 0, 255), -1)

        # Show EAR values
        cv2.putText(image, f"L_EAR: {left_EAR:.3f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"R_EAR: {right_EAR:.3f}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Detect left wink and click
        if frame_counter == 0:
            if left_EAR < wink_threshold and right_EAR > wink_threshold:
                autopy.mouse.click()
                print("LEFT WINK CLICKED")
                frame_counter = cooldown_frames

        if frame_counter > 0:
            frame_counter -= 1

    cv2.imshow("Eye Controlled Mouse", image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or Q to quit
        break

cam.release()
cv2.destroyAllWindows()
