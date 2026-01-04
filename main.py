import cv2
import mediapipe as mp
import numpy as np

# Initialisation Mediapipe Face Mesh

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_face_mesh = mp.solutions.face_mesh  # usage du module de détection de visage
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True)  # création du détecteur pour un visage.

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255,255,0))
# Ouvre la webcam
cap = cv2.VideoCapture(0) #0 pour caméra principale

while cap.isOpened():
    success, frame = cap.read() #lecture de l'image en continu
    if not success:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_LEFT_EYEBROW,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_spec)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_spec)
    cv2.imshow("Webcam",cv2.flip(frame, 1))
    # Quitte avec ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release() #libère la ressource système caméra
cv2.destroyAllWindows()
