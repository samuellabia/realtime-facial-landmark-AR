import cv2
import mediapipe as mp

# Initialisation Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh  # usage du module de détection de visage
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)  # création du détecteur pour un visage.


# Ouvre la webcam
cap = cv2.VideoCapture(0) #0 pour caméra principale

while cap.isOpened():
    success, frame = cap.read() #lecture de l'image en continu
    if not success:
        break
    cv2.imshow("Webcam", frame)
    
    # Quitte avec ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release() #libère la ressource système caméra
cv2.destroyAllWindows()
