'''import cv2
import mediapipe as mp
from mediapipe.tasks import python
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


file_name= 'IMG_2149.mov'
model_path = "pose_landmarker_lite.task"

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


# Cria a base_options com o modelo
base_options = python.BaseOptions(model_asset_path=model_path)

# Agora cria o options corretamente
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO
)

#options = python.vision.PoseLandmarkerOptions(
#   base_options = python.BaseOptions(model_asset_path=model_path),
#   #definir como será o processamento da imagem com a IA
#   running_mode=python.vision.RunningMode.VIDEO
#)

with python.vision.PoseLandmarker.create_from_options(options) as landmarker:
    cap= cv2.VideoCapture(file_name)
    calc_ts=[0.0]
    #Comando para execultar o video com sua abertura e seu fechamento
    while (cap.isOpened()):
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        
    #comando ate o primero brack para parar o video usando o 'Q'
        if ret == True:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            #time stamp
            calc_ts.append(int(calc_ts[-1] + 1000/fps))
            landmarker.process(frame)

            detection_result = landmarker.detect_for_video(mp_image,calc_ts[-1])
            cv2.imshow('Frame',frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
cap.release()

cv2.destroyAllWindows()'''


import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Caminho para vídeo e modelo
file_name = 'IMG_2149.mov'
model_path = "pose_landmarker_lite.task"

# Função para desenhar os pontos de pose no frame
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

# Configurações do modelo
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

# Inicializa o landmarker
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(file_name)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Converte o frame BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Converte para objeto mediapipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Timestamp do frame atual (em milissegundos)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Detecta landmarks
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Desenha os pontos no frame
        annotated_frame = draw_landmarks_on_image(frame_rgb, detection_result)

        # Converte de volta RGB → BGR para exibir com OpenCV
        frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Pose Detection', frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
