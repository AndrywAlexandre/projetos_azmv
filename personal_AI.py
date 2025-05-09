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
base_options = python.BaseOptions(model_asset_path=model_path)




class PersonlAI:
    def __init__(self,file_name = 'IMG_2149.mov'):
        self.file_name=file_name
        model_path = "pose_landmarker_lite.task"

        self.options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO)

    # Função para desenhar os pontos de pose no frame
    def draw_landmarks_on_image(self,rgb_image, detection_result):
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

    def process_vide(self,draw,display):
    
    # Configurações do modelo
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )

        # Inicializa o landmarker
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(self,file_name)

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


if __name__ == "__main__":
    personlAI = PersonlAI()
    personlAI.process_vide(True,True)