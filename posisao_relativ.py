import cv2
import mediapipe as mp
from mediapipe.tasks import python
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


file_name= 'IMG_2149.mov'


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


cap= cv2.VideoCapture(file_name)

options = python.vision.PoseLandmarkerOptions()

#Comando para execultar o video com sua abertura e seu fechamento
while (cap.isOpened()):
    ret, frame=cap.read()
#comando ate o primero brack para parar o video usando o 'Q'
    if ret == True:
        cv2.imshow('Frame',frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

cv2.destroyAllWindows()
