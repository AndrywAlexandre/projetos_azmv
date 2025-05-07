import cv2

file_name= 'IMG_2149.mov'

cap= cv2.VideoCapture(file_name)

while (cap.isOpened()):
    ret, frame=cap.read()
    if ret == True:
        cv2.imshow('Frame',frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

cv2.destroyAllWindows()
