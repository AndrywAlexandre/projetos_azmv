import cv2

file_name= 'IMG_2149.mov'

cap= cv2.VideoCapture(file_name)

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
