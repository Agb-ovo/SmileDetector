import cv2

trained_face = cv2.CascadeClassifier('frontalface.xml')
trained_smile = cv2.CascadeClassifier('smile.xml')

stream = cv2.VideoCapture('gyuu.mp4')

while True:
    successful_frame_read, frame = stream.read()
    gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_coordinates =  trained_face.detectMultiScale(gray_face)

    # face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
    
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 0), 2) 

        the_face = frame[y:y+h, x:x+w]

        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smile_coordinates =  trained_smile.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)
        # for (x_, y_, w_, h_) in smile_coordinates:
           
        #     cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (0, 225, 0), 2)
        if len(smile_coordinates) > 0:
                cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
               
    cv2.imshow('swagalomo', frame)
    end = cv2.waitKey(1)     

    if end == 81 or end == 113:
      break;

stream.release()  
