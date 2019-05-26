import cv2
import numpy as np
import dlib
import sys

def find_pers(landmarks):

    pt0 = (landmarks.part(0).x, landmarks.part(0).y)
    pt16 = (landmarks.part(16).x,  landmarks.part(16).y)

    pt31 = (landmarks.part(31).x, landmarks.part(31).y)
    pt35 = (landmarks.part(35).x,  landmarks.part(35).y)

    pt33 = (landmarks.part(33).x,  landmarks.part(33).y)
    pt51 = (landmarks.part(51).x, landmarks.part(51).y)

    pt48 = (landmarks.part(48).x,  landmarks.part(48).y)
    pt54 = (landmarks.part(54).x, landmarks.part(54).y)

    pt57 = (landmarks.part(57).x, landmarks.part(57).y)

    pt42 = (landmarks.part(42).x,  landmarks.part(42).y)
    pt45 = (landmarks.part(45).x, landmarks.part(45).y)

    pt36 = (landmarks.part(36).x,  landmarks.part(36).y)
    pt39 = (landmarks.part(39).x, landmarks.part(39).y)

    size_head = pt16[0] - pt0[0]
    size_noze = pt35[0] - pt31[0]
    gap_noz_mouth = pt51[1] - pt33[1]

    size_mouth = pt54[0] - pt48[0]
    eye1 = pt45[0] - pt42[0]
    eye2 = pt39[0] - pt36[0]
    
    #print("##########")
    #print("size head : ",size_head)
    #print(" size mouth : ",size_mouth/size_head)
    #print("size noze : ",size_noze/size_head)
    #print("gap : ",gap_noz_mouth/size_head)
    #print(eye1/size_head, eye2/size_head)
    return [size_mouth/size_head, size_noze/size_head,size_noze/size_head,gap_noz_mouth/size_head, eye1/size_head, eye2/size_head]



def picture(pic):
    img = cv2.imread(pic)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detector(gray)[0]
    landmarks = predictor(gray, face)
    caract = find_pers(landmarks)
    return caract

def is_pers(face, pers):
    check = True
    for i in range(len(face)):
        if (abs(face[i] - pers[i]) > 0.035):
            check = False
            break
    return check


def main():
    if (len(sys.argv) != 2):
        return
    target = picture(sys.argv[1])
    cap = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            landmarks = predictor(gray, face)
            
            cara_face = find_pers(landmarks)
            if (is_pers(cara_face, target)):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'Target',(x1,y1-10), font, 1,(255,255,255),2,cv2.LINE_AA)


        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

main()
