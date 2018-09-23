import numpy
import cv2
import os
import pickle
from PIL import Image
import  datetime


def saveFace(fileName, roi):
    if os.path.isfile(fileName):
        os.remove(fileName)
    cv2.imwrite(fileName, roi)
    return

def saveFaceVideo(fileName, frame):
    if os.path.isfile(fileName):
        os.remove(fileName)
    fourCC = cv2.VideoWriter_fourcc(*'avc1') #fourCC = cv2.VideoWriter_fourcc(*'DIVX')
    fHeight = frame.shape[0]
    fWidth = frame.shape[1]
    out = cv2.VideoWriter(fileName, fourCC, 15.0, (fWidth, fHeight))
    return out

def createBounds(color, stroke, frame, coordinates):
    endX = coordinates[x] + coordinates[w]
    endY = coordinates[y] + coordinates[h]
    cv2.rectangle(frame, (coordinates[x], coordinates[y]),(endX, endY), color, stroke)
    return

def frameText(frame, text, coordinates):
    cv2.putText(frame, text, (coordinates[0], coordinates[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return

faceCascade = cv2.CascadeClassifier('cascades/haarcascades/haarcascade_frontalface_alt2.xml')
videoCapture = cv2.VideoCapture(0)

baseRet, baseFrame = videoCapture.read(0)
video = saveFaceVideo("FaceCaptureVideo.mov", baseFrame) #video = saveFaceVideo("FaceCaptureVideo.mp4", baseFrame)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainingFaces.yml")

fishcerRecognizer = cv2.face.FisherFaceRecognizer_create()
fishcerRecognizer.read("trainingFisher.yml")

labelNames = {}

with open("faceLabels.pickle", "rb") as file:
    namesDict = pickle.load(file)
    #inverting dictionary to get names as key and id_ as value
    labelNames = {v:k for k,v in namesDict.items()}

while(True):
    ret, frame = videoCapture.read()

    #greyscale conversion
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayScale,scaleFactor=1.5, minNeighbors=5)
    faceCoordinates = { 'x':0, 'y':0, 'w':0, 'h':0 }

    if ret == True:
        video.write(frame)

    for (x, y, w, h) in faces:
        print (x,y,w,h)
        faceCoordinates[x] = x
        faceCoordinates[y] = y
        faceCoordinates[w] = w
        faceCoordinates[h] = h

        #slicing grayScaled image by the size of the area of my detected face
        #giving us our region of interest
        roiGray = grayScale[y:y+h, x:x+w]
        roiColor = frame[y:y+h, x:x+w]

        #Run predictions on recognizer
        id_, confidence = recognizer.predict(roiGray)
        fID_, fConfidence = fishcerRecognizer.predict(cv2.resize(grayScale[y:y+h, x:x+w], (280,280)))
        if confidence>=45 and fConfidence>=45:
            print(labelNames[id_])
            frameText(frame,labelNames[id_],(x,y))

        #Recognition with deep learned model ie tensorflow, scikit, keras, pytorch

        #create png file and write my caputred face to it
        saveFace("myFace.png", roiColor)
        createBounds((255,0,0),2,frame,faceCoordinates)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

videoCapture.release()
video.release()
cv2.destroyAllWindows()

