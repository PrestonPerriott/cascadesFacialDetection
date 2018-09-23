import  os
import cv2
import numpy
import pickle
from PIL import Image
#Our training document


yLabels = []
xTrain = []
fisherXTrain = []

labelIDs = {"Name" : 1}
pathArray = []
labelNameArray = []

#This function assumes your images are in the same dir as this file
def buildFilePath(folderName):
    #absolute path of this file
    basePath = os.path.dirname(os.path.abspath(__file__))
    imagesPath = os.path.join(basePath, folderName)
    print ("Our file path for the images is : " + imagesPath)
    return imagesPath

# we set important global variables here
def setVariablesFromImagePath(directory):
    currLabelIdx = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
                imagePath = os.path.join(root, file)
                imageLabel = os.path.basename(root).replace(" ", "-").lower()

                if not imageLabel in labelIDs:
                    labelIDs[imageLabel] = currLabelIdx
                    currLabelIdx = currLabelIdx + 1
                    pathArray.append(imagePath)

                labelNameArray.append(labelIDs[imageLabel])
                yLabels.append(labelIDs[imageLabel])
                print("Our YLabels are : " + str(yLabels))
    return

def grayScaleFromImagePathArray(imagePathArray):
    grayScaleArray = []
    for path in imagePathArray:
        grayScaled = Image.open(path).convert("L")
        grayScaleArray.append(grayScaled)
    return grayScaleArray

def faceDetectionFromGrayScaleArray(grayScaleArray):
    for grayedImage in grayScaleArray:
        numpyImageArray = numpy.array(imageResize((280,280), grayedImage), "uint8")
        #Detection of the faces
        faces = faceCascade.detectMultiScale(numpyImageArray, scaleFactor=1.5, minNeighbors=5)

        for (x,y,w,h) in faces:
            roi = numpyImageArray[y:y+h, x:x+w]
            xTrain.append(roi)
            # fisherXTrain.append(imageResize((250,250), roi))
            fisherXTrain.append(cv2.resize(numpyImageArray[y:y + h, x:x + w], (280, 280)))
    print ("Our X Train is : " + str(xTrain))
    return

def imageResize(size, image):
    resized = image.resize(size, Image.ANTIALIAS)
    return resized

#absolute path of the file
baseDir = os.path.dirname(os.path.abspath(__file__))
imageDir = os.path.join(baseDir, "images")
faceCascade = cv2.CascadeClassifier('cascades/haarcascades/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
fisherRecognizer = cv2.face.FisherFaceRecognizer_create()


imagesPath = buildFilePath("images")

setVariablesFromImagePath(imagesPath)

grayArray = grayScaleFromImagePathArray(pathArray)
faceDetectionFromGrayScaleArray(grayArray)

# for root, dirs, files in os.walk(imageDir):
#     #iteratre through the files
#     for file in files:
#         if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
#             imageCount += 1
#             path = os.path.join(root, file)
#             #iterate through to get file names for people
#             label = os.path.basename(root).replace(" ", "-").lower()
#
#             #add an id for each image
#             if not label in labelIDs:
#                 labelIDs[label] = currLabelIdx
#                 currLabelIdx += 1
#
#             id_ = labelIDs[label]
#
#             grayScale = Image.open(path).convert("L")
#             #converts image to a number representation of each pixel
#             numpyImageAry = numpy.array(imageResize((102,102), grayScale), "uint8")
#             #face detention with numpyArray
#             faces = faceCascade.detectMultiScale(numpyImageAry, scaleFactor=1.5, minNeighbors=5)
#
#             for (x,y,w,h) in faces:
#                 #find our region of interest or the face
#                 roi = numpyImageAry[y:y+h, x:x+w]
#                 xTrain.append(roi)
#                 yLabels.append(id_)
#
#                # fisherXTrain.append(imageResize((250,250), roi))
#                 fisherXTrain.append(cv2.resize(numpyImageAry[y:y+h, x:x+w], (280,280)))

            #print(path)

#write labels to file
with open("faceLabels.pickle", "wb") as file:
    pickle.dump(labelIDs, file)

recognizer.train(xTrain, numpy.array(yLabels))
recognizer.write("trainingFaces.yml")

fisherRecognizer.train(fisherXTrain, numpy.array(yLabels))
fisherRecognizer.write("trainingFisher.yml")
