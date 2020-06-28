# python localFiles.py -i 192.168.0.12 -o 8002 -s fabio.webm -c a -m video

from imutils.video import VideoStream
from flask import jsonify
from flask import Flask
from flask import render_template
import threading
import argparse
import numpy as np
import cv2
import time
from flask import request, Response
import psutil

from colorizer import colorize, initNetwork
import time
from sklearn.cluster import MiniBatchKMeans, KMeans
import os
import renderModes
from renderModes import *

timerStart = 0
timerEnd = 0

alpha_slider_max = 200
blur_slider_max = 100

title_window = "win"


cv2.namedWindow(title_window)

def on_trackbar(val):
    global thres1
    thres1 = val

def on_trackbar2(val):
    global thres2
    thres2 = val

def on_trackbar3(val):
    global blurAmount
    blurAmount = val

def on_trackbar4(val):
    global blurCannyAmount
    blurCannyAmount = val

trackbar_name = 'Alpha x %d' % alpha_slider_max
cv2.createTrackbar(trackbar_name, title_window , 0, alpha_slider_max, on_trackbar)

trackbar_name2 = 'Alpha2 x %d' % alpha_slider_max
cv2.createTrackbar(trackbar_name2, title_window , 0, alpha_slider_max, on_trackbar2)

trackbar_name3 = 'Alpha3 x %d' % alpha_slider_max
cv2.createTrackbar(trackbar_name3, title_window , 0, blur_slider_max, on_trackbar3)

trackbar_name4 = 'Alpha4 x %d' % alpha_slider_max
cv2.createTrackbar(trackbar_name4, title_window , 0, blur_slider_max, on_trackbar4)

thr = None
workingOn = True
outputFrame = None
resized = None
value = 0
running = False
progress = 0
fps = 0
cap = None
cap2 = None
videoResetCommand = False
startedRenderingVideo = False

lock = threading.Lock()
A = 0

app = Flask(__name__, static_url_path='/static')

streamList = [
    "videoplayback.mp4"
]

# Working adresses:
# http://94.72.19.58/mjpg/video.mjpg,
# http://91.209.234.195/mjpg/video.mjpg
# http://209.194.208.53/mjpg/video.mjpg
# http://66.57.117.166:8000/mjpg/video.mjpg

frameList = []
bufferFrames = []
frameOutList = []
vsList = []
grayFrames = []
total = []


for i in range(len(streamList)):
    vsList.append(VideoStream(streamList[i]))
    frameList.append(None)
    bufferFrames.append(None)
    frameOutList.append(None)
    grayFrames.append(None)
# vsList[i].start()

yoloNetworkColorizer = initNetwork()
yoloNetworkColorizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yoloNetworkColorizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

img = None

# time.sleep(2.0)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None

    
def checkIfUserIsConnected(timerStart):
    timerEnd = time.perf_counter()

    print(str(timerStart) + "///////" + str(timerEnd))
    
    if (timerEnd - timerStart < 5 and timerStart != 0):
        print("User is connected")
    else:
        if (timerStart != 0):
            print("User disconnected, need shutdown !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            currentPid = os.getpid()
            p = psutil.Process(currentPid)
            p.terminate()  #or p.kill()
            #shutdown_server()

def ProcessFrame():
    global cap, sourceImage, sourceMode, lock, writer, frameProcessed, progress, fps, frameBackground, totalFrames, outputFrame, colors, classIds, blurAmount, blurCannyAmount, positionValue, saturationValue, contrastValue, brightnessValue, lineThicknessValue, videoResetCommand,  startedRenderingVideo, needModeReset, options

    r = cv2.getTrackbarPos("R", "Controls")
    g = cv2.getTrackbarPos("G", "Controls")
    b = cv2.getTrackbarPos("B", "Controls")

    frameProcessed = 0
    fileIterator = 0
    totalFrames = 0
    needModeReset = True

    usingYoloNeuralNetwork = False
    usingCaffeNeuralNetwork = False
    usingMaskRcnnNetwork = False     

    blurPeople = False
    cannyPeopleOnBackground = False
    cannyPeopleOnBlack = False
    cannyPeopleRCNN = False
    extractAndReplaceBackground = False
    extractAndCutBackground = False
    applyColorCanny = False
    applyColorCannyOnBackground = False
    colorObjectsOnGrayBlur = False
    colorObjectsBlur = False
    colorObjectsOnGray = False
    videoColorization = False
    imageUpscaler = False
    cartoonEffect = False
    showAllObjects = False
    textRender = False

    font = cv2.FONT_HERSHEY_SIMPLEX
    workingOn = True
    fileToRender = args["source"]
    options = args["optionsList"]
    sourceMode = args["mode"]
    concated = None

    if (sourceMode == "video"):
        cap = cv2.VideoCapture(fileToRender)
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap2 = cv2.VideoCapture("inputVideos/snow.webm")

    if (sourceMode == "image"):
        sourceImage = args["source"]
        cap2 = cv2.VideoCapture("space.webm")

    # while True:
    # 	# grab the current frame
    # 	(grabbed, frame) = cap.read()
    
    # 	if not grabbed:
    # 		break
    
    # 	totalFrames = totalFrames + 1
    # 	print(totalFrames)

    lineType = cv2.LINE_AA

    while workingOn:
        
        #print("working...")
        if (needModeReset):
            usingYoloNeuralNetwork = False
            usingCaffeNeuralNetwork = False
            usingMaskRcnnNetwork = False
            blurPeople = False
            cannyPeopleOnBackground = False
            cannyPeopleOnBlack = False
            cannyPeopleRCNN = False
            extractAndReplaceBackground = False
            extractAndCutBackground = False
            applyColorCanny = False
            applyColorCannyOnBackground = False
            colorObjectsOnGrayBlur = False
            colorObjectsBlur = False
            colorObjectsOnGray = False
            videoColorization = False
            imageUpscaler = False
            cartoonEffect = False
            showAllObjects = False
            textRender = False

            for char in options:
                if (char == "a"):
                    showAllObjects = True
                    usingYoloNeuralNetwork = True
                    print("showAllObjects")
                if (char == "b"):
                    textRender = True
                    usingYoloNeuralNetwork = True
                    print("textRender")
                if (char == "c"):
                    cannyPeopleOnBlack = True
                    usingYoloNeuralNetwork = True
                    print("cannyPeopleOnBlack")
                if (char == "d"):
                    cannyPeopleOnBackground = True
                    usingYoloNeuralNetwork = True
                    print("cannyPeopleOnBackground")
                if (char == "e"):
                    cartoonEffect = True
                    print("cartoonEffect")
                if (char == "f"):
                    videoColorization = True
                    usingCaffeNeuralNetwork = True
                    print("videoColorization")
                if (char == "g"):
                    usingMaskRcnnNetwork = True
                    extractAndCutBackground = True
                    print("cannyPeopleRCNN + cut background")
                if (char == "h"):
                    usingMaskRcnnNetwork = True
                    applyColorCannyOnBackground = True
                    print("applyColorCannyOnBackground")
                if (char == "i"):
                    usingMaskRcnnNetwork = True
                    extractAndReplaceBackground = True
                    print("cannyPeopleRCNN + replace background")
                if (char == "j"):
                    usingMaskRcnnNetwork = True
                    applyColorCanny = True
                    print("applyColorCanny")
                if (char == "k"):
                    usingMaskRcnnNetwork = True
                    colorObjectsOnGray = True
                    print("colorObjectsOnGray")
                if (char == "l"):
                    usingMaskRcnnNetwork = True
                    colorObjectsOnGrayBlur = True
                    print("colorObjectsOnGrayBlur")
                if (char == "m"):
                    usingMaskRcnnNetwork = True
                    colorObjectsBlur = True
                    print("colorObjectsOnGrayBlur")
                if (char == "n"):
                    imageUpscaler = True
                    print("imageUpscaler")

                needModeReset = False

        classesIndex = []
        startMoment = time.time()

        for streamIndex in range(len(streamList)):
            if (sourceMode == "video"):

                if (startedRenderingVideo == False):
                    cap.set(1,positionValue)
                else:
                    if (videoResetCommand == True):
                        cap.set(1,1)
                        frameProcessed = 0
                        cap.release()
                        writer.release()
                        cap = cv2.VideoCapture(fileToRender)
                        writer = cv2.VideoWriter(f"static/output{args['port']}.avi"
                                                        f"", fourcc, 25, (
                                bufferFrames[streamIndex].shape[1], bufferFrames[streamIndex].shape[0]), True)
                        
                ret, frameList[streamIndex] = cap.read()
                ret2, frameBackground = cap2.read()
                
            if (sourceMode == "image"):
                frameList[streamIndex] = cv2.imread(sourceImage)
                ret2, frameBackground = cap2.read()

            if frameList[streamIndex] is not None:
                bufferFrames[streamIndex] = frameList[streamIndex].copy()

                frameList[streamIndex] = cv2.resize(frameList[streamIndex], (1280,720))
                bufferFrames[streamIndex] = cv2.resize(bufferFrames[streamIndex], (1280,720))

                if usingYoloNeuralNetwork:
                    boxes, indexes, classIds, confidences, classesOut = findYoloClasses(bufferFrames[streamIndex],
                                                                                        yoloNetwork, outputLayers, confidenceValue)
                    classesIndex.append(classesOut)

                    if showAllObjects:
                        bufferFrames[streamIndex] = markAllObjectsYolo(bufferFrames[streamIndex], boxes, indexes,
                                                                        classIds, confidences)

                    if textRender:
                        bufferFrames[streamIndex] = objectsToTextYolo(bufferFrames[streamIndex], boxes, indexes,
                                                                        classIds)

                    if cannyPeopleOnBlack:
                        bufferFrames[streamIndex] = cannyPeopleOnBlackYolo(bufferFrames[streamIndex], boxes, indexes,
                                                                            classIds)

                    if cannyPeopleOnBackground:
                        bufferFrames[streamIndex] = cannyPeopleOnBackgroundYolo(bufferFrames[streamIndex], boxes,
                                                                                indexes, classIds)

                if usingMaskRcnnNetwork:
                    boxes, masks, labels, colors = findRcnnClasses(bufferFrames[streamIndex], rcnnNetwork)

                    if (colorObjectsOnGray):
                        bufferFrames[streamIndex] = colorizerPeopleRcnn(bufferFrames[streamIndex],
                                                                        boxes, masks, confidenceValue)

                    if (colorObjectsOnGrayBlur):
                        bufferFrames[streamIndex] = colorizerPeopleRcnnWithBlur(bufferFrames[streamIndex],
                                                                                boxes, masks, confidenceValue
                                                                                )
                    if (colorObjectsBlur):
                        bufferFrames[streamIndex] = PeopleRcnnWithBlur(bufferFrames[streamIndex],
                                                                        boxes, masks, labels, confidenceValue
                                                                        )

                    if (extractAndCutBackground):
                        bufferFrames[streamIndex] = extractAndCutBackgroundRcnn(bufferFrames[streamIndex],
                                                                                boxes, masks, labels, confidenceValue
                                                                                )

                    if (extractAndReplaceBackground):
                        bufferFrames[streamIndex] = extractAndReplaceBackgroundRcnn(bufferFrames[streamIndex],
                                                                                    frameBackground,
                                                                                    boxes, masks, labels, colors, confidenceValue)

                    if (applyColorCanny):
                        bufferFrames[streamIndex] = colorCannyRcnn(bufferFrames[streamIndex],
                                                                    boxes, masks, labels, confidenceValue)

                    if (applyColorCannyOnBackground):
                        bufferFrames[streamIndex] = colorCannyOnColorBackgroundRcnn(bufferFrames[streamIndex], boxes,
                                                                                    masks, labels, confidenceValue)

                if usingCaffeNeuralNetwork:
                    if videoColorization:
                        bufferFrames[streamIndex] = colorize(yoloNetworkColorizer, bufferFrames[streamIndex])

                # if (imageUpscaler):
                    # kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])

                    
                    # b,g,r = cv2.split(bufferFrames[streamIndex])           # get b,g,r
                    # bufferFrames[streamIndex] = cv2.merge([r,g,b])     # switch it to rgb

                    # # Denoising
                    # dst = cv2.fastNlMeansDenoisingColored(bufferFrames[streamIndex],None,15,20,10,21)

                    # b,g,r = cv2.split(dst)           # get b,g,r
                    # bufferFrames[streamIndex] = cv2.merge([r,g,b])     # switch it to rgb

                    # bufferFrames[streamIndex] = cv2.filter2D(bufferFrames[streamIndex], -1, kernel_sharpening)          

                if cartoonEffect:
                    #bufferFrames[streamIndex] = denoise(bufferFrames[streamIndex])
                    frameCopy = bufferFrames[streamIndex].copy()                        
                    #frameCopy = sharpening(frameCopy)                        
                    
                    if (blurAmount % 2 == 0):
                        blurAmount += 1
                    else:
                        frameCopy = cv2.GaussianBlur(frameCopy, (blurAmount, blurAmount), blurAmount)

                    if (blurCannyAmount % 2 == 0):
                        blurCannyAmount += 1
                        bufferFrames[streamIndex] = cv2.GaussianBlur(bufferFrames[streamIndex],
                                                                        (blurCannyAmount, blurCannyAmount),
                                                                        blurCannyAmount)                            
                    else:
                        bufferFrames[streamIndex] = cv2.GaussianBlur(bufferFrames[streamIndex],
                                                                        (blurCannyAmount, blurCannyAmount),
                                                                        blurCannyAmount)
                        
                    #bufferFrames[streamIndex] = morphEdgeDetection(bufferFrames[streamIndex])
                    bufferFrames[streamIndex] = cv2.Canny(bufferFrames[streamIndex], thres1, thres2)
                    bufferFrames[streamIndex] = cv2.cvtColor(bufferFrames[streamIndex], cv2.COLOR_GRAY2BGR)
                    
                    cv2.imshow("videof",  bufferFrames[streamIndex])
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break

# Limit COLORS ====================================================
                    (B, G, R) = cv2.split(frameCopy)
                    M = np.maximum(np.maximum(R, G), B) - 70
                    R[R < M] = 0
                    G[G < M] = 0
                    B[B < M] = 0
                    frameCopy =  cv2.merge([B, G, R])
# Limit COLORS ====================================================

                    # for i in range(0, bufferFrames[streamIndex].shape[0] - 3):
                    #     for j in range(0, bufferFrames[streamIndex].shape[1] - 3):
                    #         #if (all(bufferFrames[streamIndex][i, j] > [100,0,0])):
                    #             diffBG = 0
                    #             diffBR = 0
                    #
                    #             diffGB = 0
                    #             diffGR = 0
                    #
                    #             diffRB = 0
                    #             diffRG = 0
                    #
                    #             if (frameCopy[i, j, 0] > frameCopy[i, j, 1]):
                    #                 diffBG = frameCopy[i, j, 0] - frameCopy[i, j, 1]
                    #             if (frameCopy[i, j, 0] > frameCopy[i, j, 2]):
                    #                 diffBR = frameCopy[i, j, 0] - frameCopy[i, j, 2]
                    #
                    #             if (frameCopy[i, j, 1] > frameCopy[i, j, 0]):
                    #                 diffGB = frameCopy[i, j, 1] - frameCopy[i, j, 0]
                    #             if (frameCopy[i, j, 1] > frameCopy[i, j, 2]):
                    #                 diffGR = frameCopy[i, j, 1] - frameCopy[i, j, 2]
                    #
                    #             if (frameCopy[i, j, 2] > frameCopy[i, j, 0]):
                    #                 diffRB = frameCopy[i, j, 2] - frameCopy[i, j, 0]
                    #             if (frameCopy[i, j, 2] > frameCopy[i, j, 1]):
                    #                 diffRG = frameCopy[i, j, 2] - frameCopy[i, j, 1]
                    #
                    #             if frameCopy[i, j, 0] > frameCopy[i, j, 1] and frameCopy[i, j, 0] > frameCopy[i, j, 2] and diffBG > 20 and diffBR > 20:
                    #                 if (frameCopy[i, j, 0] <= 205):
                    #                     fr2ameCopy[i, j, 0] += 50
                    #             if frameCopy[i, j, 1] > frameCopy[i, j, 0] and frameCopy[i, j, 1] > frameCopy[i, j, 2] and diffGB > 20 and diffGR > 20:
                    #                 if (frameCopy[i, j, 1] <= 205):
                    #                     frameCopy[i, j, 1] += 50
                    #             if frameCopy[i, j, 2] > frameCopy[i, j, 0] and frameCopy[i, j, 2] > frameCopy[i, j, 1] and diffRB > 20 and diffRG > 20:
                    #                 if (frameCopy[i, j, 2] <= 205):
                    #                     frameCopy[i, j, 2] += 50
                    #

                    # if (bufferFrames[streamIndex][i, j, 0] > bufferFrames[streamIndex][i, j, 1]):
                    #     frameCopy[i, j] = [255, 255, 255]
                    #frameCopy[np.where((bufferFrames[streamIndex] == [255, 255, 255]).all(axis=2))] = [0,0,0]

                    #bufferFrames[streamIndex] = cv2.GaussianBlur(bufferFrames[streamIndex], (3, 3), 1)
                    # kernel = np.ones((2, 2), np.float32) / 25
                    # bufferFrames[streamIndex] = cv2.filter2D( bufferFrames[streamIndex], -1, kernel)


                    #bufferFrames[streamIndex] = cv2.bilateralFilter(bufferFrames[streamIndex], 9, 175, 175)
                    #frameCopy[np.where((bufferFrames[streamIndex] > [0, 0, 0]).all(axis=2))] = [0, 0, 0]
                    # frameCopy[np.where((bufferFrames[streamIndex] > [0, 0, 0]).all(axis=2))] = [0, 0, 0]

                    #

                    # clearFrame = bufferFrames[streamIndex].copy()
                    # crop_img = bufferFrames[streamIndex][1:bufferFrames[streamIndex].shape[0], 1:bufferFrames[streamIndex].shape[1]].copy()
                    # clearFrame[0:crop_img.shape[0], 0:crop_img.shape[1]] = crop_img[0:crop_img.shape[0], 0:crop_img.shape[1]].copy()
                    # frameCopy[np.where((clearFrame == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
                    # # #
                    # # #
                    # clearFrame = bufferFrames[streamIndex].copy()
                    # crop_img = bufferFrames[streamIndex][1:bufferFrames[streamIndex].shape[0], 0:bufferFrames[streamIndex].shape[1]].copy()
                    # clearFrame[0:crop_img.shape[0], 0:crop_img.shape[1]] = crop_img[0:crop_img.shape[0], 0:crop_img.shape[1]].copy()
                    # frameCopy[np.where((clearFrame == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
                    # #
                    # clearFrame = bufferFrames[streamIndex].copy()
                    # crop_img = bufferFrames[streamIndex][0:bufferFrames[streamIndex].shape[0], 1:bufferFrames[streamIndex].shape[1]].copy()
                    # clearFrame[0:crop_img.shape[0], 0:crop_img.shape[1]] = crop_img[0:crop_img.shape[0],
                    #                                                        0:crop_img.shape[1]].copy()
                    # frameCopy[np.where((clearFrame == [255, 255, 255]).all(axis=2))] = [0, 0, 0]



                    # clearFrame = bufferFrames[streamIndex].copy()
                    # crop_img = bufferFrames[streamIndex][2:bufferFrames[streamIndex].shape[0],
                    #            0:bufferFrames[streamIndex].shape[1]].copy()
                    # clearFrame[0:crop_img.shape[0], 0:crop_img.shape[1]] = crop_img[0:crop_img.shape[0],
                    #                                                        0:crop_img.shape[1]].copy()
                    # frameCopy[np.where((clearFrame == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
                    #
                    # clearFrame = bufferFrames[streamIndex].copy()
                    # crop_img = bufferFrames[streamIndex][2:bufferFrames[streamIndex].shape[0],
                    #            1:bufferFrames[streamIndex].shape[1]].copy()
                    # clearFrame[0:crop_img.shape[0], 0:crop_img.shape[1]] = crop_img[0:crop_img.shape[0],
                    #                                                        0:crop_img.shape[1]].copy()
                    # frameCopy[np.where((clearFrame == [255, 255, 255]).all(axis=2))] = [0, 0, 0]

                    # clearFrame = bufferFrames[streamIndex].copy()
                    # crop_img = bufferFrames[streamIndex][2:bufferFrames[streamIndex].shape[0],
                    #            2:bufferFrames[streamIndex].shape[1]].copy()
                    # clearFrame[0:crop_img.shape[0], 0:crop_img.shape[1]] = crop_img[0:crop_img.shape[0],
                    #                                                        0:crop_img.shape[1]].copy()
                    # frameCopy[np.where((clearFrame == [255, 255, 255]).all(axis=2))] = [0, 0, 0]


                    #bufferFrames[streamIndex] *= np.array((1, 1, 0), np.uint8)
                    #frameCopy = cv2.GaussianBlur(frameCopy, (13, 13), 13)
                    #bufferFrames[streamIndex] = cv2.blur(bufferFrames[streamIndex], (2, 2))
                    
                    
                    
                    
                    #                         
                    
                    #frameCopy[np.where(bufferFrames[streamIndex] <= 255)] = bufferFrames[streamIndex][np.where(bufferFrames[streamIndex] <= 255)]
                    #frameCopy = cv2.addWeighted(frameCopy, 1, bufferFrames[streamIndex], 1, 0)
                    #frameCopy = np.bitwise_or(frameCopy, bufferFrames[streamIndex])
                    #frameCopy[bufferFrames[streamIndex] > [0, 0, 0]] = cv2.bitwise_not(bufferFrames[streamIndex][bufferFrames[streamIndex]>0]   ) 
                    

                    kernel = np.ones((lineThicknessValue,lineThicknessValue),np.uint8)
                    bufferFrames[streamIndex] = cv2.dilate(bufferFrames[streamIndex],kernel,iterations = 1)
                    frameCopy[np.where((bufferFrames[streamIndex] > [0, 0, 0]).all(axis=2))] = [0,0,0]

                    #frameCopy = limitColorsKmeans(frameCopy)
                    frameCopy = cv2.GaussianBlur(frameCopy, (3, 3), 2)
                    bufferFrames[streamIndex] = frameCopy
                    
                    # bufferFrames[streamIndex][np.where((bufferFrames[streamIndex] == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
                    # bufferFrames[streamIndex] = bufferFrames[streamIndex] + 10
                    # bufferFrames[streamIndex] = np.bitwise_or(bufferFrames[streamIndex], frameCopy)
                    # bufferFrames[streamIndex] += frameCopy
                    # bufferFrames[streamIndex] = cv2.blur(bufferFrames[streamIndex], (2, 2))
# BRIGHTNESS AND CONTRAST =======================o=====================================
                contrast = contrastValue / 100
                brightness = brightnessValue
                alpha = 1  # Contrast control (1.0-3.0)
                beta = 0  # Brightness control (0-100)
                
                bufferFrames[streamIndex] = cv2.convertScaleAbs(bufferFrames[streamIndex], alpha=contrast,
                                                beta=brightness)
# BRIGHTNESS AND CONTRAST =============================================================
# AMP COLORS ==========================================================================
                bufferFrames[streamIndex] = adjustSaturation(bufferFrames[streamIndex], saturationValue)

# AMP COLORS ==========================================================================
                with lock:
                    personDetected = False
                    checkIfUserIsConnected(timerStart)

                    frameProcessed = frameProcessed + 1
                    elapsedTime = time.time()
                    fps = 1 / (elapsedTime - startMoment)
                    #print(fps)

                    for streamIndex in range(len(streamList)):
                        if (showAllObjects):

                            classIndexCount = [
                                [0 for x in range(80)] for x in range(len(streamList))]

                            rowIndex = 0
                            for m in range(80):
                                for k in range(len(classesIndex[streamIndex])):
                                    if (m == classesIndex[streamIndex][k]):
                                        classIndexCount[streamIndex][m] += 1

                                if (classIndexCount[streamIndex][m] != 0):
                                    rowIndex += 1

                                    # cv2.rectangle(bufferFrames[streamIndex], (0, rowIndex*40 - 20), (200,rowIndex*40 + 8), (0,0,0), -1)
                                    # cv2.putText(bufferFrames[streamIndex], classes[m] + ": " + str(classIndexCount[streamIndex][m]), (20,rowIndex*40), font, 0.7, colors[m], 2, cv2.LINE_AA)

                                    if (classes[m] == "person"):
                                        cv2.rectangle(
                                            bufferFrames[streamIndex], (20, rowIndex * 70 - 40),
                                            (400, rowIndex * 70 + 16), (0, 0, 0), -1)
                                        cv2.putText(bufferFrames[streamIndex], classes[m] + ": " + str(
                                            classIndexCount[streamIndex][m]), (40, rowIndex * 70), font, 1.4,
                                                    (0, 255, 0), 2, lineType=cv2.LINE_AA)
                                        personDetected = True
                                    if (classes[m] == "car"):
                                        cv2.rectangle(
                                            bufferFrames[streamIndex], (20, rowIndex * 70 - 40),
                                            (400, rowIndex * 70 + 16), (0, 0, 0), -1)
                                        cv2.putText(bufferFrames[streamIndex], classes[m] + ": " + str(
                                            classIndexCount[streamIndex][m]), (40, rowIndex * 70), font, 1.4,
                                                    (255, 0, 255), 2, lineType=cv2.LINE_AA)
                                    if ((classes[m] != "car") & (classes[m] != "person")):
                                        cv2.rectangle(
                                            bufferFrames[streamIndex], (20, rowIndex * 70 - 40),
                                            (400, rowIndex * 70 + 16), (0, 0, 0), -1)
                                        cv2.putText(bufferFrames[streamIndex], classes[m] + ": " + str(
                                            classIndexCount[streamIndex][m]), (40, rowIndex * 70), font, 1.4,
                                                    colors[m], 2, lineType=cv2.LINE_AA)

                                    if (classes[m] == "handbag") | (classes[m] == "backpack"):
                                        passFlag = True
                                        print("handbag detected! -> PASS")

                        if (writer is None):
                            writer = cv2.VideoWriter(f"static/output{args['port']}.avi"
                                                        f"", fourcc, 25, (
                                bufferFrames[streamIndex].shape[1], bufferFrames[streamIndex].shape[0]), True)
                            # writer = cv2.VideoWriter(f"static/output{args['port']}.avi", fourcc, 25, (
                            #     640, 720), True)
                        else:
                            if (sourceMode == "video"):
                                progress = frameProcessed / totalFrames * 100

                            # cv2.rectangle(bufferFrames[streamIndex], (20, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 80),
                            #               (int(
                            #                   cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 20,
                            #                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 24), (0, 0, 0), -1)
                            #
                            # if (progress != "DONE"):
                            #     cv2.rectangle(bufferFrames[streamIndex],
                            #                   (20, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 80), (int(cap.get(
                            #             cv2.CAP_PROP_FRAME_WIDTH) * progress / 100) - 20, int(
                            #             cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 24), (0, 255, 0), -1)
                            #     cv2.putText(bufferFrames[streamIndex], str(int(progress)) + "%" + " | FPS: " + str(
                            #         round(fps, 2)) + " | " + "CPU: " + str(
                            #         psutil.cpu_percent()) + "%", (40, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 40),
                            #                 font, 1.4, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                            if (sourceMode == "video"):
                                cv2.putText(bufferFrames[streamIndex], f"FPS: {str(round(fps, 2))} {str(bufferFrames[streamIndex].shape[1])}x{str(bufferFrames[streamIndex].shape[0])}", (40, 40),
                                        font, 1.4, (0, 0, 255), 2)
                            # resized1 = cv2.resize(frameList[streamIndex], (640, 320))
                            # resized2 = cv2.resize(bufferFrames[streamIndex], (640, 320))

                            # if (sourceMode == "video") :
                            #     writer.write(bufferFrames[streamIndex])

                            if (sourceMode == "image"):
                                cv2.imwrite(f"static/{sourceImage}", bufferFrames[streamIndex])
                                #workingOn = False
                            if ((sourceMode == "image" and extractAndReplaceBackground == True)):
                                writer.write(bufferFrames[streamIndex])

                            # cv2.imwrite("static/t.jpg",
                            # bufferFrames[streamIndex])
                            resized1 = cv2.resize(frameList[streamIndex], (640, 360))
                            resized2 = cv2.resize(bufferFrames[streamIndex], (640, 360))

                            # im_v = cv2.vconcat([im_h, im_h2])
                            # resized = bufferFrames[streamIndex].copy()
                            #resized = cv2.resize(resized, (1280, 720))
                            concated = cv2.vconcat([resized2, resized1, ])

                            #resized = cv2.resize(concated, (640, 720))
                            resized = cv2.resize(bufferFrames[streamIndex], (1600, 900))

                            if (sourceMode == "video"):
                                writer.write(bufferFrames[streamIndex])

                            cv2.imshow("video",  resized)
                            #cv2.imshow("video",  bufferFrames[streamIndex])
                            key = cv2.waitKey(1) & 0xFF

                            if key == ord("q"):
                                break

                        # outputFrame = concated
                        outputFrame = bufferFrames[streamIndex]
            else:
                resized = cv2.resize(bufferFrames[streamIndex], (1280, 720))
                outputFrame = resized
                workingOn = False
                print("finished")

                if (sourceMode == "video"):
                    cap.release()
                    writer.release()
                    cv2.destroyAllWindows()
                while True:
                    checkIfUserIsConnected(timerStart)


def generate():
    global outputFrame, frameProcessed, lock, workingOn

    while workingOn:

        with lock:
            if outputFrame is None:
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # outputFrame = cv2.resize(outputFrame, (320,240))

            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
    # else:
    # return redirect(url_for("results"))
    # return redirect('/results')
    print("yield finished")

@app.route('/')
def index(device=None, action=None):
    return render_template("index.html", frameProcessed=frameProcessed,
                           pathToRenderedFile=f"static/output{args['port']}.avi")

@app.route("/video")
def video_feed():
    # redirect(f"http://192.168.0.12:8000/results")
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# return Response(stream_with_context(generate()))

@app.route('/update', methods=['POST'])
def update():
    global sourceMode, totalFrames

    timerStart = time.perf_counter()
    frameWidthToPage = 0
    frameHeightToPage = 0

    if (sourceMode == "video"):
        frameWidthToPage = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frameHeightToPage = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    if (sourceMode == "image"):
        frameWidthToPage = 0
        frameHeightToPage = 0  

    return jsonify({
        'value': frameProcessed,
        'totalFrames': totalFrames,
        'progress': round(progress, 2),
        'fps': round(fps, 2),
        'workingOn': workingOn,
        'cpuUsage': psutil.cpu_percent(),
        'freeRam': round((psutil.virtual_memory()[1] / 2. ** 30), 2),
        'ramPercent': psutil.virtual_memory()[2],
        'frameWidth': frameWidthToPage,
        'frameHeight': frameHeightToPage,
        'maxFrames': totalFrames
        # 'time': datetime.datetime.now().strftime("%H:%M:%S"),
    })
    
    print(timerStart + "////" + timerEnd)

@app.route('/update2', methods=['GET','POST'])
def sendCommand():
    global blurCannyAmount, positionValue, saturationValue, contrastValue, brightnessValue, videoResetCommand, startedRenderingVideo, timerStart, timerEnd, modeResetCommand, options, needModeReset, writer, confidenceValue, lineThicknessValue
    
    if request.method == 'POST':
        timerStart = time.perf_counter()
        inputData = request.get_json()
        blurCannyAmount = int(inputData["sliderValue"])
        positionValue = int(inputData["positionSliderValue"])
        saturationValue = int(inputData["saturationSliderValue"])
        contrastValue = int(inputData["contrastSliderValue"])
        brightnessValue = int(inputData["brightnessSliderValue"])
        confidenceValue = int(inputData["confidenceSliderValue"])
        lineThicknessValue = int(inputData["lineThicknessSliderValue"])
        videoResetCommand = int(inputData["videoResetCommand"])
        modeResetCommand = str(inputData["modeResetCommand"])

        if (modeResetCommand != "default"):           
            options = modeResetCommand   
            needModeReset = True                                   

        if (videoResetCommand):
            startedRenderingVideo = True

        print(videoResetCommand)

    return '', 200

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-s", "--source", type=str, default=32,
                    help="# file to render")
    ap.add_argument("-c", "--optionsList", type=str, required=True,
                    help="rendering iptions")
    ap.add_argument("-m", "--mode", type=str, required=True,
                    help="rendering mode: 'video' or 'image'")

    args = vars(ap.parse_args())

    t = threading.Thread(target=ProcessFrame)
    t.daemon = True
    t.start()
    app.run(host=args["ip"], port=args["port"], debug=False,
            threaded=True, use_reloader=False)

for j in range(len(streamList)):
    vsList[j].stop()
