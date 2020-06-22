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
from random import randint
from colorizer import colorize, initNetwork
import time
from sklearn.cluster import MiniBatchKMeans, KMeans
import os
timerStart = 0
timerEnd = 0

alpha_slider_max = 200
blur_slider_max = 100


title_window = "win"
thres1 = 50
thres2 = 50
blurAmount = 5
blurCannyAmount = 5
positionValue = 1
saturationValue = 100

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
motionDetectors = []
grayFrames = []
total = []
classes = []
yoloColors = []

for i in range(len(streamList)):
    vsList.append(VideoStream(streamList[i]))
    frameList.append(None)
    bufferFrames.append(None)
    frameOutList.append(None)
    motionDetectors.append(None)
    grayFrames.append(None)
# vsList[i].start()

yoloNetworkColorizer = initNetwork()
yoloNetworkColorizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yoloNetworkColorizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

img = None
objectIndex = 0
# time.sleep(2.0)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None


def adjustGamma(image, gamma=5.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def initializeYoloNetwork(useCuda):
    yoloNetwork = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    if (useCuda):
        yoloNetwork.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        yoloNetwork.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layers_names = yoloNetwork.getLayerNames()
    outputLayers = [layers_names[i[0] - 1] for i in yoloNetwork.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    return yoloNetwork, layers_names, outputLayers, colors


def initializeRcnnNetwork(useCuda):
    weightsPath = "mask-rcnn-coco/frozen_inference_graph.pb"
    configPath = "mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    rcnnNetwork = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    if (useCuda):
        rcnnNetwork.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        rcnnNetwork.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    return rcnnNetwork


yoloNetwork, layers_names, outputLayers, colors = initializeYoloNetwork(True)
rcnnNetwork = initializeRcnnNetwork(True)


def findYoloClasses(inputFrame, yoloNetwork):
    classesOut = []
    height, width, channels = inputFrame.shape
    blob = cv2.dnn.blobFromImage(
        inputFrame, 0.003, (608, 608), (0, 0, 0), True, crop=False)
    yoloNetwork.setInput(blob)
    outs = yoloNetwork.forward(outputLayers)

    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(class_id)
    # cv2.rectangle(bufferFrames[streamIndex], (x, y), (x + w, y + h), (0,255,0), 2)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.2)

    for i in range(len(boxes)):
        if i in indexes:
            classesOut.append(classIds[i])

    print("=========================")

    return boxes, indexes, classIds, confidences, classesOut


def findRcnnClasses(inputFrame, rcnnNetwork):
    labelsPath = "mask-rcnn-coco/object_detection_classes_coco.txt"
    labels = open(labelsPath).read().strip().split("\n")

    np.random.seed(46)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    classesOut = []
    classesOutCount = [0 for i in range(90)]

    # if not grabbed:
    # 	break

    blob = cv2.dnn.blobFromImage(inputFrame, swapRB=True, crop=False)
    # blob = cv2.dnn.blobFromImage(inputFrame, 0.5, (608, 608), (0, 0, 0), True, crop=False)
    rcnnNetwork.setInput(blob)
    (boxes, masks) = rcnnNetwork.forward(["detection_out_final",
                                          "detection_masks"])

    return boxes, masks, labels, colors


def objectsToTextYolo(inputFrame, boxes, indexes, classIds):
    global objectIndex
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[classIds[i]]
            color = colors[classIds[i]]

            if (x < 0):
                x = 0
            if (y < 0):
                y = 0

            cropImg = inputFrame[y:y + h, x:x + w]
            cropImg = cv2.GaussianBlur(cropImg, (27, 27), 27)

            renderStr = "abcdefghijklmnopqrstuvwxyz0123456789"

            if (x >= 0) & (y >= 0):
                for xx in range(0, cropImg.shape[1], 32):
                    for yy in range(0, cropImg.shape[0], 34):
                        char = randint(0, 1)
                        pixel_b, pixel_g, pixel_r = cropImg[yy, xx]
                        char = renderStr[randint(
                            0, len(renderStr)) - 1]
                        cv2.putText(cropImg, str(char),
                                    (xx, yy),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5,
                                    (int(pixel_b), int(
                                        pixel_g), int(pixel_r)),
                                    3)
            blk = np.zeros(
                inputFrame.shape, np.uint8)

            # if label == "person":
            # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
            cv2.rectangle(
                blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
            inputFrame[y:y +
                         h, x:x + w] = cropImg

            # if (blurPeople == False):
            #     cv2.rectangle(
            #         bufferFrames[streamIndex], (x, y), (x + w, y + h), (255, 255, 255), 2)

            objectIndex += 1

    return inputFrame


def markAllObjectsYolo(inputFrame, boxes, indexes, classIds, confidences):
    global objectIndex
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[classIds[i]]
            color = colors[classIds[i]]

            if (x < 0):
                x = 0
            if (y < 0):
                y = 0

            myStr = "abcdefghijklmnopqrstuvwxyz0123456789"

            blk = np.zeros(
                inputFrame.shape, np.uint8)

            if label == "person":
                cv2.putText(inputFrame, label + "[" + str(np.round(
                    confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                cv2.rectangle(
                    blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                inputFrame = cv2.addWeighted(
                    inputFrame, 1, blk, 0.2, 0)

            if label == "car":
                cv2.putText(inputFrame, label + "[" + str(np.round(
                    confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (213, 160, 47), 2, lineType=cv2.LINE_AA)
                cv2.rectangle(
                    blk, (x, y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
                inputFrame = cv2.addWeighted(
                    inputFrame, 1, blk, 0.2, 0)
            if ((label != "car") & (label != "person")):
                cv2.putText(inputFrame, label + "[" + str(np.round(
                    confidences[i], 2)) + "]", (x, y - 5), font, 0.7, color, 2, lineType=cv2.LINE_AA)
                cv2.rectangle(
                    blk, (x, y), (x + w, y + h), color, cv2.FILLED)
                inputFrame = cv2.addWeighted(
                    inputFrame, 1, blk, 0.2, 0)

            cropImg = inputFrame[y:y + h, x:x + w]

            # cv2.imwrite(f"images/{label}/{label}{str(objectIndex)}.jpg", cropImg)
            # if (blurPeople == False):
            cv2.rectangle(
                inputFrame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            objectIndex += 1

    return inputFrame


def cannyPeopleOnBlackYolo(inputFrame, boxes, indexes, classIds):
    global objectIndex
    inputFrameCopy = inputFrame
    inputFrame = np.zeros(
        (inputFrame.shape[0], inputFrame.shape[1], 3), np.uint8)
    # inputFrame = auto_canny(inputFrame)
    # inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_GRAY2RGB)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[classIds[i]]
            color = colors[classIds[i]]

            if (x < 0):
                x = 0
            if (y < 0):
                y = 0

            cropImg = inputFrameCopy[y:y + h, x:x + w]

            # cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
            cv2.imshow("df", cropImg)
            cropImg = cv2.GaussianBlur(cropImg, (5, 5), 5)
            cropImg = autoCanny(cropImg)
            # cropImg = cv2.Canny(cropImg, 100, 200)
            blank_image = np.zeros(
                (cropImg.shape[0], cropImg.shape[1], 3), np.uint8)

            myStr = "abcdefghijklmnopqrstuvwxyz0123456789"

            blk = np.zeros(
                inputFrame.shape, np.uint8)

            blk2 = np.zeros(
                inputFrame.shape, np.uint8)

            cropImg = cv2.cvtColor(cropImg, cv2.COLOR_GRAY2RGB)
            # src = cropImg
            # tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # _, alpha = cv2.threshold(
            #     tmp, 0, 255, cv2.THRESH_BINARY)
            # b, g, r = cv2.split(src)
            # rgba = [b, g, r, alpha]
            # dst = cv2.merge(rgba, 4)

            # image = cropImg
            mask = np.zeros_like(cropImg)
            rows, cols, _ = mask.shape

            # if label == "person":
            # 	mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(int(cols / 2), int(rows / 2)), angle=0, startAngle=0, endAngle=360, color=(255, 255, 0), thickness=-1)
            # if label == "car":
            # 	mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(int(cols / 2), int(rows / 2)), angle=0, startAngle=0, endAngle=360, color=(255, 0, 255), thickness=-1)
            # if label == "truck":
            # 	mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(int(cols / 2), int(rows / 2)), angle=0, startAngle=0, endAngle=360, color=(255, 0, 255), thickness=-1)
            # if label == "bus":
            # 	mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(int(cols / 2), int(rows / 2)), angle=0, startAngle=0, endAngle=360, color=(255, 0, 255), thickness=-1)
            # if label == "bicycle":
            # 	mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(int(cols / 2), int(rows / 2)), angle=0, startAngle=0, endAngle=360, color=(0, 0, 255), thickness=-1)

            mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(int(cols / 2), int(rows / 2)),
                               angle=0, startAngle=0, endAngle=360, color=(255, 255, 255), thickness=-1)
            result = np.bitwise_and(cropImg, mask)

            result = adjustGamma(result, gamma=0.3)

            mult = (w * h / 15000)

            blk2[y:y + h, x:x + w] = result

            # if (mult<1):
            # 	blk2[blk2 != 0] = 255 * mult

            if label == "person":
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                inputFrame = cv2.ellipse(inputFrame,
                                         center=(x + int(w / 2), y + int(h / 2)),
                                         axes=(int(w / 2), int(h / 2)), angle=0, startAngle=0,
                                         endAngle=360, color=(0, 0, 0), thickness=-1)
                inputFrame = cv2.addWeighted(inputFrame, 1, blk2, 1, 0)

                circleSize = int(w * h / 7000)
                cv2.circle(inputFrame, (x + int(w / 2), y - int(h / 5)), 2, (0, 0, 255), circleSize)

            if label == "car":
                inputFrame = cv2.ellipse(inputFrame,
                                         center=(x + int(w / 2), y + int(h / 2)),
                                         axes=(int(w / 2), int(h / 2)), angle=0, startAngle=0,
                                         endAngle=360, color=(0, 0, 0), thickness=-1)
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                inputFrame = cv2.addWeighted(inputFrame, 1, blk2, 1, 0)
                # bufferFrames[streamIndex] = cv2.addWeighted(bufferFrames[streamIndex], 1, blk2, 1, 1)
                circleSize = int(w * h / 7000)
                cv2.circle(inputFrame, (x + int(w / 2), y - int(h / 5)), 2, (0, 0, 255), circleSize)
            if label == "truck":
                inputFrame = cv2.ellipse(inputFrame,
                                         center=(x + int(w / 2), y + int(h / 2)),
                                         axes=(int(w / 2), int(h / 2)), angle=0, startAngle=0,
                                         endAngle=360, color=(0, 0, 0), thickness=-1)
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                inputFrame = cv2.addWeighted(inputFrame, 1, blk2, 1, 0)
                # bufferFrames[streamIndex] = cv2.addWeighted(bufferFrames[streamIndex], 1, blk2, 1, 1)
                circleSize = int(w * h / 7000)
                cv2.circle(inputFrame, (x + int(w / 2), y - int(h / 5)), 2, (0, 0, 255), circleSize)
            if label == "bus":
                inputFrame = cv2.ellipse(inputFrame,
                                         center=(x + int(w / 2), y + int(h / 2)),
                                         axes=(int(w / 2), int(h / 2)), angle=0, startAngle=0,
                                         endAngle=360, color=(0, 0, 0), thickness=-1)
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                inputFrame = cv2.addWeighted(inputFrame, 1, blk2, 1, 0)
                # bufferFrames[streamIndex] = cv2.addWeighted(bufferFrames[streamIndex], 1, blk2, 1, 1)
                circleSize = int(w * h / 7000)
                cv2.circle(inputFrame, (x + int(w / 2), y - int(h / 5)), 2, (0, 0, 255), circleSize)
            if label == "bicycle":
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                inputFrame = cv2.ellipse(inputFrame,
                                         center=(x + int(w / 2), y + int(h / 2)),
                                         axes=(int(w / 2), int(h / 2)), angle=0, startAngle=0,
                                         endAngle=360, color=(0, 0, 0), thickness=-1)
                inputFrame = cv2.addWeighted(inputFrame, 1, blk2, 1, 0)
                circleSize = int(w * h / 7000)
                cv2.circle(inputFrame, (x + int(w / 2), y - int(h / 5)), 2, (0, 0, 255), circleSize)

            if (label != "person" and label != "car" and label != "truck" and label != "bus"):
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                inputFrame = cv2.ellipse(inputFrame,
                                         center=(x + int(w / 2), y + int(h / 2)),
                                         axes=(int(w / 2), int(h / 2)), angle=0, startAngle=0,
                                         endAngle=360, color=(0, 0, 0), thickness=-1)
                inputFrame = cv2.addWeighted(inputFrame, 1, blk2, 1, 0)
                circleSize = int(w * h / 7000)
                cv2.circle(inputFrame, (x + int(w / 2), y - int(h / 5)), 2, (0, 0, 255), circleSize)
            # if (blurPeople == False):
            #     cv2.rectangle(
            #         bufferFrames[streamIndex], (x, y), (x + w, y + h), (255, 255, 255), 2)

            objectIndex += 1

    return inputFrame


def cannyPeopleOnBackgroundYolo(inputFrame, boxes, indexes, classIds):
    global objectIndex

    # inputFrameCopy = inputFrame
    # inputFrame = np.zeros(
    # 	(inputFrame.shape[0], inputFrame.shape[1], 3), np.uint8)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[classIds[i]]
            color = colors[classIds[i]]

            if (x < 0):
                x = 0
            if (y < 0):
                y = 0

            cropImg = inputFrame[y:y + h, x:x + w]

            # cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
            cv2.imshow("df", cropImg)
            cropImg = cv2.GaussianBlur(cropImg, (5, 5), 5)
            cropImg = autoCanny(cropImg)
            # cropImg = cv2.Canny(cropImg, 100, 200)
            blank_image = np.zeros(
                (cropImg.shape[0], cropImg.shape[1], 3), np.uint8)

            myStr = "abcdefghijklmnopqrstuvwxyz0123456789"

            blk = np.zeros(
                inputFrame.shape, np.uint8)

            blk2 = np.zeros(
                inputFrame.shape, np.uint8)

            cropImg = cv2.cvtColor(cropImg, cv2.COLOR_GRAY2RGB)

            mask = np.zeros_like(cropImg)
            rows, cols, _ = mask.shape
            mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(int(cols / 2), int(rows / 2)),
                               angle=0, startAngle=0, endAngle=360, color=(255, 0, 255), thickness=-1)
            result = np.bitwise_and(cropImg, mask)
            # result = adjust_gamma(result, gamma=0.3)

            mult = (w * h / 20000)

            if (mult < 1):
                result[result != 0] = 255 * mult

            blk2[y:y + h, x:x + w] = result

            if label == "person":
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                inputFrame = cv2.addWeighted(inputFrame, 1, blk2, 1, 0)
                circleSize = int(w * h / 7000)
                cv2.circle(inputFrame, (x + int(w / 2), y - int(h / 5)), 2, (0, 0, 255), circleSize)

            # if label == "car":
            #     # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
            #     # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
            #     inputFrame = cv2.addWeighted(inputFrame, 1, blk2, 1, 0)
            #     circleSize = int(w * h / 7000)
            #     cv2.circle(inputFrame, (x + int(w / 2), y - int(h / 5)), 2, (0, 0, 255), circleSize)
            #
            # if label == "bicycle":
            #     # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
            #     # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
            #     inputFrame = cv2.addWeighted(inputFrame, 1, blk2, 1, 0)
            #     circleSize = int(w * h / 7000)
            #     cv2.circle(inputFrame, (x + int(w / 2), y - int(h / 5)), 2, (0, 0, 255), circleSize)
            else:
                inputFrame = cv2.addWeighted(inputFrame, 1, blk2, 1, 0)
                circleSize = int(w * h / 7000)
                cv2.circle(inputFrame, (x + int(w / 2), y - int(h / 5)), 2, (0, 255, 0), circleSize)
            objectIndex += 1

    return inputFrame


def extractAndCutBackgroundRcnn(inputFrame, boxes, masks, labels):
    classesOut = []
    frameCanny = autoCanny(inputFrame)
    frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2RGB)
    inputFrame = np.zeros(inputFrame.shape, np.uint8)
    frameCanny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > 0.1:
            classesOut.append(classID)

            (H, W) = inputFrame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            boxW = endX - startX
            boxH = endY - startY

            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)

            mask = (mask > 0.1)

            if (labels[classID] == "person"):
                frm = frameCanny[startY:endY, startX:endX][mask]
                frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)
                inputFrame[startY:endY, startX:endX][mask] = frm
            else:
                frm = frameCanny[startY:endY, startX:endX][mask]
                frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)
                inputFrame[startY:endY, startX:endX][mask] = frm

    frameOut = inputFrame

    return frameOut


def extractAndReplaceBackgroundRcnn(inputFrame, frameBackground, boxes, masks, labels, colors):
    classesOut = []
    frameCopy = inputFrame

    inputFrame = cv2.resize(inputFrame, (1280, 720))
    frameCopy = cv2.resize(frameCopy, (1280, 720))
    inputFrame = cv2.GaussianBlur(inputFrame, (5, 5), 5)
    frameCanny = autoCanny(inputFrame)

    frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2RGB)
    frameOut = np.zeros(inputFrame.shape, np.uint8)
    inputFrame = np.zeros(inputFrame.shape, np.uint8)
    frameCanny *= np.array((1, 0, 1), np.uint8)

    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > 0.5:
            classesOut.append(classID)

            (H, W) = inputFrame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            boxW = endX - startX
            boxH = endY - startY

            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            mask = (mask > 0.1)

            color = colors[classID]

            if (labels[classID] == "person"):
                frm = frameCanny[startY:endY, startX:endX][mask]
                frm[np.all(frm == (255, 0, 255), axis=-1)] = (255, 255, 0)
                inputFrame[startY:endY, startX:endX][mask] = frm
            if (labels[classID] == "car"):
                frm = frameCanny[startY:endY, startX:endX][mask]
                frm[np.all(frm == (255, 0, 255), axis=-1)] = (255, 0, 255)
                inputFrame[startY:endY, startX:endX][mask] = frm
            # else:
            #     frm = frameCanny[startY:endY, startX:endX][mask]
            #     frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)
            #     inputFrame[startY:endY, startX:endX][mask] = frm

    # text = "{}[{:.2f}]".format(LABELS[classID], confidence)
    # fontSize = (np.sqrt(boxW * boxH) / 200)
    # cv2.putText(frameCanny, text, (startX, startY - 50),
    # cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,255), 2)

    frameOut = cv2.addWeighted(inputFrame, 1, frameBackground, 1, 0)
    return frameOut


def colorCannyRcnn(inputFrame, boxes, masks, labels):
    classesOut = []
    # frameCanny = autoCanny(inputFrame)
    inputFrame = cv2.GaussianBlur(inputFrame, (5, 5), 5)

    # frameCanny = cv2.Canny(inputFrame, 50,100)
    frameCanny = autoCanny(inputFrame, 0)
    frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2BGR)
    frameOut = np.zeros(inputFrame.shape, np.uint8)

    inputFrame = np.zeros(inputFrame.shape, np.uint8)
    frameCanny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > 0.5:
            classesOut.append(classID)

            (H, W) = inputFrame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            boxW = endX - startX
            boxH = endY - startY

            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            mask = (mask > 0.1)

            if (labels[classID] == "person"):
                frm = frameCanny[startY:endY, startX:endX][mask]
                frm[np.all(frm == (255, 255, 0), axis=-1)] = (255, 0, 255)
                inputFrame[startY:endY, startX:endX][mask] = frm
            if (labels[classID] == "car"):
                frm = frameCanny[startY:endY, startX:endX][mask]
                frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 0)
                inputFrame[startY:endY, startX:endX][mask] = frm
            # if (labels[classID] == "truck"):
            #     frm = frameCanny[startY:endY, startX:endX][mask]
            #     frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 0)
            #     inputFrame[startY:endY, startX:endX][mask] = frm
            # if (labels[classID] == "bus"):
            #     frm = frameCanny[startY:endY, startX:endX][mask]
            #     frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 0)
            #     inputFrame[startY:endY, startX:endX][mask] = frm

    # frameOut = cv2.addWeighted(inputFrame, 1, frameCanny, 1, 0)
    frameCanny = cv2.GaussianBlur(frameCanny, (13, 13), 13)

    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > 0.5:
            classesOut.append(classID)

            (H, W) = inputFrame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            boxW = endX - startX
            boxH = endY - startY

            if (labels[classID] == "person"):
                text = "{}[{:.2f}]".format(labels[classID], confidence)
                fontSize = (np.sqrt(boxW * boxH) / 200)
                cv2.putText(frameCanny, text, (startX, startY - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 255), 2)
            if (labels[classID] == "car"):
                text = "{}[{:.2f}]".format(labels[classID], confidence)
                fontSize = (np.sqrt(boxW * boxH) / 200)
                cv2.putText(frameCanny, text, (startX, startY - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 0, 255), 2)
            if (labels[classID] == "truck"):
                text = "{}[{:.2f}]".format(labels[classID], confidence)
                fontSize = (np.sqrt(boxW * boxH) / 200)
                cv2.putText(frameCanny, text, (startX, startY - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 0, 255), 2)
            if (labels[classID] == "bus"):
                text = "{}[{:.2f}]".format(labels[classID], confidence)
                fontSize = (np.sqrt(boxW * boxH) / 200)
                cv2.putText(frameCanny, text, (startX, startY - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 0, 255), 2)

    frameOut = np.bitwise_or(inputFrame, frameCanny)
    return frameOut


def colorCannyOnColorBackgroundRcnn(inputFrame, boxes, masks, labels):
    classesOut = []
    frameCanny = autoCanny(inputFrame)
    frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2RGB)
    frameCopy = inputFrame
    frameOut = np.zeros(inputFrame.shape, np.uint8)
    frameCanny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > 0.1:
            classesOut.append(classID)

            (H, W) = inputFrame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            boxW = endX - startX
            boxH = endY - startY

            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            mask = (mask > 0.5)

            if (labels[classID] == "person"):
                frm = frameCanny[startY:endY, startX:endX][mask]
                frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)
                inputFrame[startY:endY, startX:endX][mask] = frm
            else:
                frm = frameCanny[startY:endY, startX:endX][mask]
                frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 0)
                inputFrame[startY:endY, startX:endX][mask] = frm

    # frameOut = cv2.addWeighted(inputFrame, 1, frameCanny, 1, 0)
    # frameOut = np.bitwise_xor(inputFrame, frameCanny)
    frameOut = inputFrame
    return frameOut


def colorizerPeopleRcnn(inputFrame, boxes, masks):
    classesOut = []
    needGRAY2BGR = True
    alreadyBGR = False
    frameCopy = inputFrame

    hsvImg = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2HSV)
    hsvImg[..., 1] = hsvImg[..., 1] * 1.1
    # hsvImg[...,2] = hsvImg[...,2]*0.6
    frameCopy = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

    inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)
    # inputFrame = cv2.GaussianBlur(inputFrame, (19, 19), 19)
    frameCanny = autoCanny(inputFrame)
    frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2RGB)
    frameCanny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > 0.5:
            classesOut.append(classID)

            (H, W) = inputFrame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            boxW = endX - startX
            boxH = endY - startY

            smallerX = int(boxW / 30)
            smallerY = int(boxH / 50)

            if (smallerX % 2 != 0):
                smallerX += 1
            if (smallerY % 2 != 0):
                smallerY += 1

            if (boxW > smallerX):
                boxW -= smallerX
            else:
                smallerX = 0

            if (boxH > smallerY):
                boxH -= smallerY
            else:
                smallerY = 0

            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            mask = (mask > 0.2)

            frm = frameCopy[startY + int(smallerY / 2): endY - int(smallerY / 2),
                  startX + int(smallerX / 2): endX - int(smallerX / 2)][mask]
            frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)

            if (alreadyBGR == False):
                inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_GRAY2BGR)
                alreadyBGR = True

            inputFrame[startY + int(smallerY / 2):endY - int(smallerY / 2),
            startX + int(smallerX / 2):endX - int(smallerX / 2)][mask] = frm

            needGRAY2BGR = False

    # text = "{}[{:.2f}]".format(LABELS[classID], confidence)
    # cv2.putText(frameCanny, text, (startX, startY - 50),
    # cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,255), 2)

    if needGRAY2BGR:
        inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_GRAY2RGB)

    frameOut = inputFrame
    return frameOut


def colorizerPeopleRcnnWithBlur(inputFrame, boxes, masks):
    classesOut = []
    needGRAY2BGR = True
    alreadyBGR = False
    frameCopy = inputFrame

    inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)
    inputFrame = cv2.GaussianBlur(inputFrame, (17, 17), 17)
    frameCanny = autoCanny(inputFrame)
    frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2RGB)

    frameCanny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > 0.4:
            classesOut.append(classID)

            (H, W) = inputFrame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            boxW = endX - startX
            boxH = endY - startY

            smallerX = int(boxW / 10)
            smallerY = int(boxH / 10)
            # smallerX = 0
            # smallerY = 0

            if (smallerX % 2 != 0):
                smallerX += 1
            if (smallerY % 2 != 0):
                smallerY += 1

            if (boxW > smallerX):
                boxW -= smallerX
            else:
                smallerX = 0

            if (boxH > smallerY):
                boxH -= smallerY
            else:
                smallerY = 0

            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            mask = (mask > 0.1)

            frm = frameCopy[startY + int(smallerY / 2): endY - int(smallerY / 2),
                  startX + int(smallerX / 2): endX - int(smallerX / 2)][mask]
            frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)

            if (alreadyBGR == False):
                inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_GRAY2BGR)
                alreadyBGR = True

            inputFrame[startY + int(smallerY / 2):endY - int(smallerY / 2),
            startX + int(smallerX / 2):endX - int(smallerX / 2)][mask] = frm

            needGRAY2BGR = False

    # text = "{}[{:.2f}]".format(LABELS[classID], confidence)
    # cv2.putText(frameCanny, text, (startX, startY - 50),
    # cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,255), 2)

    if needGRAY2BGR:
        inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_GRAY2RGB)

    frameOut = inputFrame
    return frameOut


def PeopleRcnnWithBlur(inputFrame, boxes, masks, labels):
    classesOut = []
    needGRAY2BGR = True
    alreadyBGR = False
    frameCopy = inputFrame
    # inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)
    inputFrame = cv2.GaussianBlur(inputFrame, (17, 17), 17)
    frameCanny = autoCanny(inputFrame)
    frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2RGB)

    frameCanny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > 0.4:
            classesOut.append(classID)

            (H, W) = inputFrame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            boxW = endX - startX
            boxH = endY - startY

            smallerX = int(boxW / 10)
            smallerY = int(boxH / 10)
            # smallerX = 0
            # smallerY = 0

            if (smallerX % 2 != 0):
                smallerX += 1
            if (smallerY % 2 != 0):
                smallerY += 1

            if (boxW > smallerX):
                boxW -= smallerX
            else:
                smallerX = 0

            if (boxH > smallerY):
                boxH -= smallerY
            else:
                smallerY = 0

            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            mask = (mask > 0.1)

            # if (labels[classID] == "person"):
            frm = frameCopy[startY + int(smallerY / 2): endY - int(smallerY / 2),
                  startX + int(smallerX / 2): endX - int(smallerX / 2)][mask]
            frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)
            inputFrame[startY + int(smallerY / 2):endY - int(smallerY / 2),
            startX + int(smallerX / 2):endX - int(smallerX / 2)][mask] = frm
            # if (alreadyBGR == False):
            #     inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_GRAY2BGR)
            #     alreadyBGR = True

            needGRAY2BGR = False

    # text = "{}[{:.2f}]".format(LABELS[classID], confidence)
    # cv2.putText(frameCanny, text, (startX, startY - 50),
    # cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,255), 2)

    # if needGRAY2BGR:
    #     inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_GRAY2RGB)

    frameOut = inputFrame
    return frameOut

def sharpening(inputFrame):
    kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    inputFrame = cv2.filter2D(inputFrame, -1, kernel_sharpening)  
    return inputFrame

def denoise(inputFrame):    
    b,g,r = cv2.split(inputFrame)           # get b,g,r
    inputFrame = cv2.merge([r,g,b])     # switch it to rgb
    dst = cv2.fastNlMeansDenoisingColored(inputFrame,None,10,20,7,21)
    b,g,r = cv2.split(dst) 
    inputFrame = cv2.merge([r,g,b])
    return inputFrame

def morphEdgeDetection(inputFrame):
    morph = inputFrame.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # take morphological gradient
    gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
    # split the gradient image into channels
    image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
    channel_height, channel_width, _ = image_channels[0].shape
    # apply Otsu threshold to each channel
    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))
    # merge the channels
    image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
    # save the denoised image
    #cv2.imwrite('output.jpg', image_channels)

    image_channels = cv2.cvtColor(image_channels, cv2.COLOR_BGR2GRAY)
    image_channels = cv2.cvtColor(image_channels, cv2.COLOR_GRAY2BGR)
    image_channels = cv2.bitwise_not(image_channels)
    return image_channels

def ProcessFrame():
    global cap, sourceImage, sourceMode, lock, writer, frameProcessed, progress, fps, frameBackground, totalFrames, outputFrame, colors, classIds, blurAmount, blurCannyAmount, positionValue, saturationValue, videoResetCommand,  startedRenderingVideo

    r = cv2.getTrackbarPos("R", "Controls")
    g = cv2.getTrackbarPos("G", "Controls")
    b = cv2.getTrackbarPos("B", "Controls")

    frameProcessed = 0
    fileIterator = 0
    totalFrames = 0

    usingYoloNeuralNetwork = False
    usingCaffeNeuralNetwork = False
    usingMaskRcnnNetwork = False   

    saveOnlyWithPeople = False
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
    cannyFull = True
    showAllObjects = False
    textRender = False

    font = cv2.FONT_HERSHEY_SIMPLEX
    workingOn = True
    fileToRender = args["source"]
    options = args["optionsList"]
    sourceMode = args["mode"]
    concated = None

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
            cannyFull = True
            print("cannyFull")
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

    if (sourceMode == "video"):
        cap = cv2.VideoCapture(fileToRender)
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap2 = cv2.VideoCapture("snow.webm")

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

                    # frameList[streamIndex] = cv2.resize(frameList[streamIndex], (800,600))
                    # bufferFrames[streamIndex] = cv2.resize(bufferFrames[streamIndex], (800,600))

                    if usingYoloNeuralNetwork:
                        boxes, indexes, classIds, confidences, classesOut = findYoloClasses(bufferFrames[streamIndex],
                                                                                            yoloNetwork)
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
                                                                            boxes, masks)

                        if (colorObjectsOnGrayBlur):
                            bufferFrames[streamIndex] = colorizerPeopleRcnnWithBlur(bufferFrames[streamIndex],
                                                                                    boxes, masks
                                                                                    )
                        if (colorObjectsBlur):
                            bufferFrames[streamIndex] = PeopleRcnnWithBlur(bufferFrames[streamIndex],
                                                                           boxes, masks, labels
                                                                           )

                        if (extractAndCutBackground):
                            bufferFrames[streamIndex] = extractAndCutBackgroundRcnn(bufferFrames[streamIndex],
                                                                                    boxes, masks, labels
                                                                                    )

                        if (extractAndReplaceBackground):
                            bufferFrames[streamIndex] = extractAndReplaceBackgroundRcnn(bufferFrames[streamIndex],
                                                                                        frameBackground,
                                                                                        boxes, masks, labels, colors)

                        if (applyColorCanny):
                            bufferFrames[streamIndex] = colorCannyRcnn(bufferFrames[streamIndex],
                                                                       boxes, masks, labels)

                        if (applyColorCannyOnBackground):
                            bufferFrames[streamIndex] = colorCannyOnColorBackgroundRcnn(bufferFrames[streamIndex], boxes,
                                                                                        masks, labels)

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


                    if cannyFull:
                        # bufferFrames[streamIndex] = autoCanny(bufferFrames[streamIndex])
                        #bufferFrames[streamIndex] = colorize(yoloNetworkColorizer, bufferFrames[streamIndex])
                        #
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
                         
                        #bufferFrames[streamIndex] = cv2.bilateralFilter(bufferFrames[streamIndex], 19, 175, 175)
                          

                        #bufferFrames[streamIndex] = morphEdgeDetection(bufferFrames[streamIndex])

                        bufferFrames[streamIndex] = cv2.Canny(bufferFrames[streamIndex], thres1, thres2)
                        bufferFrames[streamIndex] = cv2.cvtColor(bufferFrames[streamIndex], cv2.COLOR_GRAY2BGR)
                        

                        cv2.imshow("videof",  bufferFrames[streamIndex])
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord("q"):
                            break


# Limit COLORS ====================================================
                        # (B, G, R) = cv2.split(frameCopy)
                        # M = np.maximum(np.maximum(R, G), B) - 50
                        # R[R < M] = 0
                        # G[G < M] = 0
                        # B[B < M] = 0
                        # frameCopy =  cv2.merge([B, G, R])
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
                        #                     frameCopy[i, j, 0] += 50
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
                       
                        
                        kernel = np.ones((3,3),np.uint8)
                        bufferFrames[streamIndex] = cv2.dilate(bufferFrames[streamIndex],kernel,iterations = 1)
                        
                        #                         
                        
                        frameCopy[np.where((bufferFrames[streamIndex] > [50, 50, 50]).all(axis=2))] = [0,0,0]
                        #frameCopy[np.where(bufferFrames[streamIndex] <= 255)] = bufferFrames[streamIndex][np.where(bufferFrames[streamIndex] <= 255)]
                        #frameCopy = cv2.addWeighted(frameCopy, 1, bufferFrames[streamIndex], 1, 0)
                        #frameCopy = np.bitwise_or(frameCopy, bufferFrames[streamIndex])
                        #frameCopy[bufferFrames[streamIndex] > [0, 0, 0]] = cv2.bitwise_not(bufferFrames[streamIndex][bufferFrames[streamIndex]>0]   ) 
                        

# BRIGHTNESS AND CONTRAST =============================================================
                        # alpha = 0.7  # Contrast control (1.0-3.0)
                        # beta = 0  # Brightness control (0-100)
                        #
                        # frameCopy = cv2.convertScaleAbs(frameCopy, alpha=alpha,
                        #                                 beta=beta)
# BRIGHTNESS AND CONTRAST =============================================================

# AMP COLORS ==========================================================================
                        saturation = saturationValue / 100
                        hsv = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2HSV)
                        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation)
                        hsv[:, :, 2] = cv2.multiply(hsv[:, :, -1], 1)
                        frameCopy = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# AMP COLORS ==========================================================================

# LIMIT COLORS WITH KMEANS ============================================================
                        # (h, w) = frameCopy.shape[:2]
                        # frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2LAB)
                        # frameCopy = frameCopy.reshape((frameCopy.shape[0] * frameCopy.shape[1], 3))
                        #
                        # clt = MiniBatchKMeans(n_clusters=32)
                        # labels = clt.fit_predict(frameCopy)
                        # quant = clt.cluster_centers_.astype("uint8")[labels]
                        #
                        # quant = quant.reshape((h, w, 3))
                        # frameCopy = frameCopy.reshape((h, w, 3))
                        #
                        # quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                        # frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_LAB2BGR)
                        #
                        # frameCopy = quant
# LIMIT COLORS WITH KMEANS ============================================================
                        #frameCopy = cv2.GaussianBlur(frameCopy, (3, 3), 2)
                        bufferFrames[streamIndex] = frameCopy
                        
                        # bufferFrames[streamIndex][np.where((bufferFrames[streamIndex] == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
                        # bufferFrames[streamIndex] = bufferFrames[streamIndex] + 10
                        # bufferFrames[streamIndex] = np.bitwise_or(bufferFrames[streamIndex], frameCopy)
                        # bufferFrames[streamIndex] += frameCopy


                        bufferFrames[streamIndex] = cv2.blur(bufferFrames[streamIndex], (2, 2))

                    with lock:
                        personDetected = False
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

def autoCanny(image: object, sigma: object = 0.33) -> object:
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


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
@app.route('/<device>/<action>')
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
    global sourceMode

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
        'frameHeight': frameHeightToPage
        # 'time': datetime.datetime.now().strftime("%H:%M:%S"),
    })
    
    print(timerStart + "////" + timerEnd)

@app.route('/update2', methods=['POST'])
def sendCommand():
    global blurCannyAmount, positionValue, saturationValue, videoResetCommand, startedRenderingVideo, timerStart, timerEnd
    
    if request.method == 'POST':
        timerStart = time.perf_counter()
        inputData = request.get_json()
        blurCannyAmount = int(inputData["sliderValue"])
        positionValue = int(inputData["positionSliderValue"])
        saturationValue = int(inputData["saturationSliderValue"])
        videoResetCommand = int(inputData["videoResetCommand"])

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
