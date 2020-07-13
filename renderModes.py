import cv2
import numpy as np
from random import randint
from sklearn.cluster import MiniBatchKMeans, KMeans
import os

classes = []
objectIndex = 0

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def initializeYoloNetwork(classes, useCuda):
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

def findYoloClasses(inputFrame, yoloNetwork, outputLayers, confidenceValue):
    classesOut = []
    height, width, channels = inputFrame.shape
    blob = cv2.dnn.blobFromImage(
        inputFrame, 0.003, (608, 608), (0, 0, 0), True, crop=False)
    yoloNetwork.setInput(blob)
    outs = yoloNetwork.forward(outputLayers)

    classIds = []
    confidences = []
    boxes = []
    localConfidence = confidenceValue
    confidenceValue = confidenceValue / 100

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidenceValue:
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

    #print("=========================")

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

    #inputFrame = cv2.resize(inputFrame, (128,128))
    blob = cv2.dnn.blobFromImage(inputFrame, swapRB=True, crop=False)
    #blob = cv2.dnn.blobFromImage(inputFrame, 0.1, (608, 608), (0, 0, 0), True, crop=False)
    # blob = cv2.dnn.blobFromImage(inputFrame, 0.5, (608, 608), (0, 0, 0), True, crop=False)
    rcnnNetwork.setInput(blob)
    (boxes, masks) = rcnnNetwork.forward(["detection_out_final",
                                          "detection_masks"])
              
    return boxes, masks, labels, colors

def objectsToTextYolo(inputFrame, boxes, indexes, classIds, fontSize, asciiDistance, blurValue, asciiThicknessValue):
    global objectIndex
    fontSize /= 10
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[classIds[i]]
            color = colorsYolo[classIds[i]]

            if (x < 0):
                x = 0
            if (y < 0):
                y = 0

            cropImg = inputFrame[y:y + h, x:x + w]
            cropImg = cv2.GaussianBlur(cropImg, (blurValue, blurValue), blurValue)

            renderStr = "abcdefghijklmnopqrstuvwxyz0123456789"

            if (x >= 0) & (y >= 0):
                for xx in range(0, cropImg.shape[1], asciiDistance):
                    for yy in range(0, cropImg.shape[0], asciiDistance):
                        char = randint(0, 1)
                        pixel_b, pixel_g, pixel_r = cropImg[yy, xx]
                        char = renderStr[randint(
                            0, len(renderStr)) - 1]
                        cv2.putText(cropImg, str(char),
                                    (xx, yy),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    fontSize,
                                    (int(pixel_b), int(
                                        pixel_g), int(pixel_r)),
                                    asciiThicknessValue)
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
   
def markAllObjectsYolo(inputFrame, boxes, indexes, classIds, confidences, zipArchive, zipIsOpened, zippedImages, sourceMode, startedRenderingMode):
    global objectIndex

    frameCopy = inputFrame.copy()
    cropList = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[classIds[i]]
            color = colorsYolo[classIds[i]]

            if (x < 0):
                x = 0
            if (y < 0):
                y = 0

            myStr = "abcdefghijklmnopqrstuvwxyz0123456789"

            blk = np.zeros(
                inputFrame.shape, np.uint8)

            cropImg = frameCopy[y:y + h, x:x + w]

            cv2.rectangle(
                inputFrame, (x, y), (x + w, y + h), (255, 255, 255), 2)

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
            
            if (startedRenderingMode and zipIsOpened and sourceMode in ("video", "youtube")):           
                cv2.imwrite(f"static/{label}{str(objectIndex)}.jpg", cropImg)
                zipArchive.write(f"static/{label}{str(objectIndex)}.jpg")
                os.remove(f"static/{label}{str(objectIndex)}.jpg")

            if (startedRenderingMode and zipIsOpened and sourceMode == "image" and zippedImages == False):           
                cv2.imwrite(f"static/{label}{str(objectIndex)}.jpg", cropImg)
                zipArchive.write(f"static/{label}{str(objectIndex)}.jpg")
                os.remove(f"static/{label}{str(objectIndex)}.jpg")
            # if (blurPeople == False):
           

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
            color = colorsYolo[classIds[i]]

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
            color = colorsYolo[classIds[i]]

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

def extractAndCutBackgroundRcnn(inputFrame, boxes, masks, labels, confidenceValue):    
    confidenceValue /= 100
    classesOut = []
    frameCanny = autoCanny(inputFrame)
    frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2RGB)
    inputFrame = np.zeros(inputFrame.shape, np.uint8)
    frameCanny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confidenceValue:
            classesOut.append(classID)

            (H, W) = inputFrame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            boxW = endX - startX
            boxH = endY - startY

            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)

            mask = (mask > 0.1)

            #if (labels[classID] == "person"):
            frm = frameCanny[startY:endY, startX:endX][mask]
            frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)
            inputFrame[startY:endY, startX:endX][mask] = frm
            #else:
                # frm = frameCanny[startY:endY, startX:endX][mask]
                # frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)
                # inputFrame[startY:endY, startX:endX][mask] = frm

    frameOut = inputFrame

    return frameOut

def extractAndReplaceBackgroundRcnn(inputFrame, frameBackground, boxes, masks, labels, colors, confidenceValue):
    confidenceValue /= 100
    classesOut = []
    frameCopy = inputFrame

    frameBackground = cv2.resize(frameBackground, (inputFrame.shape[1], inputFrame.shape[0]))
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

        if confidence > confidenceValue:
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

            #if (labels[classID] == "person"):
            frm = frameCanny[startY:endY, startX:endX][mask]
            frm[np.all(frm == (255, 0, 255), axis=-1)] = (255, 255, 0)
            inputFrame[startY:endY, startX:endX][mask] = frm
            #if (labels[classID] == "car"):
                # frm = frameCanny[startY:endY, startX:endX][mask]
                # frm[np.all(frm == (255, 0, 255), axis=-1)] = (255, 0, 255)
                # inputFrame[startY:endY, startX:endX][mask] = frm
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

def colorCannyRcnn(inputFrame, boxes, masks, labels, confidenceValue, rcnnBlurValue):
    confidenceValue /= 100
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

        if confidence > confidenceValue:
            classesOut.append(classID)

            (H, W) = inputFrame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            boxW = endX - startX
            boxH = endY - startY

            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            mask = (mask > 0.1)

            #if (labels[classID] == "person"):
            frm = frameCanny[startY:endY, startX:endX][mask]
            frm[np.all(frm == (255, 255, 0), axis=-1)] = (255, 0, 255)
            inputFrame[startY:endY, startX:endX][mask] = frm
            # if (labels[classID] == "car"):
            #     frm = frameCanny[startY:endY, startX:endX][mask]
            #     frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 0)
            #     inputFrame[startY:endY, startX:endX][mask] = frm
            # if (labels[classID] == "truck"):
            #     frm = frameCanny[startY:endY, startX:endX][mask]
            #     frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 0)
            #     inputFrame[startY:endY, startX:endX][mask] = frm
            # if (labels[classID] == "bus"):
            #     frm = frameCanny[startY:endY, startX:endX][mask]
            #     frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 0)
            #     inputFrame[startY:endY, startX:endX][mask] = frm

    # frameOut = cv2.addWeighted(inputFrame, 1, frameCanny, 1, 0)
    frameCanny = cv2.GaussianBlur(frameCanny, (rcnnBlurValue, rcnnBlurValue), rcnnBlurValue)

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

def colorCannyOnColorBackgroundRcnn(inputFrame, boxes, masks, labels, confidenceValue):
    confidenceValue /= 100
    classesOut = []
    frameCanny = autoCanny(inputFrame)
    frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2RGB)
    frameCopy = inputFrame
    frameOut = np.zeros(inputFrame.shape, np.uint8)
    frameCanny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confidenceValue:
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

def colorizerPeopleRcnn(inputFrame, boxes, masks, confidenceValue, rcnnSizeValue):
    confidenceValue /= 100
    classesOut = []
    needGRAY2BGR = True
    alreadyBGR = False
    frameCopy = inputFrame

    # hsvImg = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2HSV)
    # hsvImg[..., 1] = hsvImg[..., 1] * 1.1
    # # hsvImg[...,2] = hsvImg[...,2]*0.6
    # frameCopy = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

    inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)
    # inputFrame = cv2.GaussianBlur(inputFrame, (19, 19), 19)
    frameCanny = autoCanny(inputFrame)
    frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2RGB)
    frameCanny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confidenceValue:
            classesOut.append(classID)

            (H, W) = inputFrame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            boxW = endX - startX
            boxH = endY - startY

            if (rcnnSizeValue == 0):
                rcnnSizeValue = 2

            smallerX = int(boxW / rcnnSizeValue)
            smallerY = int(boxH / rcnnSizeValue)

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

def colorizerPeopleRcnnWithBlur(inputFrame, boxes, masks, confidenceValue):
    confidenceValue /= 100
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

        if confidence > confidenceValue:
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

def PeopleRcnnWithBlur(inputFrame, boxes, masks, labels, confidenceValue, rcnnSizeValue, rcnnBlurValue):
    confidenceValue /= 100

    classesOut = []
    needGRAY2BGR = True
    alreadyBGR = False
    frameCopy = inputFrame
    # inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)
    inputFrame = cv2.GaussianBlur(inputFrame, (rcnnBlurValue, rcnnBlurValue), rcnnBlurValue)
    frameCanny = autoCanny(inputFrame)
    frameCanny = cv2.cvtColor(frameCanny, cv2.COLOR_GRAY2RGB)

    frameCanny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confidenceValue:
            classesOut.append(classID)

            (H, W) = inputFrame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            boxW = endX - startX
            boxH = endY - startY

            if (rcnnSizeValue == 0):
                rcnnSizeValue = 2


            smallerX = int(boxW / rcnnSizeValue)
            smallerY = int(boxH / rcnnSizeValue)
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

def asciiPaint(inputFrame, fontSize, asciiDistance, asciiThicknessValue, blurValue):
    fontSize /= 10

    inputFrame = cv2.GaussianBlur(inputFrame, (blurValue,blurValue), blurValue)

    blk = np.zeros(
        inputFrame.shape, np.uint8)

    renderStr = "abcdefghijklmnopqrstuvwxyz0123456789"
    
    for xx in range(0, inputFrame.shape[1], asciiDistance):
        for yy in range(0, inputFrame.shape[0], asciiDistance):
            char = randint(0, 1)
            pixel_b, pixel_g, pixel_r = inputFrame[yy, xx]
            char = renderStr[randint(0, len(renderStr)) - 1]
            cv2.putText(blk, str(char), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (int(pixel_b), int(pixel_g), int(pixel_r)), asciiThicknessValue, lineType=cv2.LINE_AA)
    
    
    return blk

def sharpening(inputFrame, sharpeningValue, sharpeningValue2):
    kernelValue = sharpeningValue2
    kernelDiff = 9 - kernelValue        
    kernel_sharpening = np.array([[-1, -1, -1], [-1, kernelValue, -1], [-1, -1, -1]])

    while (kernelDiff != 0):
        for i in range(3):
            for j in range(3):
                if (i == 1 and j == 1):
                    kernel_sharpening[j][i] == kernelValue
                else:
                    if (kernelDiff > 0):
                        kernel_sharpening[j][i] += 1
                        kernelDiff -= 1
                    if (kernelDiff < 0):
                        kernel_sharpening[j][i] -= 1
                        kernelDiff += 1
                    if (kernelDiff == 0):
                        break
                
    inputFrame = cv2.filter2D(inputFrame, -1, kernel_sharpening)  

    inputFrame = cv2.detailEnhance(inputFrame, sigma_s=sharpeningValue, sigma_r=0.15)

    return inputFrame

def denoise(inputFrame, denoiseValue, denoiseValue2):    
    if (denoiseValue2 > 0):
        b,g,r = cv2.split(inputFrame)           # get b,g,r
        inputFrame = cv2.merge([r,g,b])     # switch it to rgb
        dst = cv2.fastNlMeansDenoisingColored(inputFrame,None,denoiseValue2,denoiseValue,7,15)
        b,g,r = cv2.split(dst) 
        inputFrame = cv2.merge([r,g,b])
        
    return inputFrame

def morphEdgeDetection(inputFrame):
    morph = inputFrame.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))    
    gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)   
    image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
    channel_height, channel_width, _ = image_channels[0].shape
    
    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))
  
    image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)   
    image_channels = cv2.cvtColor(image_channels, cv2.COLOR_BGR2GRAY)
    image_channels = cv2.cvtColor(image_channels, cv2.COLOR_GRAY2BGR)
    image_channels = cv2.bitwise_not(image_channels)
    return image_channels

def limitColorsKmeans(inputFrame, colorCount): 
    if (colorCount > 0):   
        (h, w) = inputFrame.shape[:2]
        inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2LAB)
        inputFrame = inputFrame.reshape((inputFrame.shape[0] * inputFrame.shape[1], 3))    
        clt = MiniBatchKMeans(n_clusters=colorCount)
        labels = clt.fit_predict(inputFrame)
        quant = clt.cluster_centers_.astype("uint8")[labels]    
        quant = quant.reshape((h, w, 3))
        inputFrame = inputFrame.reshape((h, w, 3))    
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_LAB2BGR)  
        inputFrame = quant
    
    return inputFrame

def autoCanny(image: object, sigma: object = 0.33) -> object:
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

def adjustGamma(image, gamma=5.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjustSaturation(inputFrame, saturation = 1):
    saturation = saturation / 100
    hsv = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation)
    hsv[:, :, 2] = cv2.multiply(hsv[:, :, -1], 1)
    inputFrame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return inputFrame

def adjustBrContrast(inputFrame, contrastValue, brightnessValue):
    contrastValue = contrastValue / 100   
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)                
    inputFrame = cv2.convertScaleAbs(inputFrame, alpha=contrastValue,
                                    beta=brightnessValue)
    return inputFrame

yoloNetwork, layers_names, outputLayers, colorsYolo = initializeYoloNetwork(classes, True)
rcnnNetwork = initializeRcnnNetwork(False)