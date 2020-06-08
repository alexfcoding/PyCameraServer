import cv2
import os
import numpy as np

def calculateMean(pixelsList):
    sum = 0
    mean = 0

    for i in range(len(pixelsList)):
        sum = sum + pixelsList[i]

    mean = sum / len(pixelsList)
    return mean

def grabPixels(squeezedFrame):
    pixelsList = []

    for x in range(0, squeezedFrame.shape[1], 1):
        for y in range(0, squeezedFrame.shape[0], 1):
            pixelColor = squeezedFrame[x, y]
            pixelsList.append(pixelColor)

    return pixelsList

#print(f"Pixels: {pixelsList}")
#print(f"MEAN: {meanColor}")

def makeBitsList(mean, pixelsList):
    bitsList = []

    for i in range(len(pixelsList)):
        if (pixelsList[i] >= mean):
            bitsList.append(255)
        else:
             bitsList.append(0)
    return bitsList

#print(f"BITS: {bitsList}")

#outputFrame = np.zeros((8,8,1), np.uint8)

def hashify(squeezedFrame, bitsList):
    bitIndex = 0
    hashedFrame = squeezedFrame

    for x in range(0, squeezedFrame.shape[1], 1):
            for y in range(0, squeezedFrame.shape[0], 1):
                hashedFrame[x, y] = bitsList[bitIndex]
                bitIndex += 1
    return hashedFrame

def hashGenerator(frame, hashSize, iterations):

    for i in range(iterations):
        if (hashSize > 2):
            carFrameSqueeze = cv2.resize(carFrame, (hashSize, hashSize))
            carFrameSqueeze = cv2.cvtColor(carFrameSqueeze, cv2.COLOR_BGR2GRAY)

            pixelsList = grabPixels(carFrameSqueeze)
            meanColor = calculateMean(pixelsList)
            bitsList = makeBitsList(meanColor, pixelsList)

            hashedFrame = hashify(carFrameSqueeze, bitsList)

            cv2.imshow(f"HashSize: {hashSize}", cv2.resize(hashedFrame, (128, 128)))
            cv2.waitKey(0)

            hashSize -= 2

carFiles = []
carFiles = (os.listdir('images/car'))
carFrame = cv2.imread(f"images/car/{carFiles[9]}")

hashGenerator(carFrame, 8, 1)
#for car in range(len(carFiles)):
    #print (carFiles[car])


