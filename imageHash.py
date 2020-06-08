import cv2
import os
import numpy as np
import shutil
from pathlib import Path

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

def makeBitsList(mean, pixelsList):
	bitsList = []

	for i in range(len(pixelsList)):
		if (pixelsList[i] >= mean):
			bitsList.append(255)
		else:
			 bitsList.append(0)
	return bitsList

def hashify(squeezedFrame, bitsList):
	bitIndex = 0
	hashedFrame = squeezedFrame

	for x in range(0, squeezedFrame.shape[1], 1):
			for y in range(0, squeezedFrame.shape[0], 1):
				hashedFrame[x, y] = bitsList[bitIndex]
				bitIndex += 1
	return hashedFrame

def hashGeneratorAnimation(frame, hashSize, iterations):

	fourcc = cv2.VideoWriter_fourcc(* "MJPG")
	writer = cv2.VideoWriter(f"static/test.avi", fourcc, 25, (frame.shape[1]*2, frame.shape[0]), True)

	for i in range(iterations):
		if (hashSize >= 16):
			carFrameSqueeze = cv2.resize(frame, (hashSize, hashSize))
			carFrameSqueeze = cv2.cvtColor(carFrameSqueeze, cv2.COLOR_BGR2GRAY)
			pixelsList = grabPixels(carFrameSqueeze)
			meanColor = calculateMean(pixelsList)
			bitsList = makeBitsList(meanColor, pixelsList)
			hashedFrame = hashify(carFrameSqueeze, bitsList)
			hashSize -= 1
			hashedFrame = cv2.cvtColor(hashedFrame, cv2.COLOR_GRAY2BGR)
			hashedFrame = cv2.resize(hashedFrame, (frame.shape[1], frame.shape[0]))
			cv2.putText(hashedFrame, f"hashSize: {hashSize}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4, lineType=cv2.LINE_AA)

			im_v = cv2.hconcat([frame, hashedFrame])

			cv2.imshow("dfs", im_v)
			cv2.waitKey(1)
			writer.write(im_v)

def generateHash(frame, hashSize):

	carFrameSqueeze = cv2.resize(frame, (hashSize, hashSize))
	carFrameSqueeze = cv2.cvtColor(carFrameSqueeze, cv2.COLOR_BGR2GRAY)
	pixelsList = grabPixels(carFrameSqueeze)
	meanColor = calculateMean(pixelsList)
	bitsList = makeBitsList(meanColor, pixelsList)
	hashedFrame = hashify(carFrameSqueeze, bitsList)
	hashedFrame = cv2.cvtColor(hashedFrame, cv2.COLOR_GRAY2BGR)
	hashedFrame = cv2.resize(hashedFrame, (128, 128))

	return bitsList, hashedFrame

def cleanFolder(pathToFolder, hashSize, threshold):
	files = (os.listdir(pathToFolder))
	listLength = len(files)
	sumDiff = 0

	for i in range(len(files)):
		sumDiff = 0

		if (i < len(files)):
			if (files[i] != None):
				carFrame = cv2.imread(f"{pathToFolder}/{files[i]}")
				bitsList, hashedFrame = generateHash(carFrame, hashSize)

			for k in range(i + 1, len(files)):
				if (k < len(files)):
					if (i != k) & (files[k] != None):
						newCarFrame = cv2.imread(f"{pathToFolder}/{files[k]}")
						newBitsList, hashedSecondFrame = generateHash(newCarFrame, hashSize)

						for j in range(hashSize):
							if (bitsList[j] != newBitsList[j]):
								sumDiff += 1

						if (sumDiff < threshold):
							Path(f"images/car/{files[k]}").rename(f"images/copiedCars/{files[k]}")
							print(f"Deleted {k} element ({files[k]}) of {listLength}")

							del files[k]

						print(f"{files[i]} -> {files[k]} sumDiff = {sumDiff}")
						sumDiff = 0

cleanFolder("images/car", 32, 4)

inputFrame = cv2.imread(f"images/harold.jpg")

hashGeneratorAnimation(inputFrame, 512, 512)




