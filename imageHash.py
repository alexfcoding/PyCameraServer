import cv2
import os
import numpy as np
import shutil
from pathlib import Path
import argparse
import time

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--inputFolder", type=str, required=True,
				help="folder to clean similar images")
ap.add_argument("-o", "--outputFolder", type=str, required=True,
				help="folder to move similar images")
ap.add_argument("-s", "--hashSize", type=int, default=32,
				help="hash size")
ap.add_argument("-t", "--threshold", type=int, required=True,
				help="threshold for detecting similar images")

args = vars(ap.parse_args())

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
			frameSqueezed = cv2.resize(frame, (hashSize, hashSize))
			frameSqueezed = cv2.cvtColor(frameSqueezed, cv2.COLOR_BGR2GRAY)
			pixelsList = grabPixels(frameSqueezed)
			meanColor = calculateMean(pixelsList)
			bitsList = makeBitsList(meanColor, pixelsList)
			hashedFrame = hashify(frameSqueezed, bitsList)
			hashedFrame = cv2.cvtColor(hashedFrame, cv2.COLOR_GRAY2BGR)
			#hashedFrame = cv2.resize(hashedFrame, (frame.shape[1], frame.shape[0]))
			cv2.putText(hashedFrame, f"hashSize: {hashSize}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4, lineType=cv2.LINE_AA)

			im_v = cv2.hconcat([frame, hashedFrame])
			cv2.imshow("dfs", im_v)
			cv2.waitKey(1)
			writer.write(im_v)
			hashSize -= 1

def generateHash(frame, hashSize):
	frameSqueezed = cv2.resize(frame, (hashSize, hashSize))
	frameSqueezed = cv2.cvtColor(frameSqueezed, cv2.COLOR_BGR2GRAY)
	pixelsList = grabPixels(frameSqueezed)
	meanColor = calculateMean(pixelsList)
	bitsList = makeBitsList(meanColor, pixelsList)
	hashedFrame = hashify(frameSqueezed, bitsList)
	hashedFrame = cv2.cvtColor(hashedFrame, cv2.COLOR_GRAY2BGR)
	#hashedFrame = cv2.resize(hashedFrame, (128, 128))

	return bitsList, hashedFrame

def cleanFolder(inputFolder, outputFolder, hashSize, threshold):
	files = (os.listdir(inputFolder))
	listLength = len(files)
	sumDiff = 0
	i = 0
	k = 1
	frame = None
	hashedFrame = None

	while (i < len(files)):
		sumDiff = 0

		if (files[i] != None):
			#frame = cv2.imread(f"{inputFolder}/person3763.jpg")
			frame = cv2.imread(f"{inputFolder}/{files[i]}")
			#frame = cv2.GaussianBlur(frame, (13,13),13)
			bitsList, hashedFrame = generateHash(frame, hashSize)

		while (k < len(files)):
			if (i != k) & (files[k] != None):
				newFrame = cv2.imread(f"{inputFolder}/{files[k]}")
				#newFrame = cv2.GaussianBlur(newFrame, (13,13),13)
				newBitsList, hashedSecondFrame = generateHash(newFrame, hashSize)

				for j in range(len(bitsList)):
					if (bitsList[j] != newBitsList[j]):
						sumDiff += 1

				print(f"{files[i]} -> {files[k]} sumDiff = {sumDiff}")

				if (sumDiff < threshold):
					Path(f"{inputFolder}/{files[k]}").rename(f"{outputFolder}/{files[k]}")
					print(f"Deleted {k} element ({files[k]}) of {listLength}")

					del files[k]

					cv2.putText(hashedSecondFrame, f"FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, lineType=cv2.LINE_AA)
					im_h = cv2.hconcat([cv2.resize(frame, (450, 450)), cv2.resize(newFrame, (450, 450))])
					im_h2 = cv2.hconcat([cv2.resize(hashedFrame, (450, 450)), cv2.resize(hashedSecondFrame, (450, 450))])
					im_v = cv2.vconcat([im_h, im_h2])
					cv2.imshow("dfs", im_v)
					cv2.waitKey(1)

				else:
					cv2.putText(hashedSecondFrame, f"NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, lineType=cv2.LINE_AA)
					im_h = cv2.hconcat([cv2.resize(frame, (450, 450)), cv2.resize(newFrame, (450, 450))])
					im_h2 = cv2.hconcat([cv2.resize(hashedFrame, (450, 450)), cv2.resize(hashedSecondFrame, (450, 450))])
					im_v = cv2.vconcat([im_h, im_h2])
					cv2.imshow("dfs", im_v)
					cv2.waitKey(1)

					k += 1

				sumDiff = 0
		i += 1
		k = i + 1

cleanFolder(args['inputFolder'], args['outputFolder'], args['hashSize'], args['threshold'])

#inputFrame = cv2.imread(f"images/harold.jpg")
#inputFrame = cv2.imread(f"images/car/car1066.jpg")

#bitsList, outputHash = generateHash(inputFrame, 16)

# cv2.imshow("hash", outputHash)
# cv2.waitKey(0)
#hashGeneratorAnimation(inputFrame, 512, 512)

#Example: python imageHash.py -i images/all_images/ -o images/s -s 16 -t 60