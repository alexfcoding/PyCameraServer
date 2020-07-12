# python localFiles.py -i 192.168.0.12 -o 8002 -s fabio.webm -c a -m video

from imutils.video import VideoStream
from flask import jsonify
from flask import Flask
from flask import render_template
from flask import send_file
import threading
import argparse
import numpy as np
import cv2
import time
from flask import request, Response
import psutil
from colorizer import colorize, initNetwork
from upscaler import initNetworkUpscale, upscaleImage
import time
import os
import renderModes
from renderModes import *
from werkzeug.utils import secure_filename
import uuid
from zipfile import ZipFile
import pafy
import youtube_dl

# url = 'https://youtu.be/ih5KhSAxfUE'
# vPafy = pafy.new(url)
# play = vPafy.streams[2]

class Container:
	pass

timerStart = 0
timerEnd = 0
alpha_slider_max = 200
blur_slider_max = 100
title_window = "win"
userTime = 0
inputData = None

settings = Container()
commands = Container()
serverStates = Container()

settings.thres1 = 50
settings.thres2 = 50
settings.blurAmount = 5
settings.blurCannyAmount = 5
settings.positionValue = 1
settings.saturationValue = 100
settings.contrastValue = 100
settings.brightnessValue = 0
settings.confidenceValue = 0
settings.lineThicknessValue = 1
settings.denoiseValue = 10
settings.denoiseValue2 = 10
settings.sharpeningValue = 9
settings.rcnnSizeValue = 10
settings.rcnnBlurValue = 17
settings.objectIndex = 0
settings.sobelValue = 5
settings.sharpeningValue2 = 5
settings.colorCountValue = 32
settings.resizeValue = 2

commands.videoResetCommand = False
commands.videoStopCommand = False
commands.modeResetCommand = False
commands.screenshotCommand = False

serverStates.sourceImage = ""
serverStates.sourceMode = ""
serverStates.screenshotPath = ""
serverStates.needToCreateScreenshot = False
serverStates.screenshotReady = False
serverStates.workingOn = True
serverStates.frameProcessed = 0
serverStates.totalFrames = 0
serverStates.options = ""
serverStates.screenshotLock = False
serverStates.videoResetLock = False
serverStates.videoStopLock = False

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

outputFrame = None
resized = None
value = 0
running = False
progress = 0
fps = 0
cap = None
cap2 = None

# videoResetCommand = False

lock = threading.Lock()
A = 0

app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

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

caffeNetworkColorizer = initNetwork()
caffeNetworkColorizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
caffeNetworkColorizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
netUpscaler = initNetworkUpscale()

img = None

# time.sleep(2.0)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None
	
def checkIfUserIsConnected(timerStart):
	global userTime
	timerEnd = time.perf_counter()

	#print(str(timerStart) + "================" + str(timerEnd))
	userTime = str(round(timerEnd)) + ":" + str(round(timerStart))

	if (timerEnd - timerStart < 7 and timerStart != 0):
		print("User is connected")
	else:
		if (timerStart != 0):
			print("User disconnected, need shutdown !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			currentPid = os.getpid()
			p = psutil.Process(currentPid)
			p.terminate()  #or p.kill()
			#shutdown_server()

def ProcessFrame():
	global cap, lock, writer, progress, fps, outputFrame, fileToRender, zipObj

	frameBackground = None

	receivedZipCommand = False
	fileChanged = False
	startedRenderingVideo = False	
	# //screenshotReady = False

	r = cv2.getTrackbarPos("R", "Controls")
	g = cv2.getTrackbarPos("G", "Controls")
	b = cv2.getTrackbarPos("B", "Controls")

	serverStates.frameProcessed = 0
	fileIterator = 0
	serverStates.totalFrames = 0
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
	denoiseAndSharpen= False    
	sobel = False	
	cartoonEffect = False
	showAllObjects = False
	textRender = False
	frameUpscale = False
	asciiPainter = False
	pencilDrawer = False
	changedResolution = False

	font = cv2.FONT_HERSHEY_SIMPLEX
	serverStates.workingOn = True
	fileToRender = args["source"]
	serverStates.options = args["optionsList"]
	serverStates.sourceMode = args["mode"]
	concated = None
	resizeValueLocal = 2
	needToCreateNewZip = True
	needToStopNewZip = False
	zipIsOpened = False
	zippedImages = False

	if (serverStates.sourceMode == "video"):
		# cap = cv2.VideoCapture(play.url)
		cap = cv2.VideoCapture(fileToRender)
		#cap = cv2.VideoCapture(0)
		serverStates.totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		cap2 = cv2.VideoCapture("inputVideos/snow.webm")

	if (serverStates.sourceMode == "image"):
		serverStates.sourceImage = args["source"]
		cap2 = cv2.VideoCapture("inputVideos/snow.webm")

	# while True:
	# 	# grab the current frame
	# 	(grabbed, frame) = cap.read()
	
	# 	if not grabbed:
	# 		break
	
	# 	totalFrames = totalFrames + 1
	# 	print(totalFrames)

	lineType = cv2.LINE_AA
	zipObj = ZipFile(f"static/objects{args['port']}.zip", 'w')
	zipIsOpened = True

	while serverStates.workingOn:
		if (inputData is not None):
			settings.blurCannyAmount = int(inputData["sliderValue"])	
			settings.saturationValue = int(inputData["saturationSliderValue"])
			settings.contrastValue = int(inputData["contrastSliderValue"])
			settings.brightnessValue = int(inputData["brightnessSliderValue"])
			settings.confidenceValue = int(inputData["confidenceSliderValue"])
			settings.lineThicknessValue = int(inputData["lineThicknessSliderValue"])
			settings.denoiseValue = int(inputData["denoiseSliderValue"])
			settings.denoiseValue2 = int(inputData["denoise2SliderValue"])
			settings.sharpeningValue = int(inputData["sharpenSliderValue"])
			settings.sharpeningValue2 = int(inputData["sharpenSliderValue2"])
			settings.rcnnSizeValue = int(inputData["rcnnSizeSliderValue"])
			settings.rcnnBlurValue = int(inputData["rcnnBlurSliderValue"])
			settings.sobelValue = int(inputData["sobelSliderValue"])
			settings.asciiSizeValue = int(inputData["asciiSizeSliderValue"])
			settings.asciiIntervalValue = int(inputData["asciiIntervalSliderValue"])
			settings.asciiThicknessValue = int(inputData["asciiThicknessSliderValue"])		
			settings.resizeValue = int(inputData["resizeSliderValue"]) / 100
			settings.colorCountValue = int(inputData["colorCountSliderValue"])
			positionValueLocal = int(inputData["positionSliderValue"])

			if (commands.modeResetCommand != "default"):           
				serverStates.options = commands.modeResetCommand   
				needModeReset = True                                   

			if (serverStates.videoResetLock == True):
				settings.positionValue = 1
				needToCreateWriter = True
				startedRenderingVideo = True
				receivedZipCommand = True
				serverStates.videoResetLock = False
				print('in loop reset')
			else:
				settings.positionValue = positionValueLocal

			if (serverStates.videoStopLock == True):
				settings.positionValue = 1			
				startedRenderingVideo = False
				serverStates.videoStopLock = False
				print('in loop stop')

			if (serverStates.screenshotLock == True):
				print('in loop screenshot')
				serverStates.needToCreateScreenshot = True
				serverStates.screenshotLock = False
			
			

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
			denoiseAndSharpen = False
			sharpener = False
			sobel = False
			asciiPainter = False
			pencilDrawer = False
			twoColored = False
			frameUpscale = False

			for char in serverStates.options:
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
					frameUpscale = True
					print("imageUpscaler")
				if (char == "o"):
					denoiseAndSharpen = True
					print("denoiseAndSharpen") 
				if (char == "p"):
					sobel = True
					print("sobel")                 
				if (char == "q"):
					asciiPainter = True
					print("asciiPainter") 
				if (char == "r"):
					pencilDrawer = True
					print("pencilDrawer")
				if (char == "s"):
					twoColored = True
					print("twoColored")

				needModeReset = False

		classesIndex = []
		startMoment = time.time()

		for streamIndex in range(len(streamList)):
			if (serverStates.sourceMode == "video"):
				if (startedRenderingVideo == False):
					cap.set(1, settings.positionValue)
					if (needToStopNewZip):
						zipObj.close()
						zipIsOpened = False
						needToStopNewZip = False
						needToCreateNewZip = True					
				else:			
					if (needToCreateWriter == True or fileChanged == True):						
						cap.set(1,1)
						serverStates.frameProcessed = 0
						cap.release()
						if (fileChanged):
							print("1")
						if (writer is not None):
							writer.release()

						cap = cv2.VideoCapture(fileToRender)
						# cap = cv2.VideoCapture(play.url)
						
						writer = cv2.VideoWriter(f"static/output{args['port']}{fileToRender}.avi"
													f"", fourcc, 25, (
							bufferFrames[streamIndex].shape[1], bufferFrames[streamIndex].shape[0]), True)

						# print("CREATING WRITER 1 WITH SIZE:" + str(round(bufferFrames[streamIndex].shape[1])))	
						
						if (needToCreateNewZip):
							zipObj = ZipFile(f"static/objects{args['port']}.zip", 'w')
							needToStopNewZip = True
							needToCreateNewZip = False
							zipIsOpened = True

						if (fileChanged):
							zipObj = ZipFile(f"static/objects{args['port']}.zip", 'w')
							zipIsOpened = True

						fileChanged = False
						commands.videoResetCommand = False
						needToCreateWriter = False						

				ret, frameList[streamIndex] = cap.read()
				ret2, frameBackground = cap2.read()
				
			if (serverStates.sourceMode == "image"):	
				if (receivedZipCommand == True or fileChanged == True):	
					zippedImages = False						
					zipObj = ZipFile(f"static/objects{args['port']}.zip", 'w')
					zipIsOpened = True	
					receivedZipCommand = False					

				if (fileChanged):
					zipObj = ZipFile(f"static/objects{args['port']}.zip", 'w')
					zipIsOpened = True
					fileChanged = False	
					needToCreateWriter = False

				frameList[streamIndex] = cv2.imread(serverStates.sourceImage)
				ret2, frameBackground = cap2.read()

			if frameList[streamIndex] is not None:
				bufferFrames[streamIndex] = frameList[streamIndex].copy()

				if usingYoloNeuralNetwork:
					boxes, indexes, classIds, confidences, classesOut = findYoloClasses(bufferFrames[streamIndex],
																						yoloNetwork, outputLayers, settings.confidenceValue)
					classesIndex.append(classesOut)

					if showAllObjects:
						bufferFrames[streamIndex] = markAllObjectsYolo(bufferFrames[streamIndex], boxes, indexes,
																		classIds, confidences, zipObj, zipIsOpened, zippedImages, serverStates.sourceMode, startedRenderingVideo)
					
					if (serverStates.sourceMode == "image" and zipIsOpened):
						zipObj.close()							

					if (serverStates.sourceMode == "image" and zippedImages == False):
							zippedImages = True
							zipIsOpened = False

					if textRender:
						bufferFrames[streamIndex] = objectsToTextYolo(bufferFrames[streamIndex], boxes, indexes,
																		classIds,  settings.asciiSizeValue,  settings.asciiIntervalValue,  settings.rcnnBlurValue,  settings.asciiThicknessValue)

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
																		boxes, masks,  settings.confidenceValue,  settings.rcnnSizeValue)

					if (colorObjectsOnGrayBlur):
						bufferFrames[streamIndex] = colorizerPeopleRcnnWithBlur(bufferFrames[streamIndex],
																				boxes, masks,  settings.confidenceValue
																				)
					if (colorObjectsBlur):
						bufferFrames[streamIndex] = PeopleRcnnWithBlur(bufferFrames[streamIndex],
																		boxes, masks, labels,  settings.confidenceValue,  settings.rcnnSizeValue,  settings.rcnnBlurValue
																		)

					if (extractAndCutBackground):
						bufferFrames[streamIndex] = extractAndCutBackgroundRcnn(bufferFrames[streamIndex],
																				boxes, masks, labels,  settings.confidenceValue
																				)

					if (extractAndReplaceBackground):
						bufferFrames[streamIndex] = extractAndReplaceBackgroundRcnn(bufferFrames[streamIndex],
																					frameBackground,
																					boxes, masks, labels, colors,  settings.confidenceValue)
																			

					if (applyColorCanny):
						bufferFrames[streamIndex] = colorCannyRcnn(bufferFrames[streamIndex],
																	boxes, masks, labels,  settings.confidenceValue,  settings.rcnnBlurValue)

					if (applyColorCannyOnBackground):
						bufferFrames[streamIndex] = colorCannyOnColorBackgroundRcnn(bufferFrames[streamIndex], boxes,
																					masks, labels,  settings.confidenceValue)

				if usingCaffeNeuralNetwork:
					if videoColorization:
						bufferFrames[streamIndex] = colorize(caffeNetworkColorizer, bufferFrames[streamIndex])

				if cartoonEffect:					
					frameCopy = bufferFrames[streamIndex].copy()   
					
					# if (blurAmount % 2 == 0):
					#     blurAmount += 1
					# else:
					#     frameCopy = cv2.GaussianBlur(frameCopy, (blurAmount, blurAmount), blurAmount)

					if (settings.blurCannyAmount % 2 == 0):
						settings.blurCannyAmount += 1
						bufferFrames[streamIndex] = cv2.GaussianBlur(bufferFrames[streamIndex],
																		(settings.blurCannyAmount, settings.blurCannyAmount),
																		settings.blurCannyAmount)                            
					else:
						bufferFrames[streamIndex] = cv2.GaussianBlur(bufferFrames[streamIndex],
																		(settings.blurCannyAmount, settings.blurCannyAmount),
																		settings.blurCannyAmount)
											
					bufferFrames[streamIndex] = cv2.Canny(bufferFrames[streamIndex], settings.thres1, settings.thres2)
					bufferFrames[streamIndex] = cv2.cvtColor(bufferFrames[streamIndex], cv2.COLOR_GRAY2BGR)	
					kernel = np.ones((settings.lineThicknessValue, settings.lineThicknessValue),np.uint8)
					bufferFrames[streamIndex] = cv2.dilate(bufferFrames[streamIndex],kernel,iterations = 1)
					frameCopy[np.where((bufferFrames[streamIndex] > [0, 0, 0]).all(axis=2))] = [0,0,0]
					frameCopy = limitColorsKmeans(frameCopy, settings.colorCountValue)
					#frameCopy = cv2.GaussianBlur(frameCopy, (3, 3), 2)
					bufferFrames[streamIndex] = frameCopy					
					bufferFrames[streamIndex] = sharpening(bufferFrames[streamIndex], settings.sharpeningValue, settings.sharpeningValue2)                    
					bufferFrames[streamIndex] = denoise(bufferFrames[streamIndex], settings.denoiseValue, settings.denoiseValue2)

				if pencilDrawer:					
					frameCopy = bufferFrames[streamIndex].copy()    

					if (settings.blurCannyAmount % 2 == 0):
						settings.blurCannyAmount += 1
						bufferFrames[streamIndex] = cv2.GaussianBlur(bufferFrames[streamIndex],
																		(settings.blurCannyAmount, settings.blurCannyAmount),
																		settings.blurCannyAmount)                            
					else:
						bufferFrames[streamIndex] = cv2.GaussianBlur(bufferFrames[streamIndex],
																		(settings.blurCannyAmount, settings.blurCannyAmount),
																		settings.blurCannyAmount)
						
					#bufferFrames[streamIndex] = morphEdgeDetection(bufferFrames[streamIndex])
					bufferFrames[streamIndex] = cv2.Canny(bufferFrames[streamIndex], settings.thres1, settings.thres2)
					bufferFrames[streamIndex] = cv2.cvtColor(bufferFrames[streamIndex], cv2.COLOR_GRAY2BGR)	
					kernel = np.ones((settings.lineThicknessValue, settings.lineThicknessValue),np.uint8)
					bufferFrames[streamIndex] = cv2.dilate(bufferFrames[streamIndex],kernel,iterations = 1)
					frameCopy[np.where((bufferFrames[streamIndex] > [0, 0, 0]).all(axis=2))] = [0,0,0]
					frameCopy = limitColorsKmeans(frameCopy, 2)
					#frameCopy = cv2.GaussianBlur(frameCopy, (3, 3), 2)
					bufferFrames[streamIndex] = frameCopy
					bufferFrames[streamIndex] = sharpening(bufferFrames[streamIndex], settings.sharpeningValue, settings.sharpeningValue2)                    
					bufferFrames[streamIndex] = denoise(bufferFrames[streamIndex], settings.denoiseValue, settings.denoiseValue2)  
					#bufferFrames[streamIndex] = np.bitwise_not(bufferFrames[streamIndex])
# Limit COLORS ====================================================
					# (B, G, R) = cv2.split(frameCopy)
					# M = np.maximum(np.maximum(R, G), B) - 70
					# R[R < M] = 0
					# G[G < M] = 0
					# B[B < M] = 0
					# frameCopy =  cv2.merge([B, G, R])
# Limit COLORS ====================================================
					

				if twoColored:					
					frameCopy = bufferFrames[streamIndex].copy()    
	
					#bufferFrames[streamIndex] = morphEdgeDetection(bufferFrames[streamIndex])
					
					kernel = np.ones((settings.lineThicknessValue, settings.lineThicknessValue),np.uint8)
					#bufferFrames[streamIndex] = cv2.dilate(bufferFrames[streamIndex],kernel,iterations = 1)
					#frameCopy[np.where((bufferFrames[streamIndex] > [0, 0, 0]).all(axis=2))] = [0,0,0]
					frameCopy = limitColorsKmeans(frameCopy, 2)
					#frameCopy = cv2.GaussianBlur(frameCopy, (3, 3), 2)
					bufferFrames[streamIndex] = frameCopy
					bufferFrames[streamIndex] = sharpening(bufferFrames[streamIndex], settings.sharpeningValue, settings.sharpeningValue2)                    
					bufferFrames[streamIndex] = denoise(bufferFrames[streamIndex], settings.denoiseValue, settings.denoiseValue2)  

				if frameUpscale:	
					bufferFrames[streamIndex] = upscaleImage(netUpscaler, bufferFrames[streamIndex])   
					bufferFrames[streamIndex] = sharpening(bufferFrames[streamIndex], settings.sharpeningValue, settings.sharpeningValue2) 											       

				if asciiPainter:					
					bufferFrames[streamIndex]  = asciiPaint(bufferFrames[streamIndex], settings.asciiSizeValue, settings.asciiIntervalValue, settings.asciiThicknessValue, settings.rcnnBlurValue)

				if denoiseAndSharpen:
					bufferFrames[streamIndex] = sharpening(bufferFrames[streamIndex], settings.sharpeningValue, settings.sharpeningValue2)                    
					bufferFrames[streamIndex] = denoise(bufferFrames[streamIndex], settings.denoiseValue, settings.denoiseValue2)  				

				if sobel:					
					bufferFrames[streamIndex] = denoise(bufferFrames[streamIndex], settings.denoiseValue, settings.denoiseValue2)
					bufferFrames[streamIndex] = sharpening(bufferFrames[streamIndex], settings.sharpeningValue, settings.sharpeningValue2) 
					bufferFrames[streamIndex] = cv2.Sobel(bufferFrames[streamIndex],cv2.CV_64F,1,0,ksize=settings.sobelValue)   
							              
				bufferFrames[streamIndex] = adjustBrContrast(bufferFrames[streamIndex], settings.contrastValue, settings.brightnessValue)
				bufferFrames[streamIndex] = adjustSaturation(bufferFrames[streamIndex], settings.saturationValue)
						
				with lock:
					personDetected = False
					checkIfUserIsConnected(timerStart)

					serverStates.frameProcessed = serverStates.frameProcessed + 1
					elapsedTime = time.time()
					fps = 1 / (elapsedTime - startMoment)
					# print(fps)
					xCoeff = 512 / bufferFrames[streamIndex].shape[0] 
					xSize = round(xCoeff * bufferFrames[streamIndex].shape[1])
					print (xSize)
					resized =  cv2.resize(bufferFrames[streamIndex], (xSize, 512))

					for streamIndex in range(len(streamList)):
						if (showAllObjects):

							classIndexCount = [
								[0 for x in range(80)] for x in range(len(streamList))]

							rowIndex = 1
							for m in range(80):
								for k in range(len(classesIndex[streamIndex])):
									if (m == classesIndex[streamIndex][k]):
										classIndexCount[streamIndex][m] += 1

								if (classIndexCount[streamIndex][m] != 0):                                    
									rowIndex += 1

									if (classes[m] == "person"):
										cv2.rectangle(
											resized, (20, rowIndex * 40 - 25),
											(270, rowIndex * 40 + 11), (0, 0, 0), -1)
										cv2.putText(resized, classes[m] + ": " + str(
											classIndexCount[streamIndex][m]), (40, rowIndex * 40), font, 1,
													(0, 255, 0), 2, lineType=cv2.LINE_AA)
										personDetected = True

									if (classes[m] == "car"):
										cv2.rectangle(
											resized, (20, rowIndex * 40 - 25),
											(270, rowIndex * 40 + 11), (0, 0, 0), -1)
										cv2.putText(resized, classes[m] + ": " + str(
											classIndexCount[streamIndex][m]), (40, rowIndex * 40), font, 1,
													(255, 0, 255), 2, lineType=cv2.LINE_AA)

									if ((classes[m] != "car") & (classes[m] != "person")):
										cv2.rectangle(
											resized, (20, rowIndex * 40 - 25),
											(270, rowIndex * 40 + 11), (0, 0, 0), -1)
										cv2.putText(resized, classes[m] + ": " + str(
											classIndexCount[streamIndex][m]), (40, rowIndex * 40), font, 1,
													colorsYolo[m], 2, lineType=cv2.LINE_AA)

									if (classes[m] == "handbag") | (classes[m] == "backpack"):
										passFlag = True
										print("handbag detected! -> PASS")

						if (serverStates.sourceMode == "image"):							
							cv2.imwrite(f"static/output{args['port']}{serverStates.sourceImage}", bufferFrames[streamIndex])
														
						if ((serverStates.sourceMode == "image" and extractAndReplaceBackground == True and writer is not None)):
							writer.write(bufferFrames[streamIndex])
						
						# resized1 = cv2.resize(frameList[streamIndex], (640, 360))
						# resized2 = cv2.resize(bufferFrames[streamIndex], (640, 360))                            
						# concated = cv2.vconcat([resized2, resized1, ])                           
						# resized = cv2.resize(bufferFrames[streamIndex], (1600, 900))

						if (serverStates.sourceMode == "video" and writer is not None and startedRenderingVideo):
							writer.write(bufferFrames[streamIndex])	
							
						cv2.imshow("video",  bufferFrames[streamIndex])                           
						key = cv2.waitKey(1) & 0xFF

						if key == ord("q"):
							break												

						if (serverStates.sourceMode == "video"):
							if (serverStates.totalFrames != 0):
								progress = serverStates.frameProcessed / serverStates.totalFrames * 100
													
						cv2.putText(resized, f"FPS: {str(round(fps, 2))} ({str(bufferFrames[streamIndex].shape[1])}x{str(bufferFrames[streamIndex].shape[0])})", (40, 35),
									font, 0.8, (0, 255, 255), 2, lineType=cv2.LINE_AA)		
						outputFrame = resized

						if (serverStates.frameProcessed == 1):
							print("started")	
						
						if (serverStates.needToCreateScreenshot == True):
							print("screenshot")
							cv2.imwrite(f"static/output{args['port']}Screenshot.png", bufferFrames[streamIndex])
							time.sleep(1)		
							serverStates.screenshotPath = f"static/output{args['port']}Screenshot.png"
							serverStates.screenshotReady = True	
																	
			else:
				xCoeff = bufferFrames[streamIndex].shape[0] / 512
				xSize = round(xCoeff * bufferFrames[streamIndex].shape[1])
				print (xSize)
				resized =  cv2.resize(bufferFrames[streamIndex], (xSize, 640))
				outputFrame = bufferFrames[streamIndex]				
				zipObj.close()
				checkIfUserIsConnected(timerStart)
				startedRenderingVideo = False
				positionValue = 1
				print("finished")										

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(
	['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'm4v', 'webm', 'mkv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def generate():
	global outputFrame, lock, serverStates

	while serverStates.workingOn:		
		with lock:
			if outputFrame is None:
				continue

			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)            

			if not flag:
				continue

		yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			   bytearray(encodedImage) + b'\r\n')
		
	# else:
	# return redirect(url_for("results"))
	# return redirect('/results')
	print("yield finished")

@app.route('/', methods=['GET', 'POST'])
def index(device=None, action=None):
	global cap, cap2, fileToRender, fileChanged, serverStates
	
	if request.method == 'POST':        
		file = request.files['file']

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

			fileExtension = filename.rsplit('.', 1)[1]

			if (fileExtension == "png" or fileExtension == "jpg" or fileExtension == "jpeg" or fileExtension == "gif"):				
				serverStates.sourceMode = "image"
				serverStates.sourceImage = filename
				cap2 = cv2.VideoCapture("inputVideos/snow.webm")
			else:				
				serverStates.sourceMode = "video"
				cap = cv2.VideoCapture(filename)
				serverStates.totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
				cap2 = cv2.VideoCapture("inputVideos/snow.webm")
				
			CRED = '\033[91m'
			CEND = '\033[0m'
			serverStates.options = request.form.getlist('check')
			mode = request.form.getlist('checkMode')

			print(CRED + f"==============  file {filename} uploaded ============== " + CEND)
			
			#startedRenderingVideo = True
			fileToRender = filename
			fileChanged = True
	
	fileOutput = fileToRender 

	if (serverStates.sourceMode == "video"):
		fileOutput = fileToRender + ".avi"	
		
	return render_template("index.html", frameProcessed=serverStates.frameProcessed,
						   pathToRenderedFile=f"static/output{args['port']}{fileOutput}", pathToZipFile=f"static/objects{args['port']}.zip")

@app.route("/video")
def video_feed():
	# redirect(f"http://192.168.0.12:8000/results")
	return Response(generate(),
					mimetype="multipart/x-mixed-replace; boundary=frame")

# return Response(stream_with_context(generate()))
@app.route('/update', methods=['POST'])
def update():
	global serverStates, userTime

	timerStart = time.perf_counter()
	frameWidthToPage = 0
	frameHeightToPage = 0
	screenshotReadyLocal = False
	
	if (serverStates.screenshotReady  == True):
		screenshotReadyLocal = True		
		serverStates.screenshotReady = False
		serverStates.needToCreateScreenshot = False

	if (serverStates.sourceMode == "video"):
		frameWidthToPage = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		frameHeightToPage = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
	if (serverStates.sourceMode == "image"):
		frameWidthToPage = 0
		frameHeightToPage = 0  

	if (screenshotReadyLocal == True):
		print("sendingScreenshot ================================================")

	return jsonify({
		'value': serverStates.frameProcessed,
		'totalFrames': serverStates.totalFrames,
		'progress': round(progress, 2),
		'fps': round(fps, 2),
		'workingOn': serverStates.workingOn,
		'cpuUsage': psutil.cpu_percent(),
		'freeRam': round((psutil.virtual_memory()[1] / 2. ** 30), 2),
		'ramPercent': psutil.virtual_memory()[2],
		'frameWidth': frameWidthToPage,
		'frameHeight': frameHeightToPage,
		'maxFrames': serverStates.totalFrames,
		'currentMode': serverStates.options,
		'userTime': userTime,
		'screenshotReady': screenshotReadyLocal,
		'screenshotPath': serverStates.screenshotPath
		# 'time': datetime.datetime.now().strftime("%H:%M:%S"),
	})
	
@app.route('/update2', methods=['GET','POST'])
def sendCommand():
	global inputData, timerStart, timerEnd, writer, serverStates, commands
	
	if request.method == 'POST':
		timerStart = time.perf_counter()
		inputData = request.get_json()

		commands.modeResetCommand = str(inputData["modeResetCommand"])

		if (serverStates.videoStopLock == False):
			if (str(inputData["videoStopCommand"]) == 'True'):				
				serverStates.videoStopLock = True
					
		if (serverStates.videoResetLock == False):
			if (str(inputData["videoResetCommand"]) == 'True'):				
				serverStates.videoResetLock = True
			
		if (serverStates.screenshotLock == False):
			if (str(inputData["screenshotCommand"]) == 'True'):				
				serverStates.screenshotLock = True
			

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

@app.route('/download')
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = "static/cat53.jpg"
    return send_file(path, as_attachment=True)

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
