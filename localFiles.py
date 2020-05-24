# python localFiles.py -i 192.168.0.12 -o 8000

from processing.motion_detection import Detector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import numpy as np
import cv2
import time
import os

outputFrame = None
lock = threading.Lock()
A = 0

app = Flask(__name__)

streamList= [
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

frameProcessed = 0
fileIterator = 0

for i in range(len(streamList)):
	vsList.append(VideoStream(streamList[i]))
	frameList.append(None)
	bufferFrames.append(None)
	frameOutList.append(None)
	motionDetectors.append(None)
	grayFrames.append(None)
	vsList[i].start()

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layers_names = net.getLayerNames()
outputLayers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes), 3))
img = None

time.sleep(2.0)


fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None

@app.route("/")

def index():
	return render_template("index.html")

def detect_motion(frameCount):
	global vsList, writer, net, fileIterator, frameProcessed, outputFrame, lock

	fileToRender=args["source"]
	cap = cv2.VideoCapture(fileToRender)

	for i in range(len(streamList)):
		total.append(None)
		total[i] = 0

	while True:
		classesIndex = []
		startMoment = time.time()
		for streamIndex in range(len(streamList)):
			ret, frameList[streamIndex] = cap.read()
			bufferFrames[streamIndex] = frameList[streamIndex].copy()
			#frameList[streamIndex] = cv2.resize(frameList[streamIndex], (800,600))
			#bufferFrames[streamIndex] = cv2.resize(bufferFrames[streamIndex], (800,600))
			height, width, channels = frameList[streamIndex].shape

			blob = cv2.dnn.blobFromImage(frameList[streamIndex], 0.003, (640,640), (0, 0, 0), True, crop=False)
			net.setInput(blob)
			outs = net.forward(outputLayers)

			class_ids = []
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
						x = int(center_x - w/2)
						y = int(center_y - h/2)
						boxes.append([x, y, w, h])
						confidences.append(float(confidence))
						class_ids.append(class_id)
						#cv2.rectangle(bufferFrames[streamIndex], (x, y), (x + w, y + h), (0,255,0), 2)

			indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.2)

			#print(indexes)
			print("=========================")
			font = cv2.FONT_HERSHEY_SIMPLEX
			lineType = cv2.LINE_AA

			classesOut = []
			objectIndex = 0
			
			for i in range(len(boxes)):
				if i in indexes:
					x, y, w, h = boxes[i]
					label = classes[class_ids[i]]
					color = colors[class_ids[i]]

					classesOut.append(class_ids[i])

					# if (x<0):
					# 	x = 0
					# if (y<0):
					# 	y=0
					# fileIterator += 1
					# #crop_img = frameList[streamIndex][y:y+h, x:x+w]
					# cv2.imwrite(label + str(fileIterator)+".jpg", crop_img)
					# #cv2.imshow("sf", crop_img)

					blk = np.zeros(bufferFrames[streamIndex].shape, np.uint8)

					if label == "person":
						cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
						cv2.rectangle(blk, (x, y), (x + w, y + h), (0,255,0), cv2.FILLED)
						bufferFrames[streamIndex] = cv2.addWeighted(bufferFrames[streamIndex], 1, blk, 0.2, 0)
					if label == "car":
						cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (213,160,47), 2, lineType = cv2.LINE_AA)
						cv2.rectangle(blk, (x, y), (x + w, y + h), (213,160,47), cv2.FILLED)
						bufferFrames[streamIndex] = cv2.addWeighted(bufferFrames[streamIndex], 1, blk, 0.2, 0)
					if ((label != "car") & (label != "person")):
						cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, color, 2, lineType = cv2.LINE_AA)
						cv2.rectangle(blk, (x, y), (x + w, y + h), color, cv2.FILLED)
						bufferFrames[streamIndex] = cv2.addWeighted(bufferFrames[streamIndex], 1, blk, 0.5, 0)
					
					# 	cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
					# 	cv2.rectangle(blk, (x, y), (x + w, y + h), (0,255,0), cv2.FILLED)
					# 	bufferFrames[streamIndex] = cv2.addWeighted(bufferFrames[streamIndex], 1, blk, 0.2, 0)
					# if (label == "handbag")|(label == "backpack"):
					# 	cv2.circle(bufferFrames[streamIndex], (x+int(round(w/2)), y+int(round(h/2))), 3, (0, 0, 255), 3)
					# 	bufferFrames[streamIndex] = cv2.addWeighted(bufferFrames[streamIndex], 1, blk, 0.2, 0)

					cv2.rectangle(bufferFrames[streamIndex], (x, y), (x + w, y + h), (255,255,255), 2)
					#cv2.imshow('123', bufferFrames[streamIndex])
					#cv2.waitKey()
					objectIndex+=1

			classesIndex.append(classesOut)

		with lock:
			frameProcessed = frameProcessed + 1
			elapsedTime = time.time()
			fps = 1 / (elapsedTime - startMoment)
			print (fps)
			for streamIndex in range(len(streamList)):
				classIndexCount = [[0 for x in range(80)] for x in range(len(streamList))]
				countLocal = [0 for x in range(80)]
				skipFlag = False
				passFlag = False
				#cv2.rectangle(bufferFrames[streamIndex], (20, 30), (400, 86), (0, 0, 0), -1)
				#cv2.putText(bufferFrames[streamIndex], "BLOB: 320x320", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, lineType=cv2.LINE_AA)
				#cv2.rectangle(bufferFrames[streamIndex], (20, 100), (400, 156), (0, 0, 0), -1)
				#cv2.putText(bufferFrames[streamIndex], "FPS: " + str(round(fps, 2)), (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, lineType=cv2.LINE_AA)
				
				rowIndex = 0
				for m in range(80):
					for k in range(len(classesIndex[streamIndex])):
						if (m == classesIndex[streamIndex][k]):
							classIndexCount[streamIndex][m]+=1

					if (classIndexCount[streamIndex][m]!=0):
						rowIndex += 1
						
						# cv2.rectangle(bufferFrames[streamIndex], (0, rowIndex*40 - 20), (200,rowIndex*40 + 8), (0,0,0), -1)
						# cv2.putText(bufferFrames[streamIndex], classes[m] + ": " + str(classIndexCount[streamIndex][m]), (20,rowIndex*40), font, 0.7, colors[m], 2, cv2.LINE_AA)
						
						if (classes[m]=="person"):
							cv2.rectangle(bufferFrames[streamIndex], (20, rowIndex* 70 - 40), (400,rowIndex * 70 + 16), (0,0,0), -1)
							cv2.putText(bufferFrames[streamIndex], classes[m] + ": " + str(classIndexCount[streamIndex][m]), (40, rowIndex * 70), font, 1.4, (0,255,0), 2, lineType=cv2.LINE_AA)
						if (classes[m]=="car"):
							cv2.rectangle(bufferFrames[streamIndex], (20, rowIndex * 70 - 40), (400, rowIndex * 70 + 16), (0, 0, 0), -1)
							cv2.putText(bufferFrames[streamIndex], classes[m] + ": " + str(classIndexCount[streamIndex][m]), (40, rowIndex * 70), font, 1.4, (213,160,47), 2, lineType=cv2.LINE_AA)
						if ((classes[m] != "car") & (classes[m] != "person")):
							cv2.rectangle(bufferFrames[streamIndex], (20, rowIndex * 70 - 40), (400, rowIndex * 70 + 16), (0, 0, 0), -1)
							cv2.putText(bufferFrames[streamIndex], classes[m] + ": " + str(classIndexCount[streamIndex][m]), (40, rowIndex * 70), font, 1.4, colors[m], 2, lineType=cv2.LINE_AA)
						
						if (classes[m]=="handbag")|(classes[m]=="backpack"):
							passFlag = True
							print("handbag detected! -> PASS")

				if writer is None:
					writer = cv2.VideoWriter("testOutput2.avi", fourcc, 30,(bufferFrames[streamIndex].shape[1], bufferFrames[streamIndex].shape[0]), True)			
				else:
					writer.write(bufferFrames[streamIndex])
					#resized = cv2.resize(bufferFrames[streamIndex], (1280, 720))
					cv2.imshow("video", bufferFrames[streamIndex])
					key = cv2.waitKey(1) & 0xFF
				
			# if (skipFlag == False)&(passFlag==True):			
			# 	writer.write(bufferFrames[streamIndex])
			# 	resized = cv2.resize(bufferFrames[streamIndex], (1280, 720))
			# 	cv2.imshow("video", resized)
			# 	key = cv2.waitKey(1) & 0xFF
			#im_v = cv2.vconcat([bufferFrames[0], bufferFrames[1]])
			#im_v2 = cv2.vconcat([bufferFrames[2], bufferFrames[3]])
			#im_v3 = cv2.hconcat([im_v, im_v2])
			#vis = np.concatenate((im_v, frameList[0]), axis=1)
			#outputFrame = im_v3.copy()
			outputFrame = bufferFrames[streamIndex]

def generate():
	global outputFrame, lock

	while True:
		with lock:
			if outputFrame is None:
				continue

			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			if not flag:
				continue

		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			bytearray(encodedImage)  + b'\r\n')

@app.route("/video")

def video_feed():
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-s", "--source", type=str, default=32,
		help="# file to render")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	
	args = vars(ap.parse_args())

	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

for j in range(len(streamList)):
	vsList[j].stop()
