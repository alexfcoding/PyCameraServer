# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
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
from cv2 import cv2

outputFrame = None
lock = threading.Lock()
A = 0
app = Flask(__name__)

streamList= [
	#"http://192.82.150.11:8083/mjpg/video.mjpg"
	"http://95.14.172.123:8081/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000",
	#"blob:https://ipeye.ru/c99b4254-224f-436a-810b-f0d1e8b42429",
	"http://212.46.249.62:8008/",
	"http://cam.butovonet.ru/axis-cgi/mjpg/video.cgi?resolution=480x576&dummy=1460609511992"
	]

frameList = []
vsList = []
motionDetectors = []
grayFrames = []
total = []
classes = []

for i in range(len(streamList)):
	vsList.append(VideoStream(streamList[i]))
	frameList.append(None)
	motionDetectors.append(None)
	grayFrames.append(None)
	vsList[i].start()

net = cv2.dnn.readNet("c:/Users/User/source/PyOpenCV/LocalExperiments/YOLO/yolov3.weights", "c:/Users/User/source/PyOpenCV/LocalExperiments/YOLO/yolov3.cfg")

with open("c:/Users/User/source/PyOpenCV/LocalExperiments/YOLO/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layers_names = net.getLayerNames()
outputLayers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes), 3))
img = None

time.sleep(2.0)

@app.route("/")

def index():
	return render_template("index.html")

def detect_motion(frameCount):
	global vsList, net, outputFrame, lock
	
	for i in range(len(streamList)):
		total.append(None)
		total[i] = 0
		
	while True:
		for streamIndex in range(len(streamList)):
			frameList[streamIndex] = vsList[streamIndex].read()
			frameList[streamIndex] = cv2.resize(frameList[streamIndex], (640,480))
			img = frameList[streamIndex]
			#img = cv2.resize(img, None, fx=1, fy=1)
			height, width, channels = img.shape

			blob = cv2.dnn.blobFromImage(img, 0.0039, (416,416), (0, 0, 0), True, crop=False)

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
					if confidence > 0.4: 
						center_x = int(detection[0] * width)
						center_y = int(detection[1] * height)
						w = int(detection[2] * width)
						h = int(detection[3] * height)

						#cv2.circle(img, (center_x, center_y), 2, (0,0,255), 2)
						x = int(center_x - w/2)
						y = int(center_y - h/2)
						boxes.append([x, y, w, h])
						confidences.append(float(confidence))
						class_ids.append(class_id)
						#cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)

			indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
			
			print(indexes)
			font = cv2.FONT_HERSHEY_PLAIN

			for i in range(len(boxes)):
				if i in indexes:
					x, y, w, h = boxes[i]
					label = classes[class_ids[i]]
					color = colors[class_ids[i]]
					cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 1)
					cv2.putText(img, label + ": " + str(np.round(confidences[i], 2)), (x, y - 5), font, 1, (0,0,255), 2)

		with lock:
			im_v = cv2.vconcat([frameList[0], frameList[1]])
			im_v2 = cv2.vconcat([frameList[2], frameList[2]])
			im_v3 = cv2.hconcat([im_v, im_v2])
			#vis = np.concatenate((im_v, frameList[0]), axis=1)
			outputFrame = im_v3
		
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
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
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
