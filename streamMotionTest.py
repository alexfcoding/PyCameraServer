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

outputFrame = None
lock = threading.Lock()
A = 0
app = Flask(__name__)

streamList= [
	#"http://192.82.150.11:8083/mjpg/video.mjpg", 
	#"http://cam.butovonet.ru/axis-cgi/mjpg/video.cgi?resolution=480x576&dummy=1460609511992",
	#"http://cam.butovonet.ru/axis-cgi/mjpg/video.cgi?resolution=480x576&dummy=1460609511992",
	"http://cam.butovonet.ru/axis-cgi/mjpg/video.cgi?resolution=480x576&dummy=1460609511992"
	]

frameList = []
vsList = []
motionDetectors = []
grayFrames = []
total = []

for i in range(len(streamList)):
	vsList.append(VideoStream(streamList[i]))
	frameList.append(None)
	motionDetectors.append(None)
	grayFrames.append(None)
	vsList[i].start()

time.sleep(2.0)

@app.route("/")

def index():
	return render_template("index.html")

def detect_motion(frameCount):
	global vsList, outputFrame, lock
	
	for i in range(len(streamList)):
		motionDetectors[i] = Detector(accumWeight=0.1)
	
	for i in range(len(streamList)):
		total.append(None)
		total[i] = 0
		
	while True:
		for streamIndex in range(len(streamList)):
			frameList[streamIndex] = vsList[streamIndex].read()
			frameList[streamIndex] = cv2.resize(frameList[streamIndex], (640,480))

			grayFrames[streamIndex] = cv2.cvtColor(frameList[streamIndex], cv2.COLOR_BGR2GRAY)
			grayFrames[streamIndex] = cv2.GaussianBlur(grayFrames[streamIndex], (7, 7), 0)

			timestamp = datetime.datetime.now()
			cv2.putText(frameList[streamIndex], timestamp.strftime(
				"%A %d %B %Y %I:%M:%S%p"), (10, frameList[streamIndex].shape[0] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

			if total[streamIndex] > frameCount:
				motion = motionDetectors[streamIndex].detect(grayFrames[streamIndex])
				
				if motion is not None:
					(thresh, (minX, minY, maxX, maxY)) = motion
					cv2.rectangle(frameList[streamIndex], (minX, minY), (maxX, maxY),
						(0, 0, 255), 2)			

			motionDetectors[streamIndex].update(grayFrames[streamIndex])
			total[streamIndex] += 1
		
		with lock:
			#im_v = cv2.vconcat([frameList[0], frameList[1]])
			#im_v2 = cv2.vconcat([frameList[2], frameList[3]])
			#im_v3 = cv2.hconcat([im_v, im_v2])
			#vis = np.concatenate((frameList[0], frameList[1], frameList[2]), axis=1)
			outputFrame = frameList[0].copy()	
		
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
