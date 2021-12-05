# Examples:
# python processing.py -i 0.0.0.0 -o 8002 -s http://192.82.150.11:8083/mjpg/video.mjpg -c a -m ipcam
# python processing.py -i 0.0.0.0 -o 8002 -s https://youtu.be/5JJu-CTDLoc -c a -m video
# python processing.py -i 0.0.0.0 -o 8002 -s my_video.avi -c a -m video
# python processing.py -i 0.0.0.0 -o 8002 -s my_image.jpg -c t -m image

from flask import jsonify
from flask import Flask
from flask import render_template
import argparse
from flask import request, Response
from werkzeug.utils import secure_filename
from cv2 import cv2
import pafy
import os
import psutil
import threading
import time
import processing
from processing import process_frame

app = Flask(__name__, static_url_path="/static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
UPLOAD_FOLDER = "static/user_uploads/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "mp4", "avi", "m4v", "webm", "mkv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


def generate():
    while processing.states_dict['working_on']:
        with processing.lock:
            if processing.output_frame is None:
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", processing.output_frame)
            if not flag:
                continue
        yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encoded_image) + b"\r\n"
        )
    print("yield finished")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = None
        textbox_string = ""

        if 'file' in request.files:
            file = request.files["file"]
            print("in file")

        if 'textbox' in request.form:
            textbox_string = request.form.get("textbox")

        if textbox_string.find("youtu") != -1:
            processing.states_dict['source_mode'] = "youtube"
            processing.states_dict['source_url'] = textbox_string
            v_pafy = pafy.new(textbox_string)
            play = v_pafy.streams[0]
            processing.cap = cv2.VideoCapture(play.url)
            processing.states_dict['total_frames'] = processing.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            processing.file_changed = True

        if textbox_string.find("mjpg") != -1:
            processing.states_dict['source_mode'] = "ipcam"
            processing.states_dict['source_url'] = textbox_string
            processing.cap = cv2.VideoCapture()
            processing.cap.open(textbox_string)
            processing.states_dict['total_frames'] = 1
            processing.file_changed = True

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            file_extension = filename.rsplit(".", 1)[1]

            if file_extension in ("png", "jpg", "jpeg"):
                processing.states_dict['source_mode'] = "image"
                processing.states_dict['source_image'] = filename
                processing.cap2 = cv2.VideoCapture("input_videos/space.webm")

            if file_extension in ("gif", "mp4", "avi", "m4v", "webm", "mkv"):
                processing.states_dict['source_mode'] = "video"
                processing.cap = cv2.VideoCapture(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                processing.states_dict['total_frames'] = processing.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                processing.cap2 = cv2.VideoCapture("input_videos/space.webm")

            CRED = "\033[91m"
            CEND = "\033[0m"
            print(
                CRED
                + f"==============  file {filename} uploaded ============== "
                + CEND
            )
            processing.file_to_render = filename
            processing.file_changed = True

    if processing.states_dict['source_mode'] == "video":
        processing.states_dict['output_file_page'] = processing.file_to_render + ".avi"
    if processing.states_dict['source_mode'] == "youtube":
        processing.states_dict['output_file_page'] = "youtube.avi"
    if processing.states_dict['source_mode'] == "ipcam":
        processing.states_dict['output_file_page'] = "ipcam.avi"

    return render_template(
        "index.html",
        frame_processed=processing.states_dict['frame_processed'],
        pathToRenderedFile=f"static/user_renders/output{args['port']}{processing.states_dict['output_file_page']}",
        pathToZipFile=f"static/user_renders/output{args['port']}.zip",
    )


@app.route("/video")
def video_feed():
    # redirect(f"http://192.168.0.12:8000/results")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats", methods=["POST"])
def send_stats():
    frame_width_to_page = 0
    frame_height_to_page = 0
    screenshot_ready_local = False

    if processing.states_dict['screenshot_ready']:
        screenshot_ready_local = True
        processing.states_dict['screenshot_ready'] = False

    if processing.states_dict['source_mode'] in ("video", "youtube", "ipcam"):
        if processing.cap is not None:
            frame_width_to_page = processing.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height_to_page = processing.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if processing.states_dict['source_mode'] == "image":
        frame_width_to_page = 0
        frame_height_to_page = 0

    return jsonify(
        {
            "value": processing.states_dict['frame_processed'],
            "totalFrames": processing.states_dict['total_frames'],
            "progress": round(processing.progress, 2),
            "fps": round(processing.fps, 2),
            "workingOn": processing.states_dict['working_on'],
            "cpuUsage": psutil.cpu_percent(),
            "freeRam": round((psutil.virtual_memory()[1] / 2.0 ** 30), 2),
            "ramPercent": psutil.virtual_memory()[2],
            "frameWidth": frame_width_to_page,
            "frameHeight": frame_height_to_page,
            "currentMode": processing.states_dict['render_mode'],
            "userTime": processing.user_time,
            "screenshotReady": screenshot_ready_local,
            "screenshotPath": processing.states_dict['screenshot_path']
            # 'time': datetime.datetime.now().strftime("%H:%M:%S"),
        }
    )


@app.route("/settings", methods=["GET", "POST"])
def receive_settings():
    if request.method == "POST":
        processing.timer_start = time.perf_counter()
        processing.settings_ajax = request.get_json()

        if not processing.states_dict['mode_reset_lock']:
            if bool(processing.settings_ajax["modeResetCommand"]):
                processing.states_dict['mode_reset_lock'] = True

        if not processing.states_dict['video_stop_lock']:
            if bool(processing.settings_ajax["videoStopCommand"]):
                processing.states_dict['video_stop_lock'] = True

        if not processing.states_dict['video_reset_lock']:
            if bool(processing.settings_ajax["videoResetCommand"]):
                processing.states_dict['video_reset_lock'] = True

        if not processing.states_dict['screenshot_lock']:
            if bool(processing.settings_ajax["screenshotCommand"]):
                processing.states_dict['screenshot_lock'] = True
                print("screenshot_lock")

    return "", 200


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--ip", type=str, required=True, help="ip address of the device"
    )
    ap.add_argument(
        "-o",
        "--port",
        type=int,
        required=True,
        help="port number of the server",
    )
    ap.add_argument("-s", "--source", type=str, default=32, help="file to render")
    ap.add_argument(
        "-c", "--optionsList", type=str, required=True, help="rendering options"
    )
    ap.add_argument(
        "-m",
        "--mode",
        type=str,
        required=True,
        help="rendering mode: 'video' or 'image'",
    )

    args = vars(ap.parse_args())
    t = threading.Thread(target=process_frame, args=(args, app))
    t.daemon = True
    t.start()

    app.run(
        host=args["ip"],
        port=args["port"],
        debug=False,
        threaded=True,
        use_reloader=False,
    )
