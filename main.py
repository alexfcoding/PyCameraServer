# example: python main.py -i 192.168.0.12 -o 8000

import os
import argparse
from flask import Flask, request, redirect
from werkzeug.utils import secure_filename
from flask import send_from_directory
from flask import render_template
import subprocess
import time

UPLOAD_FOLDER = "static/user_uploads/"
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif", "mp4", "avi", "m4v", "webm", "mkv"])

app = Flask(__name__, static_url_path="/static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
connection_port = 8000


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


def start_process(auto_start, port, source_type, source, mode, delay=5):
    """
    Starts processing.py with user port, source and mode
    auto_start=False for debug mode (manual launch with delay)
    """
    if (auto_start):
        process_started = False
        process = subprocess.Popen(
            [
                f"python",
                "-u",
                "processing.py",
                "-i",
                "192.168.0.12",
                "-o",
                str(port),
                "-s",
                str(source),
                "-c",
                mode,
                "-m",
                source_type,
            ],
            bufsize=0,
            stdout=subprocess.PIPE,
        )

        while process.poll() is None and not process_started:
            output = process.stdout.readline()
            # output = process.communicate()[0]
            out = str(output.decode("utf-8"))
            print(out)

            if out == "started\n":
                process_started = True
                # process.stdout.close()
                time.sleep(2)
    else:
        process = subprocess.Popen(
            [
                f"python",
                "-u",
                "processing.py",
                "-i",
                "192.168.0.12",
                "-o",
                str(port),
                "-s",
                source,
                "-c",
                mode,
                "-m",
                source_type,
            ],
            bufsize=0
        )
        time.sleep(delay)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    global connection_port

    if request.method == "POST":
        file = request.files["file"]
        url = request.form.get("urlInput")

        if url.find("youtu") != -1:
            source_type = "youtube"
            mode = request.form.getlist("check")
            connection_port += 1
            return start_analysis(connection_port, url, mode, source_type)

        if url.find("mjpg") != -1:
            source_type = "ipcam"
            mode = request.form.getlist("check")
            connection_port += 1
            return start_analysis(connection_port, url, mode, source_type)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            file_extension = filename.rsplit(".", 1)[1]
            source_type = ""

            if file_extension in ("png", "jpg", "jpeg"):
                source_type = "image"
            if file_extension in ("gif", "mp4", "avi", "m4v", "webm", "mkv"):
                source_type = "video"

            mode = request.form.getlist("check")
            # source_type = request.form.getlist('checksource_type')

            CRED = "\033[91m"
            CEND = "\033[0m"
            print(
                CRED
                + f"==============  file {filename} uploaded ============== "
                + CEND
            )

            connection_port = connection_port + 1
            return start_analysis(connection_port, filename, mode, source_type)

    return render_template("main.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/")
def start_analysis(port_to_render, file_to_render, mode, source_type):
    mode_str = ""

    for item in mode:
        mode_str += item

    source = ""

    if source_type in ("video", "image"):
        source = f"{UPLOAD_FOLDER}{file_to_render}"
    if source_type in ("youtube", "ipcam"):
        source = f"{file_to_render}"

    start_process(True, connection_port, source_type, source, mode_str)

    return redirect(f"http://192.168.0.12:{connection_port}")

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
        help="port number of the server (1024 to 65535)",
    )

    args = vars(ap.parse_args())

    app.run(
        host=args["ip"],
        port=args["port"],
        debug=False,
        threaded=True,
        use_reloader=False,
    )
