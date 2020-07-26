# example: python main.py -i 192.168.0.12 -o 8000

import os
import argparse
from flask import Flask, request, redirect
from werkzeug.utils import secure_filename
from flask import send_from_directory
from flask import render_template
import subprocess
import time

UPLOAD_FOLDER = ""
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif", "mp4", "avi", "m4v", "webm", "mkv"])

app = Flask(__name__, static_url_path="/static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
connection_port = 8000


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        youtube_url = request.form.get("textbox")

        if youtube_url != "":
            mode = "youtube"
            options = request.form.getlist("check")
            global connection_port
            connection_port = connection_port + 1
            return start_analysis(connection_port, youtube_url, options, mode)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            file_extension = filename.rsplit(".", 1)[1]

            if file_extension in ("png", "jpg", "jpeg"):
                mode = "image"
            else:
                mode = "video"

            options = request.form.getlist("check")
            # mode = request.form.getlist('checkMode')

            CRED = "\033[91m"
            CEND = "\033[0m"
            print(
                CRED
                + f"==============  file {filename} uploaded ============== "
                + CEND
            )

            connection_port = connection_port + 1
            return start_analysis(connection_port, filename, options, mode)

    return render_template("main.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/")
def start_analysis(port_to_render, file_to_render, options, mode):
    process_started = False
    str_from_list = ""

    for item in options:
        str_from_list += item

    # process = subprocess.Popen(
    #     [
    #         f"python",
    #         "-u",
    #         "processing.py",
    #         "-i",
    #         "192.168.0.12",
    #         "-o",
    #         str(port_to_render),
    #         "-s",
    #         str(file_to_render),
    #         "-c",
    #         str_from_list,
    #         "-m",
    #         mode,
    #     ],
    #     bufsize=0,
    #     stdout=subprocess.PIPE,
    # )

    process = subprocess.Popen(
        [
            f"python",
            "-u",
            "processing.py",
            "-i",
            "192.168.0.12",
            "-o",
            str(port_to_render),
            "-s",
            str(file_to_render),
            "-c",
            str_from_list,
            "-m",
            mode,
        ],
        bufsize=0
    )

    # while process.poll() is None and not process_started:
    #     output = process.stdout.readline()
    #     # output = process.communicate()[0]
    #     out = str(output.decode("utf-8"))
    #     print(out)
    #
    #     if out == "started\n":
    #         process_started = True
    #         # process.stdout.close()
    #         time.sleep(1)
    #         return redirect(f"http://192.168.0.12:{port_to_render}")

    time.sleep(6)
    return redirect(f"http://192.168.0.12:{port_to_render}")


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
