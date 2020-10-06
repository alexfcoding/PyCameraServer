# Examples:
# python processing.py -i 192.168.0.12 -o 8002 -s http://192.82.150.11:8083/mjpg/video.mjpg -c a -m ipcam
# python processing.py -i 192.168.0.12 -o 8002 -s https://youtu.be/5JJu-CTDLoc -c a -m video
# python processing.py -i 192.168.0.12 -o 8002 -s my_video.avi -c a -m video
# python processing.py -i 192.168.0.12 -o 8002 -s my_image.jpg -c t -m image

from flask import jsonify
from flask import Flask
from flask import render_template
import threading
import argparse
from flask import request, Response
import psutil
from mode_selector import *
from werkzeug.utils import secure_filename
from zipfile import ZipFile
import pafy

app = Flask(__name__, static_url_path="/static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
UPLOAD_FOLDER = "static/user_uploads/"
ALLOWED_EXTENSIONS = set(
    ["png", "jpg", "jpeg", "gif", "mp4", "avi", "m4v", "webm", "mkv"]
)

class State:
    # Working states and lock commands
    view_source = False
    source_image = ""
    source_url = ""
    source_mode = ""
    output_file_page = ""
    screenshot_path = ""
    need_to_create_screenshot = False
    screenshot_ready = False
    working_on = True
    frame_processed = 0
    total_frames = 0
    options = ""
    screenshot_lock = False
    video_reset_lock = False
    video_stop_lock = False
    mode_reset_lock = False
    source_lock = False
    render_mode = ""
    superres_model = "LAPSRN"
    esrgan_model = "FALCOON"


# Rendering modes dictionary
render_modes_dict = {
    'using_yolo_network': False,
    'using_caffe_network': False,
    'using_mask_rcnn_network': False,
    'canny_people_on_background': False,
    'canny_people_on_black': False,
    'extract_and_replace_background': False,
    'extract_and_cut_background': False,
    'color_canny': False,
    'color_canny_on_background': False,
    'color_objects_on_gray_blur': False,
    'color_objects_blur': False,
    'color_objects_on_gray': False,
    'caffe_colorization': False,
    'cartoon_effect': False,
    'extract_objects_yolo_mode': False,
    'text_render_yolo': False,
    'denoise_and_sharpen': False,
    'sobel': False,
    'ascii_painter': False,
    'pencil_drawer': False,
    'two_colored': False,
    'upscale_opencv': False,
    'upscale_esrgan': False,
    'boost_fps_dain': False
}

# Default rendering settings
# Values will change with AJAX requests
settings_ajax = {
    "viewSource" : False,
    "cannyBlurSliderValue" : 5,
    "cannyThresSliderValue" : 50,
    "cannyThresSliderValue2" : 50,
    "cannyThres2" : 50,
    "saturationSliderValue" : 100,
    "contrastSliderValue" : 100,
    "brightnessSliderValue" : 0,
    "positionSliderValue" : 1,
    "confidenceSliderValue" : 20,
    "lineThicknessSliderValue" : 2,
    "denoiseSliderValue" : 10,
    "denoiseSliderValue2" : 10,
    "sharpenSliderValue" : 5,
    "sharpenSliderValue2" : 5,
    "rcnnSizeSliderValue" : 10,
    "rcnnBlurSliderValue" : 17,
    "sobelSliderValue" : 3,
    "asciiSizeSliderValue" : 4,
    "asciiIntervalSliderValue" : 10,
    "asciiThicknessSliderValue" : 1,
    "resizeSliderValue" : 2,
    "colorCountSliderValue" : 32,
    "mode" : "a",
    "superresModel" : "LapSRN",
    "esrganModel" : "FALCOON",
    "urlSource": "default"
}

server_states = State() # Global instance for accessing settings from requests and rendering loop

timer_start = 0 # Start timer for stopping rendering if user closed tab
timer_end = 0 # End timer for stopping rendering if user closed tab
user_time = 0 # For user timer debug
output_frame = None # Frame to preview on page
progress = 0 # Rendering progress 0-100%
cap = None # VideoCapture object for user frames
cap2 = None # VideoCapture object for secondary video (need for some effects)
lock = threading.Lock() # Lock for thread (multiple browser connections viewing)
main_frame = None # Processing frame from video, image or youtube URL
frame_background = None # Frame for secondary video

fourcc = cv2.VideoWriter_fourcc(*"MJPG") # Format for video saving
writer = None # Writer for video saving

url = ""

def check_if_user_is_connected(timer_start, seconds_to_disconnect):
    """
    Stops rendering process after a few seconds if user closed browser tab
    :param timer_start: moment of last AJAX user ping
    :param seconds_to_disconnect: number of seconds to shutdown
    :return:
    """
    global user_time
    timer_end = time.perf_counter()
    user_time = str(round(timer_end)) + ":" + str(round(timer_start))
    # print(timer_start)

    if not (timer_end - timer_start < seconds_to_disconnect and timer_start != 0):
        # print("User is connected")
        if timer_start != 0:
            print(
                "User disconnected, shutting down!"
            )
            current_pid = os.getpid()
            p = psutil.Process(current_pid)
            p.terminate()  # or p.kill()


def process_frame():
    """
    Main rendering function
    :return:
    """
    global cap, lock, writer, progress, fps, output_frame, file_to_render, zip_obj

    frame_boost_list = [] # List for Depth-Aware Video Frame Interpolation frames
    frame_boost_sequence = [] # Interpolated frame sequence for video writing

    server_states.frame_processed = 0 # Total frames processed
    server_states.total_frames = 0 # Total frames in video

    received_zip_command = False # Trigger for receiving YOLO objects downloading command by user
    file_changed = False # Trigger for file changing by user
    started_rendering_video = False # Trigger for start rendering video by user
    need_mode_reset = True # Trigger for rendering mode changing by user
    server_states.working_on = True # Rendering state is ON

    concated = None
    need_to_create_new_zip = True # Loop state to open new zip archive
    need_to_stop_new_zip = False # Loop state to close zip archive
    zip_is_opened = True # Loop state for saving new images to zip archive
    zipped_images = False # Loop state for closed zip archive
    font = cv2.FONT_HERSHEY_SIMPLEX # Font for rendering stats on frame by OpenCV
    resized = None # Resized frame to put on page
    fps = 0 # FPS rate
    frameEdge = None # Last frame of interpolation sequence

    path_to_file, file_to_render = os.path.split(args["source"]) # Get filename from full path
    print ("Processing file: " + file_to_render)
    server_states.source_url = args["source"] # Youtube URL

    server_states.render_mode = args["optionsList"] # Rendering mode from command line
    server_states.source_mode = args["mode"] # Source type from command line

    # Set source for youtube capturing
    if server_states.source_mode == "youtube":
        vPafy = pafy.new(server_states.source_url)
        play = vPafy.streams[0]
        cap = cv2.VideoCapture(play.url)
        server_states.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Set source for ip camera capturing
    if server_states.source_mode == "ipcam":
        cap = cv2.VideoCapture()
        cap.open(server_states.source_url)
        server_states.total_frames = 1

    # Set source for video file capturing
    if server_states.source_mode == "video":
        cap = cv2.VideoCapture(f"{app.config['UPLOAD_FOLDER']}{file_to_render}")
        server_states.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Set source for image file capturing
    if server_states.source_mode == "image":
        path_to_image, image_file = os.path.split(args["source"])
        server_states.source_image = image_file

    cap2 = cv2.VideoCapture("input_videos/space.webm") # Secondary video for background replacement
    zip_obj = ZipFile(f"static/user_renders/output{args['port']}.zip", "w") # Zip file with user port name

    # Initialize all models
    caffe_network = initialize_caffe_network()
    caffe_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    caffe_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    superres_network = initialize_superres_network("LAPSRN")
    esrgan_network, device = initialize_esrgan_network("FALCOON", True)
    rcnn_network = initialize_rcnn_network(True)
    dain_network = initialize_dain_network(True)
    yolo_network, layers_names, output_layers, colors_yolo = initialize_yolo_network(
        classes, True
    )

    frame_interp_num = 0 # Interpolated frame number
    main_frame = None
    f = f1 = None # Two source frames for interpolation

    # =============================== Main processing loop ===============================

    while server_states.working_on:
        # Receive all HTML slider values from JSON dictionary
        if settings_ajax is not None:
            mode_from_page = str(settings_ajax["mode"])
            superres_model_from_page = str(settings_ajax["superresModel"])
            esrgan_model_from_page = str(settings_ajax["esrganModel"])
            position_value_local = int(settings_ajax["positionSliderValue"])
            server_states.view_source = bool(settings_ajax["viewSource"])

            # Check if mode change command was received
            if server_states.mode_reset_lock:
                server_states.render_mode = mode_from_page
                server_states.superres_model = superres_model_from_page
                server_states.esrgan_model = esrgan_model_from_page
                server_states.mode_reset_lock = False
                need_mode_reset = True

            # Check if video rendering start command was received
            if server_states.video_reset_lock:
                position_value = 1 # Reset position
                need_to_create_writer = True # Create new writer
                started_rendering_video = True
                received_zip_command = True
                server_states.video_reset_lock = False
                # print("in loop reset")
            else:
                position_value = position_value_local # Read frame position from slider

            # Check if video rendering stop command was received
            if server_states.video_stop_lock:
                position_value = 1
                started_rendering_video = False
                server_states.video_stop_lock = False
                # print("in loop stop")

            # Check if taking screenshot command was received
            if server_states.screenshot_lock:
                # print("in loop screenshot")
                server_states.need_to_create_screenshot = True
                server_states.screenshot_lock = False
        else:
            position_value = 1

        # If user changed rendering mode
        if need_mode_reset:
            frame_interp_num = 0
            # Reset all modes
            for mode in render_modes_dict:
                render_modes_dict[mode] = False
            # print("need mode reset")

            # Reinitialize upscale networks with user models from page
            superres_network = initialize_superres_network(server_states.superres_model)
            esrgan_network, device = initialize_esrgan_network(server_states.esrgan_model, True)

            # Set processing algorithm from HTML page
            for mode in server_states.render_mode:
                if mode == "a":
                    render_modes_dict['extract_objects_yolo_mode'] = True
                    render_modes_dict['using_yolo_network'] = True
                    print("extract_objects_yolo")
                if mode == "b":
                    render_modes_dict['text_render_yolo'] = True
                    render_modes_dict['using_yolo_network'] = True
                    print("text_render_yolo")
                if mode == "c":
                    render_modes_dict['canny_people_on_black'] = True
                    render_modes_dict['using_yolo_network'] = True
                    print("canny_people_on_black")
                if mode == "d":
                    render_modes_dict['canny_people_on_background'] = True
                    render_modes_dict['using_yolo_network'] = True
                    print("canny_people_on_background")
                if mode == "e":
                    render_modes_dict['cartoon_effect'] = True
                    print("cartoon_effect")
                if mode == "f":
                    render_modes_dict['caffe_colorization'] = True
                    render_modes_dict['using_caffe_network'] = True
                    print("caffe_colorization")
                if mode == "g":
                    render_modes_dict['using_mask_rcnn_network'] = True
                    render_modes_dict['extract_and_cut_background'] = True
                    print("cannyPeopleRCNN + cut background")
                if mode == "h":
                    render_modes_dict['using_mask_rcnn_network'] = True
                    render_modes_dict['color_canny_on_background'] = True
                    print("color_canny_on_background")
                if mode == "i":
                    render_modes_dict['using_mask_rcnn_network'] = True
                    render_modes_dict['extract_and_replace_background'] = True
                    print("cannyPeopleRCNN + replace background")
                if mode == "j":
                    render_modes_dict['using_mask_rcnn_network'] = True
                    render_modes_dict['color_canny'] = True
                    print("color_canny")
                if mode == "k":
                    render_modes_dict['using_mask_rcnn_network'] = True
                    render_modes_dict['color_objects_on_gray'] = True
                    print("color_objects_on_gray")
                if mode == "l":
                    render_modes_dict['using_mask_rcnn_network'] = True
                    render_modes_dict['color_objects_on_gray_blur'] = True
                    print("color_objects_on_gray_blur")
                if mode == "m":
                    render_modes_dict['using_mask_rcnn_network'] = True
                    render_modes_dict['color_objects_blur'] = True
                    print("color_objects_on_gray_blur")
                if mode == "n":
                    render_modes_dict['upscale_opencv'] = True
                    print("imageUpscaler")
                if mode == "o":
                    render_modes_dict['denoise_and_sharpen'] = True
                    print("denoise_and_sharpen")
                if mode == "p":
                    render_modes_dict['sobel'] = True
                    print("sobel")
                if mode == "q":
                    render_modes_dict['ascii_painter'] = True
                    print("ascii_painter")
                if mode == "r":
                    render_modes_dict['pencil_drawer'] = True
                    print("pencil_drawer")
                if mode == "s":
                    render_modes_dict['two_colored'] = True
                    print("two_colored")
                if mode == "t":
                    render_modes_dict['upscale_esrgan'] = True
                    print("upscale_esrgan")
                if mode == "z":
                    render_modes_dict['boost_fps_dain'] = True
                    print("boost_fps_dain")

                need_mode_reset = False
        
        # Prepare settings if source is a video file or youtube/ipcam url
        if server_states.source_mode in ("video", "youtube", "ipcam"):
            # If stopped rendering
            if not started_rendering_video:
                # print("in stop loop")
                if (cap is not None):
                    cap.set(1, position_value) # Set current video position from HTML slider value
                    server_states.frame_processed = 0

                if need_to_stop_new_zip:
                    zip_obj.close()
                    zip_is_opened = False
                    need_to_stop_new_zip = False
                    need_to_create_new_zip = True
            else:
                # If started rendering
                if need_to_create_writer or file_changed:
                    # cap.set(1, 1)
                    server_states.frame_processed = 0
                    # cap.release()
                    if writer is not None:
                        writer.release()
                    if server_states.source_mode == "video":
                        # cap = cv2.VideoCapture(f"{app.config['UPLOAD_FOLDER']}{file_to_render}")
                        server_states.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                        if (render_modes_dict['boost_fps_dain']):
                            # fps_out = cap.get(cv2.CAP_PROP_FRAME_COUNT) * 7
                            # Change FPS output with DAIN mode
                            writer = cv2.VideoWriter(
                                f"static/user_renders/output{args['port']}{file_to_render}.avi",
                                fourcc,
                                60,
                                (main_frame.shape[1], main_frame.shape[0]),
                                True,
                            )
                        else:
                            writer = cv2.VideoWriter(
                                f"static/user_renders/output{args['port']}{file_to_render}.avi",
                                fourcc,
                                25,
                                (main_frame.shape[1], main_frame.shape[0]),
                                True,
                            )

                    if server_states.source_mode == "youtube":
                        vPafy = pafy.new(server_states.source_url)
                        play = vPafy.streams[0]
                        # cap = cv2.VideoCapture(play.url)
                        server_states.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                        if (render_modes_dict['boost_fps_dain']):
                            # fps_out = cap.get(cv2.CAP_PROP_FRAME_COUNT) * 7
                            # Change FPS output with DAIN mode
                            writer = cv2.VideoWriter(
                                f"static/user_renders/output{args['port']}youtube.avi",
                                fourcc,
                                60,
                                (main_frame.shape[1], main_frame.shape[0]),
                                True,
                            )
                        else:
                            writer = cv2.VideoWriter(
                                f"static/user_renders/output{args['port']}youtube.avi",
                                fourcc,
                                25,
                                (main_frame.shape[1], main_frame.shape[0]),
                                True,
                            )

                    if server_states.source_mode == "ipcam":
                        # source_url = str(settings_ajax["urlSource"])
                        cap = cv2.VideoCapture()
                        cap.open(server_states.source_url)
                        server_states.total_frames = 1
                        # server_states.source_lock = False
                        writer = cv2.VideoWriter(
                            f"static/user_renders/output{args['port']}ipcam.avi",
                            fourcc,
                            25,
                            (main_frame.shape[1], main_frame.shape[0]),
                            True,
                        )
                    # print("CREATING WRITER 1 WITH SIZE:" + str(round(main_frame.shape[1])))

                    # Prepare zip opening for YOLO objects
                    if need_to_create_new_zip:
                        zip_obj = ZipFile(f"static/user_renders/output{args['port']}.zip", "w")
                        need_to_stop_new_zip = True
                        need_to_create_new_zip = False
                        zip_is_opened = True
                    if file_changed:
                        zip_obj = ZipFile(f"static/user_renders/output{args['port']}.zip", "w")
                        zip_is_opened = True
                    file_changed = False
                    need_to_create_writer = False

            # Fill f and f1 pair of frames for DAIN interpolation
            if (render_modes_dict['boost_fps_dain']):
                if (started_rendering_video):
                    if (frame_interp_num == 0):
                        cap.set(1, 0)
                        ret, f = cap.read()
                        ret, f1 = cap.read()
                        if (f1 is not None):
                            main_frame = f1.copy()
                            frame_interp_num += 1
                    else:
                        f = frameEdge
                        ret, f1 = cap.read()

                        if (f1 is not None):
                            main_frame = f1.copy()
                        else:
                            main_frame = None
                else:
                    ret, main_frame = cap.read()
                    ret2, frame_background = cap2.read()
            # ... otherwise read by one frame
            else:
                if (cap is not None):
                    ret, main_frame = cap.read()
                    ret2, frame_background = cap2.read()

        # Prepare settings for image file
        if server_states.source_mode == "image":
            # Prepare zip opening for YOLO objects
            if received_zip_command or file_changed:
                zipped_images = False
                zip_obj = ZipFile(f"static/user_renders/output{args['port']}.zip", "w")
                zip_is_opened = True
                received_zip_command = False
                # print("CREATED ZIP =========================")

            if file_changed:
                zip_obj = ZipFile(f"static/user_renders/output{args['port']}.zip", "w")
                zip_is_opened = True
                file_changed = False
                need_to_create_writer = False

            main_frame = cv2.imread(f"{app.config['UPLOAD_FOLDER']}{server_states.source_image}")
            ret2, frame_background = cap2.read()

        classes_index = []
        start_moment = time.time()  # Timer for FPS calculation
        
        # =============================== Draw frame with render modes and settings ===============================

        if main_frame is not None:
            if not server_states.view_source:
                main_frame, frame_boost_sequence, frame_boost_list, classes_index, zipped_images, zip_obj, zip_is_opened = \
                    render_with_mode(render_modes_dict, settings_ajax, main_frame, frame_background, f, f1, yolo_network,
                                     rcnn_network, caffe_network, superres_network, dain_network, esrgan_network,
                                     device, output_layers, classes_index, zip_obj, zip_is_opened, zipped_images,
                                     server_states, started_rendering_video)

            with lock:
                check_if_user_is_connected(timer_start, 7) # Terminate process if browser tab was closed
                server_states.frame_processed += 1

                elapsed_time = time.time()
                fps = 1 / (elapsed_time - start_moment)

                # Resize frame for HTML preview with correct aspect ratio
                x_coeff = 460 / main_frame.shape[0]
                x_size = round(x_coeff * main_frame.shape[1])
                resized = cv2.resize(main_frame, (x_size, 460))

                if render_modes_dict['extract_objects_yolo_mode'] and not server_states.view_source:
                    resized = draw_yolo_stats(resized, classes_index, font)

                if server_states.source_mode == "image":
                    cv2.imwrite(
                        f"static/user_renders/output{args['port']}{server_states.source_image}",
                        main_frame,
                    )

                if (
                        server_states.source_mode == "image"
                        and render_modes_dict['extract_and_replace_background']
                        and writer is not None
                ):
                    writer.write(main_frame)

                # Two frames in one example
                # resized1 = cv2.resize(frameList[streamIndex], (640, 360))
                # resized2 = cv2.resize(main_frame, (640, 360))
                # concated = cv2.vconcat([resized2, resized1, ])
                # resized = cv2.resize(main_frame, (1600, 900))

                # Write DAIN interpolated frames to file
                if (
                        server_states.source_mode in ("video", "youtube", "ipcam")
                        and writer is not None
                        and started_rendering_video
                ):
                    if render_modes_dict['boost_fps_dain'] and started_rendering_video:
                        frame_boost_sequence, frame_boost_list = zip(
                            *sorted(zip(frame_boost_sequence, frame_boost_list)))
                        frameEdge = frame_boost_list[len(frame_boost_list)-1]

                        for i in range (len(frame_boost_list) - 1):
                            writer.write(frame_boost_list[i])
                            # cv2.imshow("video", frame)
                            key = cv2.waitKey(1) & 0xFF

                            if key == ord("q"):
                                break

                    else:
                        writer.write(main_frame)

                # Preview rendering on server
                cv2.imshow("video", main_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                # Calculate progress
                if server_states.source_mode in ("video", "youtube"):
                    if server_states.total_frames != 0:
                        progress = (
                                server_states.frame_processed
                                / server_states.total_frames
                                * 100
                        )

                # Draw stats on frame

                cv2.putText(
                    resized,
                    f"FPS: {str(round(fps, 2))} ({str(main_frame.shape[1])}x{str(main_frame.shape[0])})",
                    (40, 35),
                    font,
                    0.8,
                    (0, 0, 255),
                    2,
                    lineType=cv2.LINE_AA
                )

                if started_rendering_video:
                    out_file = ""
                    if server_states.source_mode == "youtube":
                        out_file = server_states.output_file_page
                    if server_states.source_mode in ("video", "image"):
                        out_file = f"output{args['port']}{file_to_render}"

                    cv2.putText(
                        resized,
                        f"Writing to '{out_file}' ({round(progress, 2)}%)",
                        (40, resized.shape[0] - 20),
                        font,
                        0.8,
                        (255, 0, 255),
                        2,
                        lineType=cv2.LINE_AA
                    )

                # Copy resized frame to HTML output
                output_frame = resized

                # Need for communication between start page and user process
                if server_states.frame_processed == 1:
                    print("started")

                # Take screenshot if needed
                if server_states.need_to_create_screenshot:
                    server_states.need_to_create_screenshot = False

                    print("Taking screenshot...")
                    cv2.imwrite(
                        f"static/user_renders/output{args['port']}Screenshot.png", main_frame
                    )
                    time.sleep(0.5)
                    server_states.screenshot_path = (
                        f"static/user_renders/output{args['port']}Screenshot.png"
                    )
                    server_states.screenshot_ready = True

                if (server_states.source_mode == "image"):
                    started_rendering_video = False
        # ... otherwise stop rendering
        else:
            zip_obj.close()
            check_if_user_is_connected(timer_start, 7)
            started_rendering_video = False
            if (writer):
                writer.release()
            position_value = 1
            # print("==================== finished ====================")


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


def generate():
    global output_frame, lock, server_states

    while server_states.working_on:
        with lock:
            if output_frame is None:
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encoded_image) + b"\r\n"
        )
    print("yield finished")


@app.route("/", methods=["GET", "POST"])
def index(device=None, action=None):
    global cap, cap2, file_to_render, file_changed, server_states

    if request.method == "POST":
        file = None
        textbox_string = ""

        if 'file' in request.files:
            file = request.files["file"]
            print("in file")

        if 'textbox' in request.form:
            textbox_string = request.form.get("textbox")

        # print(textbox_string.find("you"))

        if textbox_string.find("youtu") != -1:
            server_states.source_mode = "youtube"
            server_states.source_url = textbox_string
            vPafy = pafy.new(textbox_string)
            play = vPafy.streams[0]
            cap = cv2.VideoCapture(play.url)
            server_states.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            file_changed = True

        if textbox_string.find("mjpg") != -1:
            server_states.source_mode = "ipcam"
            server_states.source_url = textbox_string
            cap = cv2.VideoCapture()
            cap.open(textbox_string)
            server_states.total_frames = 1
            file_changed = True

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            file_extension = filename.rsplit(".", 1)[1]

            if file_extension in("png", "jpg", "jpeg"):
                server_states.source_mode = "image"
                server_states.source_image = filename
                cap2 = cv2.VideoCapture("input_videos/space.webm")
            if file_extension in ("gif", "mp4", "avi", "m4v", "webm", "mkv"):
                server_states.source_mode = "video"
                cap = cv2.VideoCapture(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                server_states.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap2 = cv2.VideoCapture("input_videos/space.webm")
            
            CRED = "\033[91m"
            CEND = "\033[0m"
            print(
                CRED
                + f"==============  file {filename} uploaded ============== "
                + CEND
            )

            file_to_render = filename
            file_changed = True

    if server_states.source_mode == "video":
        server_states.output_file_page = file_to_render + ".avi"
    if server_states.source_mode == "youtube":
        server_states.output_file_page = "youtube.avi"
    if server_states.source_mode == "ipcam":
        server_states.output_file_page = "ipcam.avi"

    print("server_states.source_mode")

    return render_template(
        "index.html",
        frame_processed=server_states.frame_processed,
        pathToRenderedFile=f"static/user_renders/output{args['port']}{server_states.output_file_page}",
        pathToZipFile=f"static/user_renders/output{args['port']}.zip",
    )


@app.route("/video")
def video_feed():
    # redirect(f"http://192.168.0.12:8000/results")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats", methods=["POST"])
def send_stats():
    global server_states, user_time

    # timer_start = time.perf_counter()
    frame_width_to_page = 0
    frame_height_to_page = 0
    screenshot_ready_local = False

    if server_states.screenshot_ready:
        screenshot_ready_local = True
        server_states.screenshot_ready = False

    if server_states.source_mode in ("video", "youtube", "ipcam"):
        if (cap != None):
            frame_width_to_page = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height_to_page = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if server_states.source_mode == "image":
        frame_width_to_page = 0
        frame_height_to_page = 0

    return jsonify(
        {
            "value": server_states.frame_processed,
            "totalFrames": server_states.total_frames,
            "progress": round(progress, 2),
            "fps": round(fps, 2),
            "workingOn": server_states.working_on,
            "cpuUsage": psutil.cpu_percent(),
            "freeRam": round((psutil.virtual_memory()[1] / 2.0 ** 30), 2),
            "ramPercent": psutil.virtual_memory()[2],
            "frameWidth": frame_width_to_page,
            "frameHeight": frame_height_to_page,
            "currentMode": server_states.render_mode,
            "userTime": user_time,
            "screenshotReady": screenshot_ready_local,
            "screenshotPath": server_states.screenshot_path
            # 'time': datetime.datetime.now().strftime("%H:%M:%S"),
        }
    )


@app.route("/settings", methods=["GET", "POST"])
def receive_settings():
    global settings_ajax, timer_start, timer_end, writer, server_states, commands

    if request.method == "POST":
        # print("POST")
        timer_start = time.perf_counter()
        settings_ajax = request.get_json()

        if not server_states.mode_reset_lock:
            if bool(settings_ajax["modeResetCommand"]):
                server_states.mode_reset_lock = True

        if not server_states.video_stop_lock:
            if bool(settings_ajax["videoStopCommand"]):
                server_states.video_stop_lock = True

        if not server_states.video_reset_lock:
            if bool(settings_ajax["videoResetCommand"]):
                server_states.video_reset_lock = True

        if not server_states.screenshot_lock:
            if bool(settings_ajax["screenshotCommand"]):
                server_states.screenshot_lock = True
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

    t = threading.Thread(target=process_frame)
    t.daemon = True
    t.start()

    app.run(
        host=args["ip"],
        port=args["port"],
        debug=False,
        threaded=True,
        use_reloader=False,
    )
