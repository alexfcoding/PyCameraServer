# python processing.py -i 192.168.0.12 -o 8002 -s fabio.webm -c a -m video

from flask import jsonify
from flask import Flask
from flask import render_template
import threading
import argparse
from flask import request, Response
import psutil
import time
from render_modes import *
from werkzeug.utils import secure_filename
from zipfile import ZipFile
import pafy


class Command:
    video_reset_command = False
    video_stop_command = False
    mode_reset_command = False
    screenshot_command = False


class ServerState:
    source_image = ""
    source_mode = ""
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


commands = Command()
server_states = ServerState()

timer_start = 0
timer_end = 0
user_time = 0
input_data = None
thr = None
output_frame = None
resized = None
value = 0
running = False
progress = 0
fps = 0
cap = None
cap2 = None
lock = threading.Lock()

app = Flask(__name__, static_url_path="/static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

stream_list = ["videoplayback.mp4"]

# Working adresses:
# http://94.72.19.58/mjpg/video.mjpg,
# http://91.209.234.195/mjpg/video.mjpg
# http://209.194.208.53/mjpg/video.mjpg
# http://66.57.117.166:8000/mjpg/video.mjpg
main_frame = None
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None


def check_if_user_is_connected(timer_start):
    global user_time
    timer_end = time.perf_counter()
    user_time = str(round(timer_end)) + ":" + str(round(timer_start))

    if timer_end - timer_start < 7 and timer_start != 0:
        print("User is connected")
    else:
        if timer_start != 0:
            print(
                "User disconnected, shutting down!"
            )
            current_pid = os.getpid()
            p = psutil.Process(current_pid)
            p.terminate()  # or p.kill()


def process_frame():
    global cap, lock, writer, progress, fps, output_frame, file_to_render, zip_obj, youtube_url

    blur_canny_value = 5
    position_value = 1
    saturation_value = 100
    contrast_value = 100
    brightness_value = 0
    confidence_value = 0
    line_thickness_value = 1
    denoise_value = 10
    denoise_value2 = 10
    sharpening_value = 9
    rcnn_size_value = 10
    rcnn_blur_value = 17
    object_index = 0
    sobel_value = 3
    sharpening_value2 = 5
    color_count_value = 32
    resize_value = 2

    server_states.frame_processed = 0
    server_states.total_frames = 0

    frame_background = None
    received_zip_command = False
    file_changed = False
    started_rendering_video = False

    need_mode_reset = True
    server_states.working_on = True

    concated = None
    need_to_create_new_zip = True
    need_to_stop_new_zip = False
    zip_is_opened = False
    zipped_images = False
    font = cv2.FONT_HERSHEY_SIMPLEX

    file_to_render = args["source"]
    youtube_url = args["source"]
    server_states.options = args["optionsList"]
    server_states.source_mode = args["mode"]

    if server_states.source_mode == "youtube":
        vPafy = pafy.new(youtube_url)
        play = vPafy.streams[1]
        cap = cv2.VideoCapture(play.url)
        server_states.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if server_states.source_mode == "video":
        cap = cv2.VideoCapture(file_to_render)
        server_states.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if server_states.source_mode == "image":
        server_states.source_image = args["source"]

    cap2 = cv2.VideoCapture("input_videos/snow.webm")

    zip_obj = ZipFile(f"static/objects{args['port']}.zip", "w")
    zip_is_opened = True

    while server_states.working_on:
        if input_data is not None:
            blur_canny_value = int(input_data["cannyBlurSliderValue"])
            saturation_value = int(input_data["saturationSliderValue"])
            contrast_value = int(input_data["contrastSliderValue"])
            brightness_value = int(input_data["brightnessSliderValue"])
            confidence_value = int(input_data["confidenceSliderValue"])
            line_thickness_value = int(input_data["lineThicknessSliderValue"])
            denoise_value = int(input_data["denoiseSliderValue"])
            denoise_value2 = int(input_data["denoise2SliderValue"])
            sharpening_value = int(input_data["sharpenSliderValue"])
            sharpening_value2 = int(input_data["sharpenSliderValue2"])
            rcnn_size_value = int(input_data["rcnnSizeSliderValue"])
            rcnn_blur_value = int(input_data["rcnnBlurSliderValue"])
            sobel_value = int(input_data["sobelSliderValue"])
            ascii_size_value = int(input_data["asciiSizeSliderValue"])
            ascii_interval_value = int(input_data["asciiIntervalSliderValue"])
            ascii_thickness_value = int(input_data["asciiThicknessSliderValue"])
            resize_value = int(input_data["resizeSliderValue"]) / 100
            color_count_value = int(input_data["color_countSliderValue"])
            position_valueLocal = int(input_data["positionSliderValue"])

            if commands.mode_reset_command != "default":
                server_states.options = commands.mode_reset_command
                need_mode_reset = True

            if server_states.video_reset_lock:
                position_value = 1
                need_to_create_writer = True
                started_rendering_video = True
                received_zip_command = True
                server_states.video_reset_lock = False
                print("in loop reset")
            else:
                position_value = position_valueLocal

            if server_states.video_stop_lock:
                position_value = 1
                started_rendering_video = False
                server_states.video_stop_lock = False
                print("in loop stop")

            if server_states.screenshot_lock:
                print("in loop screenshot")
                server_states.need_to_create_screenshot = True
                server_states.screenshot_lock = False

        # print("working...")
        if need_mode_reset:
            using_yolo_network = False
            using_caffe_network = False
            using_mask_rcnn_network = False
            canny_people_on_background = False
            canny_people_on_black = False
            extract_and_replace_background = False
            extract_and_cut_background = False
            color_canny = False
            color_canny_on_background = False
            color_objects_on_gray_blur = False
            color_objects_blur = False
            color_objects_on_gray = False
            caffe_colorization = False
            cartoon_effect = False
            extract_objects_yolo_mode = False
            text_render_yolo = False
            denoise_and_sharpen = False
            sobel = False
            ascii_painter = False
            pencil_drawer = False
            two_colored = False
            upscale_opencv = False

            for char in server_states.options:
                if char == "a":
                    extract_objects_yolo_mode = True
                    using_yolo_network = True
                    print("extract_objects_yolo")
                if char == "b":
                    text_render_yolo = True
                    using_yolo_network = True
                    print("text_render_yolo")
                if char == "c":
                    canny_people_on_black = True
                    using_yolo_network = True
                    print("canny_people_on_black")
                if char == "d":
                    canny_people_on_background = True
                    using_yolo_network = True
                    print("canny_people_on_background")
                if char == "e":
                    cartoon_effect = True
                    print("cartoon_effect")
                if char == "f":
                    caffe_colorization = True
                    using_caffe_network = True
                    print("caffe_colorization")
                if char == "g":
                    using_mask_rcnn_network = True
                    extract_and_cut_background = True
                    print("cannyPeopleRCNN + cut background")
                if char == "h":
                    using_mask_rcnn_network = True
                    color_canny_on_background = True
                    print("color_canny_on_background")
                if char == "i":
                    using_mask_rcnn_network = True
                    extract_and_replace_background = True
                    print("cannyPeopleRCNN + replace background")
                if char == "j":
                    using_mask_rcnn_network = True
                    color_canny = True
                    print("color_canny")
                if char == "k":
                    using_mask_rcnn_network = True
                    color_objects_on_gray = True
                    print("color_objects_on_gray")
                if char == "l":
                    using_mask_rcnn_network = True
                    color_objects_on_gray_blur = True
                    print("color_objects_on_gray_blur")
                if char == "m":
                    using_mask_rcnn_network = True
                    color_objects_blur = True
                    print("color_objects_on_gray_blur")
                if char == "n":
                    upscale_opencv = True
                    print("imageUpscaler")
                if char == "o":
                    denoise_and_sharpen = True
                    print("denoise_and_sharpen")
                if char == "p":
                    sobel = True
                    print("sobel")
                if char == "q":
                    ascii_painter = True
                    print("ascii_painter")
                if char == "r":
                    pencil_drawer = True
                    print("pencil_drawer")
                if char == "s":
                    two_colored = True
                    print("two_colored")

                need_mode_reset = False

        classes_index = []
        start_moment = time.time()

        if server_states.source_mode in ("video", "youtube"):
            if started_rendering_video == False:
                cap.set(1, position_value)
                if need_to_stop_new_zip:
                    zip_obj.close()
                    zip_is_opened = False
                    need_to_stop_new_zip = False
                    need_to_create_new_zip = True
            else:
                if need_to_create_writer or file_changed:
                    cap.set(1, 1)
                    server_states.frame_processed = 0
                    cap.release()
                    if file_changed:
                        print("1")
                    if writer is not None:
                        writer.release()

                    if server_states.source_mode == "video":
                        cap = cv2.VideoCapture(file_to_render)
                        server_states.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                    if server_states.source_mode == "youtube":
                        cap = cv2.VideoCapture(play.url)
                        server_states.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                    writer = cv2.VideoWriter(
                        f"static/output{args['port']}{file_to_render}.avi" f"",
                        fourcc,
                        25,
                        (main_frame.shape[1], main_frame.shape[0]),
                        True,
                    )

                    # print("CREATING WRITER 1 WITH SIZE:" + str(round(main_frame.shape[1])))

                    if need_to_create_new_zip:
                        zip_obj = ZipFile(f"static/objects{args['port']}.zip", "w")
                        need_to_stop_new_zip = True
                        need_to_create_new_zip = False
                        zip_is_opened = True

                    if file_changed:
                        zip_obj = ZipFile(f"static/objects{args['port']}.zip", "w")
                        zip_is_opened = True

                    file_changed = False
                    commands.video_reset_command = False
                    need_to_create_writer = False

            ret, main_frame = cap.read()
            ret2, frame_background = cap2.read()

        if server_states.source_mode == "image":
            if received_zip_command or file_changed:
                zipped_images = False
                zip_obj = ZipFile(f"static/objects{args['port']}.zip", "w")
                zip_is_opened = True
                received_zip_command = False

            if file_changed:
                zip_obj = ZipFile(f"static/objects{args['port']}.zip", "w")
                zip_is_opened = True
                file_changed = False
                need_to_create_writer = False

            main_frame = cv2.imread(server_states.source_image)
            ret2, frame_background = cap2.read()

        if main_frame is not None:
            if using_yolo_network:
                boxes, indexes, class_ids, confidences, classes_out = find_yolo_classes(
                    main_frame, yolo_network, output_layers, confidence_value
                )
                classes_index.append(classes_out)

                if extract_objects_yolo_mode:
                    main_frame = extract_objects_yolo(
                        main_frame,
                        boxes,
                        indexes,
                        class_ids,
                        confidences,
                        zip_obj,
                        zip_is_opened,
                        zipped_images,
                        server_states.source_mode,
                        started_rendering_video,
                    )

                if server_states.source_mode == "image" and zip_is_opened:
                    zip_obj.close()

                if server_states.source_mode == "image" and zipped_images == False:
                    zipped_images = True
                    zip_is_opened = False

                if text_render_yolo:
                    main_frame = objects_to_text_yolo(
                        main_frame,
                        boxes,
                        indexes,
                        class_ids,
                        ascii_size_value,
                        ascii_interval_value,
                        rcnn_blur_value,
                        ascii_thickness_value,
                    )

                if canny_people_on_black:
                    main_frame = canny_people_on_black_yolo(
                        main_frame, boxes, indexes, class_ids
                    )

                if canny_people_on_background:
                    main_frame = canny_people_on_background_yolo(
                        main_frame, boxes, indexes, class_ids
                    )

            if using_mask_rcnn_network:
                boxes, masks, labels, colors = find_rcnn_classes(main_frame, rcnn_network)

                if color_objects_on_gray:
                    main_frame = colorizer_people_rcnn(
                        main_frame, boxes, masks, confidence_value, rcnn_size_value
                    )

                if color_objects_on_gray_blur:
                    main_frame = colorizer_people_with_blur_rcnn(
                        main_frame, boxes, masks, confidence_value
                    )
                if color_objects_blur:
                    main_frame = people_with_blur_rcnn(
                        main_frame,
                        boxes,
                        masks,
                        labels,
                        confidence_value,
                        rcnn_size_value,
                        rcnn_blur_value,
                    )

                if extract_and_cut_background:
                    main_frame = extract_and_cut_background_rcnn(
                        main_frame, boxes, masks, labels, confidence_value
                    )

                if extract_and_replace_background:
                    main_frame = extract_and_replace_background_rcnn(
                        main_frame,
                        frame_background,
                        boxes,
                        masks,
                        labels,
                        colors,
                        confidence_value,
                    )

                if color_canny:
                    main_frame = color_canny_rcnn(
                        main_frame,
                        boxes,
                        masks,
                        labels,
                        confidence_value,
                        rcnn_blur_value,
                    )

                if color_canny_on_background:
                    main_frame = color_canny_on_color_background_rcnn(
                        main_frame, boxes, masks, labels, confidence_value
                    )

            if using_caffe_network:
                if caffe_colorization:
                    main_frame = colorizer_caffe(caffe_network, main_frame)

            if cartoon_effect:
                frame_copy = main_frame.copy()

                if blur_canny_value % 2 == 0:
                    blur_canny_value += 1
                    main_frame = cv2.GaussianBlur(
                        main_frame,
                        (blur_canny_value, blur_canny_value),
                        blur_canny_value,
                    )
                else:
                    main_frame = cv2.GaussianBlur(
                        main_frame,
                        (blur_canny_value, blur_canny_value),
                        blur_canny_value,
                    )

                main_frame = cv2.Canny(main_frame, 50, 50)
                main_frame = cv2.cvtColor(main_frame, cv2.COLOR_GRAY2BGR)
                kernel = np.ones((line_thickness_value, line_thickness_value), np.uint8)
                main_frame = cv2.dilate(main_frame, kernel, iterations=1)
                frame_copy[np.where((main_frame > [0, 0, 0]).all(axis=2))] = [0, 0, 0]
                frame_copy = limit_colors_kmeans(frame_copy, color_count_value)
                # frame_copy = cv2.GaussianBlur(frame_copy, (3, 3), 2)
                main_frame = frame_copy
                main_frame = sharpening(main_frame, sharpening_value, sharpening_value2)
                main_frame = denoise(main_frame, denoise_value, denoise_value2)

            if pencil_drawer:
                frame_copy = main_frame.copy()

                if blur_canny_value % 2 == 0:
                    blur_canny_value += 1
                    main_frame = cv2.GaussianBlur(
                        main_frame,
                        (blur_canny_value, blur_canny_value),
                        blur_canny_value,
                    )
                else:
                    main_frame = cv2.GaussianBlur(
                        main_frame,
                        (blur_canny_value, blur_canny_value),
                        blur_canny_value,
                    )

                # main_frame = morph_edge_detection(main_frame)
                main_frame = cv2.Canny(main_frame, 50, 50)
                main_frame = cv2.cvtColor(main_frame, cv2.COLOR_GRAY2BGR)
                kernel = np.ones((line_thickness_value, line_thickness_value), np.uint8)
                main_frame = cv2.dilate(main_frame, kernel, iterations=1)
                frame_copy[np.where((main_frame > [0, 0, 0]).all(axis=2))] = [0, 0, 0]
                frame_copy = limit_colors_kmeans(frame_copy, 2)
                # frame_copy = cv2.GaussianBlur(frame_copy, (3, 3), 2)
                main_frame = frame_copy
                main_frame = sharpening(main_frame, sharpening_value, sharpening_value2)
                main_frame = denoise(main_frame, denoise_value, denoise_value2)
                # main_frame = np.bitwise_not(main_frame)

            if two_colored:
                frame_copy = main_frame.copy()
                # main_frame = morph_edge_detection(main_frame)
                kernel = np.ones((line_thickness_value, line_thickness_value), np.uint8)
                # main_frame = cv2.dilate(main_frame,kernel,iterations = 1)
                # frame_copy[np.where((main_frame > [0, 0, 0]).all(axis=2))] = [0,0,0]
                frame_copy = limit_colors_kmeans(frame_copy, 2)
                # frame_copy = cv2.GaussianBlur(frame_copy, (3, 3), 2)
                main_frame = frame_copy
                main_frame = sharpening(main_frame, sharpening_value, sharpening_value2)
                main_frame = denoise(main_frame, denoise_value, denoise_value2)

            if upscale_opencv:
                main_frame = upscale_image(superres_network, main_frame)
                main_frame = sharpening(main_frame, sharpening_value, sharpening_value2)

            if ascii_painter:
                main_frame = ascii_paint(
                    main_frame,
                    ascii_size_value,
                    ascii_interval_value,
                    ascii_thickness_value,
                    rcnn_blur_value,
                )

            if denoise_and_sharpen:
                main_frame = sharpening(main_frame, sharpening_value, sharpening_value2)
                main_frame = denoise(main_frame, denoise_value, denoise_value2)

            if sobel:
                main_frame = denoise(main_frame, denoise_value, denoise_value2)
                main_frame = sharpening(main_frame, sharpening_value, sharpening_value2)
                grad_x = cv2.Sobel(main_frame, cv2.CV_64F, 1, 0, ksize=sobel_value)
                grad_y = cv2.Sobel(main_frame, cv2.CV_64F, 0, 1, ksize=sobel_value)
                main_frame = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

            main_frame = adjust_br_contrast(main_frame, contrast_value, brightness_value)
            main_frame = adjust_saturation(main_frame, saturation_value)

            with lock:
                personDetected = False
                check_if_user_is_connected(timer_start)

                server_states.frame_processed = server_states.frame_processed + 1
                elapsedTime = time.time()
                fps = 1 / (elapsedTime - start_moment)
                # print(fps)
                xCoeff = 512 / main_frame.shape[0]
                xSize = round(xCoeff * main_frame.shape[1])
                resized = cv2.resize(main_frame, (xSize, 512))

                if extract_objects_yolo_mode:
                    class_index_count = [
                        [0 for x in range(80)] for x in range(len(stream_list))
                    ]

                    row_index = 1
                    for m in range(80):
                        for k in range(len(classes_index[0])):
                            if m == classes_index[0][k]:
                                class_index_count[0][m] += 1

                        if class_index_count[0][m] != 0:
                            row_index += 1

                            if classes[m] == "person":
                                cv2.rectangle(
                                    resized,
                                    (20, row_index * 40 - 25),
                                    (270, row_index * 40 + 11),
                                    (0, 0, 0),
                                    -1,
                                )
                                cv2.putText(
                                    resized,
                                    classes[m] + ": " + str(class_index_count[0][m]),
                                    (40, row_index * 40),
                                    font,
                                    1,
                                    (0, 255, 0),
                                    2,
                                    lineType=cv2.LINE_AA,
                                )
                                personDetected = True

                            if classes[m] == "car":
                                cv2.rectangle(
                                    resized,
                                    (20, row_index * 40 - 25),
                                    (270, row_index * 40 + 11),
                                    (0, 0, 0),
                                    -1,
                                )
                                cv2.putText(
                                    resized,
                                    classes[m] + ": " + str(class_index_count[0][m]),
                                    (40, row_index * 40),
                                    font,
                                    1,
                                    (255, 0, 255),
                                    2,
                                    lineType=cv2.LINE_AA,
                                )

                            if (classes[m] != "car") & (classes[m] != "person"):
                                cv2.rectangle(
                                    resized,
                                    (20, row_index * 40 - 25),
                                    (270, row_index * 40 + 11),
                                    (0, 0, 0),
                                    -1,
                                )
                                cv2.putText(
                                    resized,
                                    classes[m] + ": " + str(class_index_count[0][m]),
                                    (40, row_index * 40),
                                    font,
                                    1,
                                    colors_yolo[m],
                                    2,
                                    lineType=cv2.LINE_AA,
                                )

                            if (classes[m] == "handbag") | (classes[m] == "backpack"):
                                passFlag = True
                                print("handbag detected! -> PASS")

                if server_states.source_mode == "image":
                    cv2.imwrite(
                        f"static/output{args['port']}{server_states.source_image}",
                        main_frame,
                    )

                if (
                    server_states.source_mode == "image"
                    and extract_and_replace_background == True
                    and writer is not None
                ):
                    writer.write(main_frame)

                # resized1 = cv2.resize(frameList[streamIndex], (640, 360))
                # resized2 = cv2.resize(main_frame, (640, 360))
                # concated = cv2.vconcat([resized2, resized1, ])
                # resized = cv2.resize(main_frame, (1600, 900))

                if (
                    server_states.source_mode in ("video", "youtube")
                    and writer is not None
                    and started_rendering_video
                ):
                    writer.write(main_frame)

                cv2.imshow("video", main_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if server_states.source_mode in ("video", "youtube"):
                    if server_states.total_frames != 0:
                        progress = (
                            server_states.frame_processed
                            / server_states.total_frames
                            * 100
                        )

                cv2.putText(
                    resized,
                    f"FPS: {str(round(fps, 2))} ({str(main_frame.shape[1])}x{str(main_frame.shape[0])})",
                    (40, 35),
                    font,
                    0.8,
                    (0, 0, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
                output_frame = resized

                if server_states.frame_processed == 1:
                    print("started")

                if server_states.need_to_create_screenshot == True:
                    print("screenshot")
                    cv2.imwrite(
                        f"static/output{args['port']}Screenshot.png", main_frame
                    )
                    time.sleep(1)
                    server_states.screenshot_path = (
                        f"static/output{args['port']}Screenshot.png"
                    )
                    server_states.screenshot_ready = True

        else:
            zip_obj.close()
            check_if_user_is_connected(timer_start)
            started_rendering_video = False
            position_value = 1
            print("finished")


UPLOAD_FOLDER = ""
ALLOWED_EXTENSIONS = set(
    ["png", "jpg", "jpeg", "mp4", "avi", "m4v", "webm", "mkv"]
)

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
        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            file_extension = filename.rsplit(".", 1)[1]

            if (
                file_extension == "png"
                or file_extension == "jpg"
                or file_extension == "jpeg"
            ):
                server_states.source_mode = "image"
                server_states.source_image = filename
                cap2 = cv2.VideoCapture("input_videos/snow.webm")
            else:
                server_states.source_mode = "video"
                cap = cv2.VideoCapture(filename)
                server_states.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap2 = cv2.VideoCapture("input_videos/snow.webm")

            server_states.options = request.form.getlist("check")
            mode = request.form.getlist("checkMode")

            CRED = "\033[91m"
            CEND = "\033[0m"
            print(
                CRED
                + f"==============  file {filename} uploaded ============== "
                + CEND
            )

            file_to_render = filename
            file_changed = True

    file_output = file_to_render

    if server_states.source_mode in ("video", "youtube"):
        file_output = file_to_render + ".avi"

    return render_template(
        "index.html",
        frame_processed=server_states.frame_processed,
        pathToRenderedFile=f"static/output{args['port']}{file_output}",
        pathToZipFile=f"static/objects{args['port']}.zip",
    )


@app.route("/video")
def video_feed():
    # redirect(f"http://192.168.0.12:8000/results")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats", methods=["POST"])
def send_stats():
    global server_states, user_time

    timer_start = time.perf_counter()
    frame_width_to_page = 0
    frame_height_to_page = 0
    screenshot_ready_local = False

    if server_states.screenshot_ready:
        screenshot_ready_local = True
        server_states.screenshot_ready = False
        server_states.need_to_create_screenshot = False

    if server_states.source_mode in ("video", "youtube"):
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
            "currentMode": server_states.options,
            "userTime": user_time,
            "screenshotReady": screenshot_ready_local,
            "screenshotPath": server_states.screenshot_path
            # 'time': datetime.datetime.now().strftime("%H:%M:%S"),
        }
    )


@app.route("/settings", methods=["GET", "POST"])
def receive_settings():
    global input_data, timer_start, timer_end, writer, server_states, commands

    if request.method == "POST":
        timer_start = time.perf_counter()
        input_data = request.get_json()

        commands.mode_reset_command = str(input_data["modeResetCommand"])

        if not server_states.video_stop_lock:
            if bool(input_data["videoStopCommand"]) == True:
                server_states.video_stop_lock = True

        if not server_states.video_reset_lock:
            if bool(input_data["videoResetCommand"]) == True:
                server_states.video_reset_lock = True

        if not server_states.screenshot_lock:
            if bool(input_data["screenshotCommand"]) == True:
                server_states.screenshot_lock = True

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
        help="ephemeral port number of the server (1024 to 65535)",
    )
    ap.add_argument("-s", "--source", type=str, default=32, help="# file to render")
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
