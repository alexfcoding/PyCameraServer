import threading
import psutil
from mode_selector import *
from zipfile import ZipFile
import pafy
import cv2
from settings import states_dict, render_modes_dict, settings_ajax

timer_start = 0  # Start timer for stopping rendering if user closed tab
timer_end = 0  # End timer for stopping rendering if user closed tab
user_time = 0  # For user timer debug
output_frame = None  # Frame to preview on page
progress = 0  # Rendering progress 0-100%
fps = 0  # Rendering fps
cap = None  # VideoCapture object for user frames
cap2 = None  # VideoCapture object for secondary video (need for some effects)
lock = threading.Lock()  # Lock for thread (multiple browser connections viewing)
main_frame = None  # Processing frame from video, image or youtube URL
frame_background = None  # Frame for secondary video
fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # Format for video saving
writer = None  # Writer for video saving
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

    if not (timer_end - timer_start < seconds_to_disconnect and timer_start != 0):
        print("User is connected")
        if timer_start != 0:
            print(
                "User disconnected, shutting down!"
            )
            current_pid = os.getpid()
            p = psutil.Process(current_pid)
            p.terminate()  # or p.kill()


def process_frame(args, app):
    """
    Main rendering function
    :return:
    """
    global cap, lock, writer, progress, fps, output_frame, file_to_render, zip_obj

    frame_boost_list = []  # List for Depth-Aware Video Frame Interpolation frames
    frame_boost_sequence = []  # Interpolated frame sequence for video writing
    states_dict['frame_processed'] = 0  # Total frames processed
    states_dict['total_frames'] = 0  # Total frames in video
    received_zip_command = False  # Trigger for receiving YOLO objects downloading command by user
    file_changed = False  # Trigger for file changing by user
    started_rendering_video = False  # Trigger for start rendering video by user
    need_mode_reset = True  # Trigger for rendering mode changing by user
    states_dict['working_on'] = True  # Rendering state is ON
    concated = None
    need_to_create_new_zip = True  # Loop state to open new zip archive
    need_to_stop_new_zip = False  # Loop state to close zip archive
    zip_is_opened = True  # Loop state for saving new images to zip archive
    zipped_images = False  # Loop state for closed zip archive
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font for rendering stats on frame by OpenCV
    resized = None  # Resized frame to put on page
    fps = 0  # FPS rate
    frame_edge = None  # Last frame of interpolation sequence
    path_to_file, file_to_render = os.path.split(args["source"])  # Get filename from full path
    print("Processing file: " + file_to_render)
    states_dict['source_url'] = args["source"]  # Youtube URL
    states_dict['render_mode'] = args["optionsList"]  # Rendering mode from command line
    # Set rendering mode from args and settings dictionary
    rendering_mode = [k for k, v in render_modes_dict.items() if v == args['optionsList']][0]
    states_dict['source_mode'] = args["mode"]  # Source type from command line

    # Set source for youtube capturing
    if states_dict['source_mode'] == "youtube":
        vPafy = pafy.new(states_dict['source_url'])
        play = vPafy.streams[0]
        cap = cv2.VideoCapture(play.url)
        states_dict['total_frames'] = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Set source for ip camera capturing
    if states_dict['source_mode'] == "ipcam":
        cap = cv2.VideoCapture()
        cap.open(states_dict['source_url'])
        states_dict['total_frames'] = 1

    # Set source for video file capturing
    if states_dict['source_mode'] == "video":
        cap = cv2.VideoCapture(f"{app.config['UPLOAD_FOLDER']}{file_to_render}")
        states_dict['total_frames'] = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Set source for image file capturing
    if states_dict['source_mode'] == "image":
        path_to_image, image_file = os.path.split(args["source"])
        states_dict['source_image'] = image_file

    cap2 = cv2.VideoCapture("input_videos/space.webm")  # Secondary video for background replacement
    zip_obj = ZipFile(f"static/user_renders/output{args['port']}.zip", "w")  # Zip file with user port name

    # Initialize all models
    caffe_network = initialize_caffe_network()
    caffe_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    caffe_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    superres_network = initialize_superres_network("LAPSRN")
    esrgan_network, device = initialize_esrgan_network("FALCOON", True)
    rcnn_network = initialize_rcnn_network(False)
    dain_network = initialize_dain_network(True)
    yolo_network, layers_names, output_layers, colors_yolo = initialize_yolo_network(classes, True)

    frame_interp_num = 0  # Interpolated frame number
    main_frame = None
    f = f1 = None  # Two source frames for interpolation

    # Main processing loop
    while states_dict['working_on']:
        # Receive all HTML slider values from JSON dictionary
        if settings_ajax is not None:
            mode_from_page = str(settings_ajax["mode"])
            superres_model_from_page = str(settings_ajax["superresModel"])
            esrgan_model_from_page = str(settings_ajax["esrganModel"])
            position_value_local = int(settings_ajax["positionSliderValue"])
            states_dict['view_source'] = bool(settings_ajax["viewSource"])

            # Check if mode change command was received
            if states_dict['mode_reset_lock']:
                states_dict['render_mode'] = mode_from_page
                rendering_mode = [k for k, v in render_modes_dict.items() if v == mode_from_page][0]
                states_dict['superres_model'] = superres_model_from_page
                states_dict['esrgan_model'] = esrgan_model_from_page
                states_dict['mode_reset_lock'] = False
                need_mode_reset = True

            # Check if video rendering start command was received
            if states_dict['video_reset_lock']:
                position_value = 1  # Reset position
                need_to_create_writer = True  # Create new writer
                started_rendering_video = True
                received_zip_command = True
                states_dict['video_reset_lock'] = False
                # print("in loop reset")
            else:
                position_value = position_value_local  # Read frame position from slider

            # Check if video rendering stop command was received
            if states_dict['video_stop_lock']:
                position_value = 1
                started_rendering_video = False
                states_dict['video_stop_lock'] = False
                # print("in loop stop")

            # Check if taking screenshot command was received
            if states_dict['screenshot_lock']:
                # print("in loop screenshot")
                states_dict['need_to_create_screenshot'] = True
                states_dict['screenshot_lock'] = False
        else:
            position_value = 1

        # If user changed rendering mode
        if need_mode_reset:
            frame_interp_num = 0
            # Reinitialize upscale networks with user models from page
            superres_network = initialize_superres_network(states_dict['superres_model'])
            esrgan_network, device = initialize_esrgan_network(states_dict['esrgan_model'], True)
            need_mode_reset = False

        # Prepare settings if source is a video file or youtube/ipcam url
        if states_dict['source_mode'] in ("video", "youtube", "ipcam"):
            # If stopped rendering
            if not started_rendering_video:
                # print("in stop loop")
                if cap is not None:
                    cap.set(1, position_value)  # Set current video position from HTML slider value
                    states_dict['frame_processed'] = 0

                if need_to_stop_new_zip:
                    zip_obj.close()
                    zip_is_opened = False
                    need_to_stop_new_zip = False
                    need_to_create_new_zip = True
            else:
                # If started rendering
                if need_to_create_writer or file_changed:
                    # cap.set(1, 1)
                    states_dict['frame_processed'] = 0
                    # cap.release()
                    if writer is not None:
                        writer.release()
                    if states_dict['source_mode'] == "video":
                        # cap = cv2.VideoCapture(f"{app.config['UPLOAD_FOLDER']}{file_to_render}")
                        states_dict['total_frames'] = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                        if rendering_mode == 'boost_fps_dain':
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

                    if states_dict['source_mode'] == "youtube":
                        vPafy = pafy.new(states_dict['source_url'])
                        play = vPafy.streams[0]
                        # cap = cv2.VideoCapture(play.url)
                        states_dict['total_frames'] = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                        if rendering_mode == 'boost_fps_dain':
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

                    if states_dict['source_mode'] == "ipcam":
                        # source_url = str(settings_ajax["urlSource"])
                        cap = cv2.VideoCapture()
                        cap.open(states_dict['source_url'])
                        states_dict['total_frames'] = 1
                        # states_dict['source_lock'] = False
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
            if rendering_mode == 'boost_fps_dain':
                if started_rendering_video:
                    if frame_interp_num == 0:
                        cap.set(1, 0)
                        ret, f = cap.read()
                        ret, f1 = cap.read()
                        if f1 is not None:
                            main_frame = f1.copy()
                            frame_interp_num += 1
                    else:
                        f = frame_edge
                        ret, f1 = cap.read()

                        if f1 is not None:
                            main_frame = f1.copy()
                        else:
                            main_frame = None
                else:
                    ret, main_frame = cap.read()
                    ret2, frame_background = cap2.read()
            # ... otherwise read by one frame
            else:
                if cap is not None:
                    ret, main_frame = cap.read()
                    ret2, frame_background = cap2.read()

        # Prepare settings for image file
        if states_dict['source_mode'] == "image":
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

            main_frame = cv2.imread(f"{app.config['UPLOAD_FOLDER']}{states_dict['source_image']}")
            ret2, frame_background = cap2.read()

        classes_index = []
        start_moment = time.time()  # Timer for FPS calculation

        # Draw frame with render modes and settings
        if main_frame is not None:
            if not states_dict['view_source']:
                main_frame, frame_boost_sequence, frame_boost_list, classes_index, zipped_images, zip_obj, zip_is_opened = \
                    render_with_mode(rendering_mode, settings_ajax, main_frame, frame_background, f, f1, yolo_network,
                                     rcnn_network, caffe_network, superres_network, dain_network, esrgan_network,
                                     device, output_layers, classes_index, zip_obj, zip_is_opened, zipped_images,
                                     states_dict, started_rendering_video)

            with lock:
                check_if_user_is_connected(timer_start, 7)  # Terminate process if browser tab was closed
                states_dict['frame_processed'] += 1
                elapsed_time = time.time()
                fps = 1 / (elapsed_time - start_moment)
                # Resize frame for HTML preview with correct aspect ratio
                x_coeff = 460 / main_frame.shape[0]
                x_size = round(x_coeff * main_frame.shape[1])
                resized = cv2.resize(main_frame, (x_size, 460))

                if rendering_mode == 'extract_objects_yolo_mode' and not states_dict['view_source']:
                    resized = draw_yolo_stats(resized, classes_index, font)

                if states_dict['source_mode'] == "image":
                    cv2.imwrite(
                        f"static/user_renders/output{args['port']}{states_dict['source_image']}",
                        main_frame,
                    )

                if (
                        states_dict['source_mode'] == "image"
                        and rendering_mode == 'extract_and_replace_background'
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
                        states_dict['source_mode'] in ("video", "youtube", "ipcam")
                        and writer is not None
                        and started_rendering_video
                ):
                    if rendering_mode == 'boost_fps_dain' and started_rendering_video:
                        frame_boost_sequence, frame_boost_list = zip(
                            *sorted(zip(frame_boost_sequence, frame_boost_list)))
                        frame_edge = frame_boost_list[len(frame_boost_list) - 1]

                        for i in range(len(frame_boost_list) - 1):
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
                if states_dict['source_mode'] in ("video", "youtube"):
                    if states_dict['total_frames'] != 0:
                        progress = (
                                states_dict['frame_processed']
                                / states_dict['total_frames']
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
                    if states_dict['source_mode'] == "youtube":
                        out_file = states_dict['output_file_page']
                    if states_dict['source_mode'] in ("video", "image"):
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
                if states_dict['frame_processed'] == 1:
                    print("started")

                # Take screenshot if needed
                if states_dict['need_to_create_screenshot']:
                    states_dict['need_to_create_screenshot'] = False

                    print("Taking screenshot...")
                    cv2.imwrite(
                        f"static/user_renders/output{args['port']}Screenshot.png", main_frame
                    )
                    time.sleep(0.5)
                    states_dict['screenshot_path'] = (
                        f"static/user_renders/output{args['port']}Screenshot.png"
                    )
                    states_dict['screenshot_ready'] = True

                if states_dict['source_mode'] == "image":
                    started_rendering_video = False
        # ... otherwise stop rendering
        else:
            zip_obj.close()
            check_if_user_is_connected(timer_start, 7)
            started_rendering_video = False
            if writer:
                writer.release()
            position_value = 1
