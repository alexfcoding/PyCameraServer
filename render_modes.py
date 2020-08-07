import numpy as np
import numpy
from random import randint
from sklearn.cluster import MiniBatchKMeans
import ESRGAN.architecture as arch
import time
import os
from torch.autograd import Variable
import torch
import cv2
import random
import DAIN.networks

classes = []

with open("models/yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors_yolo = np.random.uniform(0, 255, size=(len(classes), 3))
object_index = 0


def initialize_yolo_network(classes, use_cuda):
    yolo_network = cv2.dnn.readNet(
        "models/yolo/yolov3.weights", "models/yolo/yolov3.cfg"
    )

    if use_cuda:
        yolo_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        yolo_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layers_names = yolo_network.getLayerNames()
    output_layers = [
        layers_names[i[0] - 1] for i in yolo_network.getUnconnectedOutLayers()
    ]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    return yolo_network, layers_names, output_layers, colors


def initialize_rcnn_network(use_cuda):
    weights_path = "models/mask-rcnn/frozen_inference_graph.pb"
    config_path = "models/mask-rcnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    rcnn_network = cv2.dnn.readNetFromTensorflow(weights_path, config_path)

    if use_cuda:
        rcnn_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        rcnn_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    return rcnn_network


def initialize_caffe_network():
    net = cv2.dnn.readNetFromCaffe(
        "models/caffe/colorization_deploy_v2.prototxt",
        "models/caffe/colorization_release_v2.caffemodel",
    )
    pts = np.load("models/caffe/pts_in_hull.npy")
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net


def initialize_superres_network(model_type):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    if (model_type == "EDSR"):
        sr.readModel("models/upscalers/EDSR_x4.pb")
        sr.setModel("edsr", 4)
    if (model_type == "LAPSRN"):
        sr.readModel("models/upscalers/LapSRN_x4.pb")
        sr.setModel("lapsrn", 4)
    if (model_type == "FSRCNN"):
        sr.readModel("models/upscalers/FSRCNN_x4.pb")
        sr.setModel("fsrcnn", 4)
    if (model_type == "FSRCNN_SMALL"):
        sr.readModel("models/upscalers/FSRCNN-small_x4.pb")
        sr.setModel("fsrcnn", 4)
    return sr


def initialize_esrgan_network(model_type, use_cuda):
    model_path = ""
    device = torch.device("cpu")

    if (model_type == "FALCOON"):
        model_path = "models/esrgan/falcoon.pth"

    if (model_type == "MANGA"):
        model_path = "models/esrgan/Manga109Attempt.pth"

    if (model_type == "RRDB_ESRGAN"):
        model_path = "models/esrgan/RRDB_ESRGAN_x4_old_arch.pth"

    if (model_type == "RRDB_PSNR"):
        model_path = "models/esrgan/RRDB_PSNR_x4_old_arch.pth"

    if (model_type == "RRDB_INTERP_0.2"):
        model_path = "models/esrgan/interp_02.pth"

    if (model_type == "RRDB_INTERP_0.4"):
        model_path = "models/esrgan/interp_04.pth"

    if (model_type == "RRDB_INTERP_0.6"):
        model_path = "models/esrgan/interp_06.pth"

    if (model_type == "RRDB_INTERP_0.8"):
        model_path = "models/esrgan/interp_08.pth"

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device('cpu')

    test_img_folder = "LR/*"

    model = arch.RRDB_Net(
        3,
        3,
        64,
        23,
        gc=32,
        upscale=4,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        res_scale=1,
        upsample_mode="upconv",
    )
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    return model, device


def initialize_dain_network(use_cuda):
    torch.backends.cudnn.benchmark = True
    model = DAIN.networks.__dict__['DAIN'](channel=3,
                                            filter_size=4,
                                            timestep=0.5,
                                            training=False)

    if use_cuda:
        model = model.cuda()

    SAVED_MODEL = 'DAIN/model_weights/best.pth'
    if os.path.exists(SAVED_MODEL):
        print("The testing model weight is: " + SAVED_MODEL)
        if not use_cuda:
            pretrained_dict = torch.load(SAVED_MODEL, map_location=lambda storage, loc: storage)
            # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
        else:
            pretrained_dict = torch.load(SAVED_MODEL)
            # model.load_state_dict(torch.load(args.SAVED_MODEL))

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # 4. release the pretrained dict for saving memory
        pretrained_dict = []
    else:
        print("*****************************************************************")
        print("**** We don't load any trained weights **************************")
        print("*****************************************************************")

    model = model.eval()  # deploy mode

    unique_id = str(random.randint(0, 100000))
    print("The unique id for current testing is: " + str(unique_id))

    return model


def find_yolo_classes(input_frame, yolo_network, output_layers, confidence_value):
    classes_out = []
    height, width, channels = input_frame.shape
    blob = cv2.dnn.blobFromImage(
        input_frame, 0.003, (608, 608), (0, 0, 0), True, crop=False
    )
    yolo_network.setInput(blob)
    outs = yolo_network.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    confidence_value = confidence_value / 100

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_value:
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.2)

    for i in range(len(boxes)):
        if i in indexes:
            classes_out.append(class_ids[i])

    return boxes, indexes, class_ids, confidences, classes_out


def find_rcnn_classes(input_frame, rcnn_network):
    labels_path = "models/mask-rcnn/object_detection_classes_coco.txt"
    labels = open(labels_path).read().strip().split("\n")

    np.random.seed(46)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    blob = cv2.dnn.blobFromImage(input_frame, swapRB=True, crop=False)
    rcnn_network.setInput(blob)
    (boxes, masks) = rcnn_network.forward(["detection_out_final", "detection_masks"])

    return boxes, masks, labels, colors


def extract_objects_yolo(
    input_frame,
    boxes,
    indexes,
    class_ids,
    confidences,
    zip_archive,
    zip_is_opened,
    zipped_images,
    source_mode,
    started_rendering_mode,
):
    global object_index

    frame_copy = input_frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = colors_yolo[class_ids[i]]

            if x < 0:
                x = 0
            if y < 0:
                y = 0

            blk = np.zeros(input_frame.shape, np.uint8)

            crop_img = frame_copy[y : y + h, x : x + w]

            cv2.rectangle(input_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            if label == "person":
                cv2.putText(
                    input_frame,
                    label + "[" + str(np.round(confidences[i], 2)) + "]",
                    (x, y - 5),
                    font,
                    0.7,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                input_frame = cv2.addWeighted(input_frame, 1, blk, 0.2, 0)

            if label == "car":
                cv2.putText(
                    input_frame,
                    label + "[" + str(np.round(confidences[i], 2)) + "]",
                    (x, y - 5),
                    font,
                    0.7,
                    (213, 160, 47),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.rectangle(blk, (x, y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
                input_frame = cv2.addWeighted(input_frame, 1, blk, 0.2, 0)

            if (label != "car") & (label != "person"):
                cv2.putText(
                    input_frame,
                    label + "[" + str(np.round(confidences[i], 2)) + "]",
                    (x, y - 5),
                    font,
                    0.7,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.rectangle(blk, (x, y), (x + w, y + h), color, cv2.FILLED)
                input_frame = cv2.addWeighted(input_frame, 1, blk, 0.2, 0)

            if (
                started_rendering_mode
                and zip_is_opened
                and source_mode in ("video", "youtube", "ipcam")
            ):
                cv2.imwrite(f"static/user_renders/{label}{str(object_index)}.jpg", crop_img)
                zip_archive.write(f"static/user_renders/{label}{str(object_index)}.jpg")
                os.remove(f"static/user_renders/{label}{str(object_index)}.jpg")

            if (
                started_rendering_mode
                and zip_is_opened
                and source_mode == "image"
                and zipped_images == False
            ):
                cv2.imwrite(f"static/user_renders/{label}{str(object_index)}.jpg", crop_img)
                zip_archive.write(f"static/user_renders/{label}{str(object_index)}.jpg")
                os.remove(f"static/user_renders/{label}{str(object_index)}.jpg")

            object_index += 1

    if (
        started_rendering_mode
        and zip_is_opened
        and source_mode == "image"
        and zipped_images == False
    ):
        zipped_images = True
        zip_archive.close()
        zip_is_opened = False

    return input_frame, zipped_images, zip_archive, zip_is_opened


def objects_to_text_yolo(
    input_frame,
    boxes,
    indexes,
    class_ids,
    font_size,
    ascii_distance_value,
    blur_value,
    ascii_thickness_value,
):
    global object_index

    font_size /= 10

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = colors_yolo[class_ids[i]]

            if x < 0:
                x = 0
            if y < 0:
                y = 0

            crop_img = input_frame[y : y + h, x : x + w]
            crop_img = cv2.GaussianBlur(crop_img, (blur_value, blur_value), blur_value)
            render_str = "abcdefghijklmnopqrstuvwxyz0123456789"

            if (x >= 0) & (y >= 0):
                for xx in range(0, crop_img.shape[1], ascii_distance_value):
                    for yy in range(0, crop_img.shape[0], ascii_distance_value):
                        # char = randint(0, 1)
                        pixel_b, pixel_g, pixel_r = crop_img[yy, xx]
                        char = render_str[randint(0, len(render_str)) - 1]
                        cv2.putText(
                            crop_img,
                            str(char),
                            (xx, yy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size,
                            (int(pixel_b), int(pixel_g), int(pixel_r)),
                            ascii_thickness_value, lineType=cv2.LINE_AA
                        )
            blk = np.zeros(input_frame.shape, np.uint8)

            cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
            input_frame[y : y + h, x : x + w] = crop_img

            object_index += 1

    return input_frame


def canny_people_on_black_yolo(input_frame, boxes, indexes, class_ids):
    global object_index
    input_frame_copy = input_frame
    input_frame = np.zeros((input_frame.shape[0], input_frame.shape[1], 3), np.uint8)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = colors_yolo[class_ids[i]]

            if x < 0:
                x = 0
            if y < 0:
                y = 0

            crop_img = input_frame_copy[y : y + h, x : x + w]

            cv2.imshow("df", crop_img)
            crop_img = cv2.GaussianBlur(crop_img, (5, 5), 5)
            crop_img = auto_canny(crop_img)

            blank_image = np.zeros((crop_img.shape[0], crop_img.shape[1], 3), np.uint8)

            blk = np.zeros(input_frame.shape, np.uint8)

            blk2 = np.zeros(input_frame.shape, np.uint8)

            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
            mask = np.zeros_like(crop_img)
            rows, cols, _ = mask.shape

            # if label == "person":
            # 	mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(int(cols / 2), int(rows / 2)), angle=0, startAngle=0, endAngle=360, color=(255, 255, 0), thickness=-1)
            # if label == "car":
            # 	mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(int(cols / 2), int(rows / 2)), angle=0, startAngle=0, endAngle=360, color=(255, 0, 255), thickness=-1)
            # if label == "truck":
            # 	mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(int(cols / 2), int(rows / 2)), angle=0, startAngle=0, endAngle=360, color=(255, 0, 255), thickness=-1)
            # if label == "bus":
            # 	mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(int(cols / 2), int(rows / 2)), angle=0, startAngle=0, endAngle=360, color=(255, 0, 255), thickness=-1)
            # if label == "bicycle":
            # 	mask = cv2.ellipse(mask, center=(int(cols / 2), int(rows / 2)), axes=(int(cols / 2), int(rows / 2)), angle=0, startAngle=0, endAngle=360, color=(0, 0, 255), thickness=-1)

            mask = cv2.ellipse(
                mask,
                center=(int(cols / 2), int(rows / 2)),
                axes=(int(cols / 2), int(rows / 2)),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255, 255, 255),
                thickness=-1,
            )
            result = np.bitwise_and(crop_img, mask)

            result = adjust_gamma(result, gamma=0.3)

            mult = w * h / 15000

            blk2[y : y + h, x : x + w] = result

            # if (mult<1):
            # 	blk2[blk2 != 0] = 255 * mult

            if label == "person":
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                input_frame = cv2.ellipse(
                    input_frame,
                    center=(x + int(w / 2), y + int(h / 2)),
                    axes=(int(w / 2), int(h / 2)),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=(0, 0, 0),
                    thickness=-1,
                )
                input_frame = cv2.addWeighted(input_frame, 1, blk2, 1, 0)

                circleSize = int(w * h / 7000)
                cv2.circle(
                    input_frame,
                    (x + int(w / 2), y - int(h / 5)),
                    2,
                    (0, 0, 255),
                    circleSize,
                )

            if label == "car":
                input_frame = cv2.ellipse(
                    input_frame,
                    center=(x + int(w / 2), y + int(h / 2)),
                    axes=(int(w / 2), int(h / 2)),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=(0, 0, 0),
                    thickness=-1,
                )
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                input_frame = cv2.addWeighted(input_frame, 1, blk2, 1, 0)
                # bufferFrames[streamIndex] = cv2.addWeighted(bufferFrames[streamIndex], 1, blk2, 1, 1)
                circleSize = int(w * h / 7000)
                cv2.circle(
                    input_frame,
                    (x + int(w / 2), y - int(h / 5)),
                    2,
                    (0, 0, 255),
                    circleSize,
                )

            if label == "truck":
                input_frame = cv2.ellipse(
                    input_frame,
                    center=(x + int(w / 2), y + int(h / 2)),
                    axes=(int(w / 2), int(h / 2)),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=(0, 0, 0),
                    thickness=-1,
                )
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                input_frame = cv2.addWeighted(input_frame, 1, blk2, 1, 0)
                # bufferFrames[streamIndex] = cv2.addWeighted(bufferFrames[streamIndex], 1, blk2, 1, 1)
                circleSize = int(w * h / 7000)
                cv2.circle(
                    input_frame,
                    (x + int(w / 2), y - int(h / 5)),
                    2,
                    (0, 0, 255),
                    circleSize,
                )

            if label == "bus":
                input_frame = cv2.ellipse(
                    input_frame,
                    center=(x + int(w / 2), y + int(h / 2)),
                    axes=(int(w / 2), int(h / 2)),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=(0, 0, 0),
                    thickness=-1,
                )
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                input_frame = cv2.addWeighted(input_frame, 1, blk2, 1, 0)
                # bufferFrames[streamIndex] = cv2.addWeighted(bufferFrames[streamIndex], 1, blk2, 1, 1)
                circleSize = int(w * h / 7000)
                cv2.circle(
                    input_frame,
                    (x + int(w / 2), y - int(h / 5)),
                    2,
                    (0, 0, 255),
                    circleSize,
                )
            if label == "bicycle":
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                input_frame = cv2.ellipse(
                    input_frame,
                    center=(x + int(w / 2), y + int(h / 2)),
                    axes=(int(w / 2), int(h / 2)),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=(0, 0, 0),
                    thickness=-1,
                )
                input_frame = cv2.addWeighted(input_frame, 1, blk2, 1, 0)
                circleSize = int(w * h / 7000)
                cv2.circle(
                    input_frame,
                    (x + int(w / 2), y - int(h / 5)),
                    2,
                    (0, 0, 255),
                    circleSize,
                )

            if (
                label != "person"
                and label != "car"
                and label != "truck"
                and label != "bus"
            ):
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                input_frame = cv2.ellipse(
                    input_frame,
                    center=(x + int(w / 2), y + int(h / 2)),
                    axes=(int(w / 2), int(h / 2)),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=(0, 0, 0),
                    thickness=-1,
                )
                input_frame = cv2.addWeighted(input_frame, 1, blk2, 1, 0)
                circleSize = int(w * h / 7000)
                cv2.circle(
                    input_frame,
                    (x + int(w / 2), y - int(h / 5)),
                    2,
                    (0, 0, 255),
                    circleSize,
                )
            # if (blurPeople == False):
            #     cv2.rectangle(
            #         bufferFrames[streamIndex], (x, y), (x + w, y + h), (255, 255, 255), 2)

            object_index += 1

    return input_frame


def canny_people_on_background_yolo(input_frame, boxes, indexes, class_ids):
    global object_index

    # input_frame_copy = input_frame
    # input_frame = np.zeros(
    # 	(input_frame.shape[0], input_frame.shape[1], 3), np.uint8)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = colors_yolo[class_ids[i]]

            if x < 0:
                x = 0
            if y < 0:
                y = 0

            crop_img = input_frame[y : y + h, x : x + w]

            # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("df", crop_img)
            crop_img = cv2.GaussianBlur(crop_img, (5, 5), 5)
            crop_img = auto_canny(crop_img)
            # crop_img = cv2.Canny(crop_img, 100, 200)
            blank_image = np.zeros((crop_img.shape[0], crop_img.shape[1], 3), np.uint8)

            myStr = "abcdefghijklmnopqrstuvwxyz0123456789"

            blk = np.zeros(input_frame.shape, np.uint8)

            blk2 = np.zeros(input_frame.shape, np.uint8)

            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)

            mask = np.zeros_like(crop_img)
            rows, cols, _ = mask.shape
            mask = cv2.ellipse(
                mask,
                center=(int(cols / 2), int(rows / 2)),
                axes=(int(cols / 2), int(rows / 2)),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255, 0, 255),
                thickness=-1,
            )
            result = np.bitwise_and(crop_img, mask)
            # result = adjust_gamma(result, gamma=0.3)

            mult = w * h / 20000

            if mult < 1:
                result[result != 0] = 255 * mult

            blk2[y : y + h, x : x + w] = result

            if label == "person":
                # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
                # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                input_frame = cv2.addWeighted(input_frame, 1, blk2, 1, 0)
                circleSize = int(w * h / 7000)
                cv2.circle(
                    input_frame,
                    (x + int(w / 2), y - int(h / 5)),
                    2,
                    (0, 0, 255),
                    circleSize,
                )

            # if label == "car":
            #     # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
            #     # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
            #     input_frame = cv2.addWeighted(input_frame, 1, blk2, 1, 0)
            #     circleSize = int(w * h / 7000)
            #     cv2.circle(input_frame, (x + int(w / 2), y - int(h / 5)), 2, (0, 0, 255), circleSize)
            #
            # if label == "bicycle":
            #     # cv2.putText(bufferFrames[streamIndex], label + "[" + str(np.round(confidences[i], 2)) + "]", (x, y - 5), font, 0.7, (0,255,0), 2, lineType = cv2.LINE_AA)
            #     # cv2.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
            #     input_frame = cv2.addWeighted(input_frame, 1, blk2, 1, 0)
            #     circleSize = int(w * h / 7000)
            #     cv2.circle(input_frame, (x + int(w / 2), y - int(h / 5)), 2, (0, 0, 255), circleSize)
            else:
                input_frame = cv2.addWeighted(input_frame, 1, blk2, 1, 0)
                circleSize = int(w * h / 7000)
                cv2.circle(
                    input_frame,
                    (x + int(w / 2), y - int(h / 5)),
                    2,
                    (0, 255, 0),
                    circleSize,
                )
            object_index += 1

    return input_frame


def extract_and_cut_background_rcnn(
    input_frame, boxes, masks, labels, confidence_value
):
    confidence_value /= 100
    classes_out = []
    frame_canny = auto_canny(input_frame)
    frame_canny = cv2.cvtColor(frame_canny, cv2.COLOR_GRAY2RGB)
    input_frame = np.zeros(input_frame.shape, np.uint8)
    frame_canny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        class_id = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confidence_value:
            classes_out.append(class_id)

            (H, W) = input_frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            box_w = end_x - start_x
            box_h = end_y - start_y

            mask = masks[i, class_id]
            mask = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_CUBIC)
            mask = mask > 0.1

            # if (labels[class_id] == "person"):
            frm = frame_canny[start_y:end_y, start_x:end_x][mask]
            frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)
            input_frame[start_y:end_y, start_x:end_x][mask] = frm
            # else:
            # frm = frame_canny[start_y:end_y, start_x:end_x][mask]
            # frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)
            # input_frame[start_y:end_y, start_x:end_x][mask] = frm

    frame_out = input_frame

    return frame_out


def extract_and_replace_background_rcnn(
    input_frame, frame_background, boxes, masks, labels, colors, confidence_value, canny_thres1, canny_thres2
):
    confidence_value /= 100
    classes_out = []
    frame_copy = input_frame

    frame_background = cv2.resize(
        frame_background, (input_frame.shape[1], input_frame.shape[0])
    )
    frame_copy = cv2.resize(frame_copy, (1280, 720))

    input_frame = cv2.GaussianBlur(input_frame, (5, 5), 5)
    frame_canny = cv2.Canny(input_frame, canny_thres1, canny_thres2)
    # frame_canny = auto_canny(input_frame)

    frame_canny = cv2.cvtColor(frame_canny, cv2.COLOR_GRAY2RGB)
    frame_out = np.zeros(input_frame.shape, np.uint8)
    input_frame = np.zeros(input_frame.shape, np.uint8)
    frame_canny *= np.array((1, 0, 1), np.uint8)

    for i in range(0, boxes.shape[2]):
        class_id = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confidence_value:
            classes_out.append(class_id)

            (H, W) = input_frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            box_w = end_x - start_x
            box_h = end_y - start_y

            mask = masks[i, class_id]
            mask = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_CUBIC)
            mask = mask > 0.1

            color = colors[class_id]

            # if (labels[class_id] == "person"):
            frm = frame_canny[start_y:end_y, start_x:end_x][mask]
            frm[np.all(frm == (255, 0, 255), axis=-1)] = (255, 255, 0)
            input_frame[start_y:end_y, start_x:end_x][mask] = frm
            # if (labels[class_id] == "car"):
            # frm = frame_canny[start_y:end_y, start_x:end_x][mask]
            # frm[np.all(frm == (255, 0, 255), axis=-1)] = (255, 0, 255)
            # input_frame[start_y:end_y, start_x:end_x][mask] = frm
            # else:
            #     frm = frame_canny[start_y:end_y, start_x:end_x][mask]
            #     frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)
            #     input_frame[start_y:end_y, start_x:end_x][mask] = frm

    # text = "{}[{:.2f}]".format(LABELS[class_id], confidence)
    # font_size = (np.sqrt(box_w * box_h) / 200)
    # cv2.putText(frame_canny, text, (start_x, start_y - 50),
    # cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 2)

    frame_out = cv2.addWeighted(input_frame, 1, frame_background, 1, 0)
    return frame_out


def color_canny_rcnn(
    input_frame, boxes, masks, labels, confidence_value, rcnn_blur_value, canny_blur, canny_thres1, canny_thres2
):
    confidence_value /= 100
    classes_out = []
    # frame_canny = auto_canny(input_frame)
    input_frame = cv2.GaussianBlur(input_frame, (canny_blur, canny_blur), canny_blur)

    frame_canny = cv2.Canny(input_frame, canny_thres1, canny_thres2)
    # frame_canny = auto_canny(input_frame, 0)
    frame_canny = cv2.cvtColor(frame_canny, cv2.COLOR_GRAY2BGR)
    frame_out = np.zeros(input_frame.shape, np.uint8)

    input_frame = np.zeros(input_frame.shape, np.uint8)
    frame_canny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        class_id = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confidence_value:
            classes_out.append(class_id)

            (H, W) = input_frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            box_w = end_x - start_x
            box_h = end_y - start_y

            mask = masks[i, class_id]
            mask = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_CUBIC)
            mask = mask > 0.1

            # if (labels[class_id] == "person"):
            frm = frame_canny[start_y:end_y, start_x:end_x][mask]
            frm[np.all(frm == (255, 255, 0), axis=-1)] = (255, 0, 255)
            input_frame[start_y:end_y, start_x:end_x][mask] = frm
            # if (labels[class_id] == "car"):
            #     frm = frame_canny[start_y:end_y, start_x:end_x][mask]
            #     frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 0)
            #     input_frame[start_y:end_y, start_x:end_x][mask] = frm
            # if (labels[class_id] == "truck"):
            #     frm = frame_canny[start_y:end_y, start_x:end_x][mask]
            #     frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 0)
            #     input_frame[start_y:end_y, start_x:end_x][mask] = frm
            # if (labels[class_id] == "bus"):
            #     frm = frame_canny[start_y:end_y, start_x:end_x][mask]
            #     frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 0)
            #     input_frame[start_y:end_y, start_x:end_x][mask] = frm

    # frame_out = cv2.addWeighted(input_frame, 1, frame_canny, 1, 0)
    frame_canny = cv2.GaussianBlur(
        frame_canny, (rcnn_blur_value, rcnn_blur_value), rcnn_blur_value
    )

    for i in range(0, boxes.shape[2]):
        class_id = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > 0.5:
            classes_out.append(class_id)

            (H, W) = input_frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            box_w = end_x - start_x
            box_h = end_y - start_y

            if labels[class_id] == "person":
                text = "{}[{:.2f}]".format(labels[class_id], confidence)
                font_size = np.sqrt(box_w * box_h) / 200
                cv2.putText(
                    frame_canny,
                    text,
                    (start_x, start_y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (0, 255, 255),
                    2,
                )
            if labels[class_id] == "car":
                text = "{}[{:.2f}]".format(labels[class_id], confidence)
                font_size = np.sqrt(box_w * box_h) / 200
                cv2.putText(
                    frame_canny,
                    text,
                    (start_x, start_y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 0, 255),
                    2,
                )
            if labels[class_id] == "truck":
                text = "{}[{:.2f}]".format(labels[class_id], confidence)
                font_size = np.sqrt(box_w * box_h) / 200
                cv2.putText(
                    frame_canny,
                    text,
                    (start_x, start_y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 0, 255),
                    2,
                )
            if labels[class_id] == "bus":
                text = "{}[{:.2f}]".format(labels[class_id], confidence)
                font_size = np.sqrt(box_w * box_h) / 200
                cv2.putText(
                    frame_canny,
                    text,
                    (start_x, start_y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 0, 255),
                    2,
                )

    frame_out = np.bitwise_or(input_frame, frame_canny)
    return frame_out


def color_canny_on_color_background_rcnn(
    input_frame, boxes, masks, labels, confidence_value
):
    confidence_value /= 100
    classes_out = []
    frame_canny = auto_canny(input_frame)
    frame_canny = cv2.cvtColor(frame_canny, cv2.COLOR_GRAY2RGB)
    frame_copy = input_frame
    frame_out = np.zeros(input_frame.shape, np.uint8)
    frame_canny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        class_id = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confidence_value:
            classes_out.append(class_id)

            (H, W) = input_frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            box_w = end_x - start_x
            box_h = end_y - start_y

            mask = masks[i, class_id]
            mask = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_CUBIC)
            mask = mask > 0.5

            if labels[class_id] == "person":
                frm = frame_canny[start_y:end_y, start_x:end_x][mask]
                frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)
                input_frame[start_y:end_y, start_x:end_x][mask] = frm
            else:
                frm = frame_canny[start_y:end_y, start_x:end_x][mask]
                frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 0)
                input_frame[start_y:end_y, start_x:end_x][mask] = frm

    # frame_out = cv2.addWeighted(input_frame, 1, frame_canny, 1, 0)
    # frame_out = np.bitwise_xor(input_frame, frame_canny)
    frame_out = input_frame
    return frame_out


def colorizer_people_rcnn(input_frame, boxes, masks, confidence_value, rcnn_size_value):
    confidence_value /= 100
    classes_out = []
    need_gray_to_bgr = True
    already_bgr = False
    frame_copy = input_frame

    # hsvImg = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
    # hsvImg[..., 1] = hsvImg[..., 1] * 1.1
    # # hsvImg[...,2] = hsvImg[...,2]*0.6
    # frame_copy = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    # input_frame = cv2.GaussianBlur(input_frame, (19, 19), 19)
    frame_canny = auto_canny(input_frame)
    frame_canny = cv2.cvtColor(frame_canny, cv2.COLOR_GRAY2RGB)
    frame_canny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        class_id = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confidence_value:
            classes_out.append(class_id)

            (H, W) = input_frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            box_w = end_x - start_x
            box_h = end_y - start_y

            if rcnn_size_value == 0:
                rcnn_size_value = 2

            smaller_x = int(box_w / rcnn_size_value)
            smaller_y = int(box_h / rcnn_size_value)

            if smaller_x % 2 != 0:
                smaller_x += 1
            if smaller_y % 2 != 0:
                smaller_y += 1

            if box_w > smaller_x:
                box_w -= smaller_x
            else:
                smaller_x = 0

            if box_h > smaller_y:
                box_h -= smaller_y
            else:
                smaller_y = 0

            mask = masks[i, class_id]
            mask = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_CUBIC)
            mask = mask > 0.2

            frm = frame_copy[
                start_y + int(smaller_y / 2) : end_y - int(smaller_y / 2),
                start_x + int(smaller_x / 2) : end_x - int(smaller_x / 2),
            ][mask]
            frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)

            if already_bgr == False:
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2BGR)
                already_bgr = True

            input_frame[
                start_y + int(smaller_y / 2) : end_y - int(smaller_y / 2),
                start_x + int(smaller_x / 2) : end_x - int(smaller_x / 2),
            ][mask] = frm

            need_gray_to_bgr = False

    # text = "{}[{:.2f}]".format(LABELS[class_id], confidence)
    # cv2.putText(frame_canny, text, (start_x, start_y - 50),
    # cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 2)

    if need_gray_to_bgr:
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2RGB)

    frame_out = input_frame
    return frame_out


def colorizer_people_with_blur_rcnn(input_frame, boxes, masks, confidence_value):
    confidence_value /= 100
    classes_out = []
    need_gray_to_bgr = True
    already_bgr = False
    frame_copy = input_frame

    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    input_frame = cv2.GaussianBlur(input_frame, (17, 17), 17)
    frame_canny = auto_canny(input_frame)
    frame_canny = cv2.cvtColor(frame_canny, cv2.COLOR_GRAY2RGB)

    frame_canny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        class_id = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confidence_value:
            classes_out.append(class_id)

            (H, W) = input_frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            box_w = end_x - start_x
            box_h = end_y - start_y

            smaller_x = int(box_w / 10)
            smaller_y = int(box_h / 10)
            # smaller_x = 0
            # smaller_y = 0

            if smaller_x % 2 != 0:
                smaller_x += 1
            if smaller_y % 2 != 0:
                smaller_y += 1

            if box_w > smaller_x:
                box_w -= smaller_x
            else:
                smaller_x = 0

            if box_h > smaller_y:
                box_h -= smaller_y
            else:
                smaller_y = 0

            mask = masks[i, class_id]
            mask = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_CUBIC)
            mask = mask > 0.1

            frm = frame_copy[
                start_y + int(smaller_y / 2) : end_y - int(smaller_y / 2),
                start_x + int(smaller_x / 2) : end_x - int(smaller_x / 2),
            ][mask]
            frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)

            if already_bgr == False:
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2BGR)
                already_bgr = True

            input_frame[
                start_y + int(smaller_y / 2) : end_y - int(smaller_y / 2),
                start_x + int(smaller_x / 2) : end_x - int(smaller_x / 2),
            ][mask] = frm

            need_gray_to_bgr = False

    # text = "{}[{:.2f}]".format(LABELS[class_id], confidence)
    # cv2.putText(frame_canny, text, (start_x, start_y - 50),
    # cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 2)

    if need_gray_to_bgr:
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2RGB)

    frame_out = input_frame
    return frame_out


def people_with_blur_rcnn(
    input_frame,
    boxes,
    masks,
    labels,
    confidence_value,
    rcnn_size_value,
    rcnn_blur_value,
):
    confidence_value /= 100

    classes_out = []
    need_gray_to_bgr = True
    already_bgr = False
    frame_copy = input_frame.copy()
    # input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    input_frame = cv2.GaussianBlur(
        input_frame, (rcnn_blur_value, rcnn_blur_value), rcnn_blur_value
    )
    frame_canny = auto_canny(input_frame)
    frame_canny = cv2.cvtColor(frame_canny, cv2.COLOR_GRAY2RGB)

    frame_canny *= np.array((1, 1, 0), np.uint8)

    for i in range(0, boxes.shape[2]):
        class_id = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confidence_value:
            classes_out.append(class_id)

            (H, W) = input_frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            box_w = end_x - start_x
            box_h = end_y - start_y

            if rcnn_size_value == 0:
                rcnn_size_value = 2

            smaller_x = int(box_w / rcnn_size_value)
            smaller_y = int(box_h / rcnn_size_value)
            # smaller_x = 0
            # smaller_y = 0

            if smaller_x % 2 != 0:
                smaller_x += 1
            if smaller_y % 2 != 0:
                smaller_y += 1

            if box_w > smaller_x:
                box_w -= smaller_x
            else:
                smaller_x = 0

            if box_h > smaller_y:
                box_h -= smaller_y
            else:
                smaller_y = 0

            mask = masks[i, class_id].copy()
            mask = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_CUBIC)
            mask = mask > 0.1

            # if (labels[class_id] == "person"):
            frm = frame_copy[
                start_y + int(smaller_y / 2) : end_y - int(smaller_y / 2),
                start_x + int(smaller_x / 2) : end_x - int(smaller_x / 2),
            ][mask].copy()
            frm[np.all(frm == (255, 255, 0), axis=-1)] = (0, 255, 255)
            input_frame[
                start_y + int(smaller_y / 2) : end_y - int(smaller_y / 2),
                start_x + int(smaller_x / 2) : end_x - int(smaller_x / 2),
            ][mask] = frm.copy()

            # if (already_bgr == False):
            #     input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2BGR)
            #     already_bgr = True

            need_gray_to_bgr = False

    # text = "{}[{:.2f}]".format(LABELS[class_id], confidence)
    # cv2.putText(frame_canny, text, (start_x, start_y - 50),
    # cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 2)

    # if need_gray_to_bgr:
    #     input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2RGB)

    frame_out = input_frame
    return frame_out


def colorizer_caffe(net, image):
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]

    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # cv2.imshow("Original", image)
    # cv2.imshow("Colorized", colorized)
    # cv2.waitKey(0)
    return colorized


def upscale_with_superres(network, image):
    # result = cv2.resize(image, (round(image.shape[1]*resize_value), round(image.shape[0]*resize_value)))
    result = network.upsample(image)
    return result


def upscale_with_esrgan(network, device, image):
    image = image * 1.0 / 255
    image = torch.from_numpy(np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = image.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = network(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    # cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

    return output


def boost_fps_with_dain(dain_network, f, f1, use_cuda):
    save_which = 1
    dtype = torch.cuda.FloatTensor

    fpsBoost = 7
    frame_sequence = None
    frame_write_sequence = None

    if (fpsBoost == 7):
        frame_sequence = [0, 1, 2, 0, 2, 3, 0, 3, 4, 3, 2, 5, 2, 1, 6, 2, 6, 7, 6, 1, 8]
        frame_write_sequence = [0, 8, 4, 2, 1, 3, 6, 5, 7]

    if (fpsBoost == 3):
        frame_sequence = [0, 1, 2, 0, 2, 3, 2, 1, 4]
        frame_write_sequence = [0, 4, 2, 1, 3]

    if (fpsBoost == 2):
        frame_sequence = [0, 1, 2]
        frame_write_sequence = [0, 2, 1]

    frames = []
    frame_index = 0

    frames.append(f)
    frames.append(f1)

    for i in range(fpsBoost - 1):
        X0 = torch.from_numpy(
            np.transpose(frames[frame_sequence[frame_index]], (2, 0, 1)).astype("float32") / 255.0).type(dtype)
        frame_index += 1
        X1 = torch.from_numpy(
            np.transpose(frames[frame_sequence[frame_index]], (2, 0, 1)).astype("float32") / 255.0).type(dtype)
        frame_index += 1
        y_ = torch.FloatTensor()

        assert (X0.size(1) == X1.size(1))
        assert (X0.size(2) == X1.size(2))

        intWidth = X0.size(2)
        intHeight = X0.size(1)
        channel = X0.size(0)

        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft = int((intWidth_pad - intWidth) / 2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 32
            intPaddingRight = 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

        torch.set_grad_enabled(False)
        X0 = Variable(torch.unsqueeze(X0, 0))
        X1 = Variable(torch.unsqueeze(X1, 0))
        X0 = pader(X0)
        X1 = pader(X1)

        if use_cuda:
            X0 = X0.cuda()
            X1 = X1.cuda()

        proc_end = time.time()
        y_s, offset, filter = dain_network(torch.stack((X0, X1), dim=0))
        y_ = y_s[save_which]

        end = time.time()

        print("*****************current image process time \t " + str(
            time.time() - proc_end) + "s ******************")
        if use_cuda:
            X0 = X0.data.cpu().numpy()
            y_ = y_.data.cpu().numpy()
            offset = [offset_i.data.cpu().numpy() for offset_i in offset]
            filter = [filter_i.data.cpu().numpy() for filter_i in filter] if filter[0] is not None else None
            X1 = X1.data.cpu().numpy()
        else:
            X0 = X0.data.numpy()
            y_ = y_.data.numpy()
            offset = [offset_i.data.numpy() for offset_i in offset]
            filter = [filter_i.data.numpy() for filter_i in filter]
            X1 = X1.data.numpy()

        X0 = np.transpose(255.0 * X0.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                  intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))
        y_ = np.transpose(255.0 * y_.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                  intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))
        offset = [np.transpose(
            offset_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
            (1, 2, 0)) for offset_i in offset]
        filter = [np.transpose(
            filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
            (1, 2, 0)) for filter_i in filter] if filter is not None else None
        X1 = np.transpose(255.0 * X1.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                  intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))

        # imageio.imwrite(arguments_strOut, np.round(y_).astype(numpy.uint8))
        f2 = np.round(y_).astype(numpy.uint8)

        frames.append(f2)

        frame_index += 1

    frame_index = 0

    return frame_write_sequence, frames


def ascii_paint_zoom(
    input_frame, font_size, ascii_distance_value, ascii_thickness_value, blur_value
):
    # font_size /= 10
    if ascii_distance_value < 20:
        ascii_thickness_value = 1

    if ascii_distance_value >= 20 and ascii_distance_value < 40:
        ascii_thickness_value = 2

    if ascii_distance_value >= 40:
        ascii_thickness_value = 3

    font_size = ascii_distance_value / font_size / 4
    input_frame = cv2.GaussianBlur(input_frame, (blur_value, blur_value), blur_value)

    blk = np.zeros(input_frame.shape, np.uint8)

    render_str = "abcdefghkmnopqstuwxyz"
    if (ascii_distance_value > 3):
        for xx in range(0, input_frame.shape[1], ascii_distance_value):
            for yy in range(0, input_frame.shape[0], ascii_distance_value):
                char = randint(0, 1)
                pixel_b, pixel_g, pixel_r = input_frame[yy, xx]
                avg_brightness = sum([pixel_b, pixel_g, pixel_r]) / 3
                position = int(avg_brightness / 255 * 20)

                char = render_str[randint(0, len(render_str)) - 1]
                char = render_str[position]
                cv2.putText(
                    blk,
                    str(char),
                    (xx, yy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (int(pixel_b), int(pixel_g), int(pixel_r)),
                    ascii_thickness_value,
                    lineType=cv2.LINE_AA,
                )
    else:
        blk = input_frame


    return blk

def ascii_paint(
    input_frame, font_size, ascii_distance_value, ascii_thickness_value, blur_value
):
    font_size /= 10
    input_frame = cv2.GaussianBlur(input_frame, (blur_value, blur_value), blur_value)

    blk = np.zeros(input_frame.shape, np.uint8)

    render_str = "abcdefghkmnopqstuwxyz"

    for xx in range(0, input_frame.shape[1], ascii_distance_value):
        for yy in range(0, input_frame.shape[0], ascii_distance_value):
            char = randint(0, 1)
            pixel_b, pixel_g, pixel_r = input_frame[yy, xx]
            avg_brightness = sum([pixel_b, pixel_g, pixel_r]) / 3
            position = int(avg_brightness / 255 * 20)

            char = render_str[randint(0, len(render_str)) - 1]
            char = render_str[position]
            cv2.putText(
                blk,
                str(char),
                (xx, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (int(pixel_b), int(pixel_g), int(pixel_r)),
                ascii_thickness_value,
                lineType=cv2.LINE_AA,
            )

    return blk

def sharpening(input_frame, sharpening_value, sharpening_value2):
    kernel_value = sharpening_value2
    kernel_diff = 9 - kernel_value
    kernel_sharpening = np.array([[-1, -1, -1], [-1, kernel_value, -1], [-1, -1, -1]])

    while kernel_diff != 0:
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    kernel_sharpening[j][i] == kernel_value
                else:
                    if kernel_diff > 0:
                        kernel_sharpening[j][i] += 1
                        kernel_diff -= 1
                    if kernel_diff < 0:
                        kernel_sharpening[j][i] -= 1
                        kernel_diff += 1
                    if kernel_diff == 0:
                        break

    input_frame = cv2.filter2D(input_frame, -1, kernel_sharpening)

    input_frame = cv2.detailEnhance(input_frame, sigma_s=sharpening_value, sigma_r=0.15)

    return input_frame


def denoise(input_frame, denoise_value, denoise_value2):
    if denoise_value2 > 0:
        b, g, r = cv2.split(input_frame)  # get b,g,r
        input_frame = cv2.merge([r, g, b])  # switch it to rgb
        dst = cv2.fastNlMeansDenoisingColored(
            input_frame, None, denoise_value2, denoise_value, 7, 15
        )
        b, g, r = cv2.split(dst)
        input_frame = cv2.merge([r, g, b])

    return input_frame


def morph_edge_detection(input_frame):
    morph = input_frame.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
    image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
    channel_height, channel_width, _ = image_channels[0].shape

    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(
            ~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY
        )
        image_channels[i] = np.reshape(
            image_channels[i], newshape=(channel_height, channel_width, 1)
        )

    image_channels = np.concatenate(
        (image_channels[0], image_channels[1], image_channels[2]), axis=2
    )
    image_channels = cv2.cvtColor(image_channels, cv2.COLOR_BGR2GRAY)
    image_channels = cv2.cvtColor(image_channels, cv2.COLOR_GRAY2BGR)
    image_channels = cv2.bitwise_not(image_channels)
    return image_channels


def limit_colors_kmeans(input_frame, color_count):
    if color_count > 0:
        (h, w) = input_frame.shape[:2]
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2BGR)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2LAB)
        input_frame = input_frame.reshape(
            (input_frame.shape[0] * input_frame.shape[1], 3)
        )
        clt = MiniBatchKMeans(n_clusters=color_count)
        labels = clt.fit_predict(input_frame)
        quant = clt.cluster_centers_.astype("uint8")[labels]
        quant = quant.reshape((h, w, 3))
        input_frame = input_frame.reshape((h, w, 3))
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_LAB2BGR)
        input_frame = quant

    return input_frame


def auto_canny(image: object, sigma: object = 0.33) -> object:
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    return cv2.LUT(image, table)


def adjust_saturation(input_frame, saturation=1):
    saturation = saturation / 100
    hsv = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation)
    hsv[:, :, 2] = cv2.multiply(hsv[:, :, -1], 1)
    input_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return input_frame


def adjust_br_contrast(input_frame, contrast_value, brightness_value):
    contrast_value = contrast_value / 100
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    input_frame = cv2.convertScaleAbs(
        input_frame, alpha=contrast_value, beta=brightness_value
    )
    return input_frame
