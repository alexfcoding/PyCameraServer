import cv2

def initNetworkUpscale():
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("EDSR_x3.pb")
    sr.setModel("edsr", 3)
    return sr

def upscaleImage(network, image):
    result = network.upsample(image)
    return result
