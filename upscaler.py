import cv2

def initNetworkUpscale():
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    # sr.readModel("EDSR_x4.pb")
    # sr.setModel("edsr", 4)
    sr.readModel("LapSRN_x4.pb")
    sr.setModel("lapsrn", 4)
    # sr.readModel("FSRCNN_x4.pb")
    # sr.setModel("fsrcnn", 4)
    return sr

def upscaleImage(network, image):   
    #result = cv2.resize(image, (round(image.shape[1]*resizeValue), round(image.shape[0]*resizeValue)))
    result = network.upsample(image)
    return result
