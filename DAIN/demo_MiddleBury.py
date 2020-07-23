import time
import os
from torch.autograd import Variable
import math
import torch
import cv2
import random
import numpy as np
import numpy
import DAIN.networks
from DAIN.my_args import  args

# from scipy.misc import imread, imsave
from DAIN.AverageMeter import  *
from PIL import Image
import scipy
import imageio

frameEdge = None		

torch.backends.cudnn.benchmark = True # to speed up the

DO_MiddleBurryOther = True
MB_Other_DATA = "./MiddleBurySet/other-data/"
MB_Other_RESULT = "./MiddleBurySet/other-result-author/"
MB_Other_GT = "./MiddleBurySet/other-gt-interp/"
if not os.path.exists(MB_Other_RESULT):
    os.mkdir(MB_Other_RESULT)

model = DAIN.networks.__dict__[args.netName](channel=args.channels,
                            filter_size = args.filter_size ,
                            timestep=args.time_step,
                            training=False)

if args.use_cuda:
    model = model.cuda()

args.SAVED_MODEL = './model_weights/best.pth'
if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
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

model = model.eval() # deploy mode

use_cuda=args.use_cuda

save_which=args.save_which
dtype = args.dtype
unique_id =str(random.randint(0, 100000))
print("The unique id for current testing is: " + str(unique_id))

interp_error = AverageMeter()

if DO_MiddleBurryOther:
    subdir = os.listdir(MB_Other_DATA)
    gen_dir = os.path.join(MB_Other_RESULT, unique_id)
    os.mkdir(gen_dir)

    tot_timer = AverageMeter()
    proc_timer = AverageMeter()
    end = time.time()

#  =================================================================================
fpsBoost = 7
frame_sequence = None
frame_write_sequence = None

if (fpsBoost == 7):
    frame_sequence = [0, 1, 2, 0, 2, 3, 0, 3, 4, 3, 2, 5, 2, 1, 6, 2, 6, 7, 6, 1, 8]
    frame_write_sequence = [0, 8, 4, 2, 1, 3, 6, 5, 7]
if (fpsBoost == 3):
    frame_sequence = [0, 1, 2, 0, 2, 3, 2, 1, 4]
    frame_write_sequence = [0, 4, 2, 1, 3]

frames = []
frame_index = 0
frameEdge = None

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
cap = cv2.VideoCapture("monkeygif.gif")

ret, f = cap.read()

bufferFrames = None

writer = cv2.VideoWriter("test.avi",fourcc, 120, (
							f.shape[1], f.shape[0]), True)

iter = 0        

while True:    
   
    if (iter == 0):
        ret, f = cap.read() 
        # ret, f55 = cap.read()
        # ret, f55 = cap.read()  
        ret, f1 = cap.read() 
       
        frames.append(f)
        frames.append(f1)
    else:
        f = frameEdge
        ret, f1 = cap.read() 
        frames.append(f)
        frames.append(f1)
       
    for i in range(fpsBoost):
        X0 =  torch.from_numpy( np.transpose(frames[frame_sequence[frame_index]] , (2,0,1)).astype("float32")/ 255.0).type(dtype)
        frame_index += 1
        X1 =  torch.from_numpy( np.transpose(frames[frame_sequence[frame_index]] , (2,0,1)).astype("float32")/ 255.0).type(dtype)
        frame_index += 1
        y_ = torch.FloatTensor()

        assert (X0.size(1) == X1.size(1))
        assert (X0.size(2) == X1.size(2))

        intWidth = X0.size(2)
        intHeight = X0.size(1)
        channel = X0.size(0)    

        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft =int(( intWidth_pad - intWidth)/2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 32
            intPaddingRight= 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

        torch.set_grad_enabled(False)
        X0 = Variable(torch.unsqueeze(X0,0))
        X1 = Variable(torch.unsqueeze(X1,0))
        X0 = pader(X0)
        X1 = pader(X1)

        if use_cuda:
            X0 = X0.cuda()
            X1 = X1.cuda()
        proc_end = time.time()
        y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
        y_ = y_s[save_which]

        proc_timer.update(time.time() -proc_end)
        tot_timer.update(time.time() - end)
        end  = time.time()

        print("*****************current image process time \t " + str(time.time()-proc_end )+"s ******************" )
        if use_cuda:
            X0 = X0.data.cpu().numpy()
            y_ = y_.data.cpu().numpy()
            offset = [offset_i.data.cpu().numpy() for offset_i in offset]
            filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
            X1 = X1.data.cpu().numpy()
        else:
            X0 = X0.data.numpy()
            y_ = y_.data.numpy()
            offset = [offset_i.data.numpy() for offset_i in offset]
            filter = [filter_i.data.numpy() for filter_i in filter]
            X1 = X1.data.numpy()

        X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
        y_ = np.transpose(255.0 * y_.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
        offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
        filter = [np.transpose(filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0)) for filter_i in filter]  if filter is not None else None
        X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

        #imageio.imwrite(arguments_strOut, np.round(y_).astype(numpy.uint8))
        f2 = np.round(y_).astype(numpy.uint8)
        
        
        frames.append(f2)  
        
       
        

        iter += 1
        frame_index += 1


    # list1, list2 = zip(*sorted(zip(list1, list2)))
    frame_write_sequence, frames = zip(*sorted(zip(frame_write_sequence, frames)))

    # for frame_position in frame_write_sequence:
    #     writer.write(frames[frame_position])
    #     cv2.imshow("video",  frames[frame_position])   
    #     key = cv2.waitKey(1) & 0xFF

    #     if key == ord("q"):
    #         break
    for frame in frames:
        writer.write(frame)
        cv2.imshow("video",  frame)   
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # frameEdge = frames[1] 
    frameEdge = frames[len(frames)-1]     
    frames = list(frames)
    frames.clear()
    frame_write_sequence = list(frame_write_sequence)

    if (fpsBoost == 7):
        frame_sequence = [0, 1, 2, 0, 2, 3, 0, 3, 4, 3, 2, 5, 2, 1, 6, 2, 6, 7, 6, 1, 8]
        frame_write_sequence = [0, 8, 4, 2, 1, 3, 6, 5, 7]
    if (fpsBoost == 3):
        frame_sequence = [0, 1, 2, 0, 2, 3, 2, 1, 4]
        frame_write_sequence = [0, 4, 2, 1, 3]

    frame_index = 0

    
    # ==============================================================================
