from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import os
import sys
import argparse
import cv2
from PIL import Image
import torch
from torch.autograd import Variable
from datetime import date, datetime

# FOR TELLO
from djitellopy import tello
from simple_pid import PID
import keyControl
import pidTello

# DONATELLO V2.0
# Create an tello object
donatello = tello.Tello()
donatello.connect()

# General information
print("Battery level: ", donatello.get_battery(), "%")
print("Temparature: ", donatello.get_temperature(), "~")

# Enables exchange of images
donatello.streamon()

# Global variables
w, h = 1080, 720          # Image size
fbRange = [200, 400]  # Desired Range
pError = 0

# Simple PID for Person + Color
pidForYaw = PID(0.18, 0.00005, 0.01, setpoint=w//2)    # Pid object for yaw
pidForYaw.output_limits = (-100, 100)      # yawSpeed limits

pidForZaxis = PID(0.8, 0.0001, 0.0001, setpoint=h //
                  2)    # Pid object for Z axis
pidForZaxis.output_limits = (-100, 100)      # updownSpeed limits

pidForXaxis = PID(0.001, 0.00001, 0.0001, setpoint=(
    fbRange[1] + fbRange[0])//2)    # Pid object for X axis
pidForXaxis.output_limits = (-80, 80)      # forwardspeed limits

# Simple PID for Person + Bicycle
pidForYawBic = PID(0.12, 0.000001, 0.001, setpoint=w//2)    # Pid object for yaw
pidForYawBic.output_limits = (-100, 100)      # yawSpeed limits

pidForZaxisBic = PID(0.3, 0.00001, 0.0001, setpoint=h //
                  2)    # Pid object for Z axis
pidForZaxisBic.output_limits = (-100, 100)      # updownSpeed limits

pidForXaxisBic = PID(0.8, 0.00005, 0.001, setpoint=(
    fbRange[1] + fbRange[0])//2)    # Pid object for X axis
pidForXaxisBic.output_limits = (-80, 80)      # forwardspeed limits

optionMenu = 0


def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def Convertir_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


''' For color detection:    '''
# kernel window for morphological operations
kernel = np.ones((5, 5), np.uint8)
# Upper and lower limits for the color ORANGE in HSV color space
lower_yellow = np.array([0, 135, 150])
upper_yellow = np.array([20, 255, 255])


def colorDetection(img):
    cx, cy = -1, -1
    # Smooth the frame
    frameColorPicker = cv2.GaussianBlur(img, (11, 11), 0)
    # Convert to HSV color space
    hsv = cv2.cvtColor(frameColorPicker, cv2.COLOR_BGR2HSV)
    # Mask to extract just the yellow pixels
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # morphological opening
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # morphological closing
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    # Detect contours from the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    if (len(cnts) > 0):
        # Contour with greatest area
        c = max(cnts, key=cv2.contourArea)
        # Radius and center pixel coordinate of the largest contour
        # ((x,y),radius) = cv2.minEnclosingCircle(c)
        x, y, w, h = cv2.boundingRect(c)

        cx = ((w)//2) + x
        cy = ((h)//2) + y

        # Draw the center
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # display the image
    # cv2.imshow('frame', img)
    # Mask image
    # cv2.imshow('mask', mask)

    return cx, cy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str,
                        default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str,
                        default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str,
                        default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float,
                        default=0.8, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=1,
                        help="Is the video processed video? 1 = Yes, 0 == no")
    parser.add_argument("--nms_thres", type=float, default=0.4,
                        help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int,
                        default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416,
                        help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str,
                        help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str,
                        help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # if opt.webcam==1:
    #     cap = cv2.VideoCapture(0)
    #     # cap = donatello.get_frame_read().frame
    #     out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,960))
    # else:
    #     cap = cv2.VideoCapture(opt.directorio_video)
    #     # frame_width = int(cap.get(3))
    #     # frame_height = int(cap.get(4))
    #     out = cv2.VideoWriter('outp.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,960))
    
    # output 
    out = cv2.VideoWriter(f'output{datetime.timestamp(datetime.now())}.avi', cv2.VideoWriter_fourcc(*'MPEG'), 12, (w, h))
    
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a = []

    while True:
        # while cap:
        # ret, frame = cap.read()
        # if ret is False:
        # break

        # Get frames from Tello cam
        frame = donatello.get_frame_read().frame
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
        out.write(frame)

        # For color detection:
        colorCenterX, colorCenterY = colorDetection(frame)

        # LA imagen viene en Blue, Green, Red y la convertimos a RGB que es la entrada que requiere el modelo
        RGBimg = Convertir_RGB(frame)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))

        # Keyboard Control
        # print('getting key path')
        lr, fb, ud, yv, optionMenu = keyControl.getKeyboardInput(
            donatello, optionMenu, frame)
        if (optionMenu == 0):
            donatello.send_rc_control(lr, fb, ud, yv)

        # YOLO
        if (optionMenu == 1):
            with torch.no_grad():
                detections = model(imgTensor)
                detections = non_max_suppression(
                    detections, opt.conf_thres, opt.nms_thres)

            for detection in detections:
                if detection is not None:
                    detection = rescale_boxes(
                        detection, opt.img_size, RGBimg.shape[:2])
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                        box_w = x2 - x1
                        box_h = y2 - y1
                        color = [int(c) for c in colors[int(cls_pred)]]
                        # Only for person detection
                        if cls_pred == 0:
                            # print("Se detectó {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                            # frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 5)
                            cv2.rectangle(frame, (x1, y1 + box_h),
                                          (x2, y1), color, 3)
                            # Nombre de la clase detectada
                            cv2.putText(
                                frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                            # Certeza de prediccion de la clase
                            cv2.putText(frame, str("%.2f" % float(
                                conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3)
                            area = int(box_w*box_h)
                            c = [int(x1+box_w//2), int(y1+box_h//2)]
                            info = [c, area]
                            # print("c: {}, {}     area: {}".format(c[0], c[1], area))
                            # Shows the center of the rectangle/face
                            cv2.circle(frame, (c[0], c[1]),
                                       5, (0, 255, 0), cv2.FILLED)

                            # Some references:
                            # Shows the center of the frame
                            cv2.circle(frame, (w//2, h//2), 5,
                                       (255, 0, 0), cv2.FILLED)
                            cv2.rectangle(frame, (w//4, h//4),
                                          (3*w//4, 3*h//4), (255, 0, 0), 2)

                            # Color detection:
                            booleanExpresion = colorCenterX > int(x1) and colorCenterX < int(
                                x2) and colorCenterY > int(y1) and colorCenterY < int(y2)
                            # print("orangeCenter: ", colorCenterX, ", ", colorCenterY)
                            # print(booleanExpresion)
                            if booleanExpresion:
                                print("Persona con color naranja")
                                cv2.circle(
                                    frame, (c[0], c[1]), 15, (0, 0, 255), cv2.FILLED)
                                pidTello.track(
                                    donatello, pidForYaw, pidForZaxis, pidForXaxis, info, fbRange)
                            else:
                                print("No hay persona con color naranja")
                                donatello.send_rc_control(0, 0, 0, 0)
                        else:
                            print("No se detecto persona")
                            donatello.send_rc_control(0, 0, 0, 0)
        if(optionMenu == 2):
            with torch.no_grad():
                detections = model(imgTensor)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            existsPerson = False
            existsBicycle = False
            infoPerson = []
            centerPersons = []
            areaPersons = []

            infoBicycle = []
            centerBicycles = []
            areaBicycles = []
            xValuesBic = []

            for detection in detections:
                if detection is not None:
                    detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                        box_w = x2 - x1
                        box_h = y2 - y1
                        color = [int(c) for c in colors[int(cls_pred)]]
                        # Only for person detection

                        # Person
                        if cls_pred == 0:
                            existsPerson = True
                            arPer = int(box_w*box_h)
                            cenPer = [int(x1+box_w//2), int(y1+box_h//2)]
                            centerPersons.append(cenPer)
                            areaPersons.append(arPer)
                            infoPerson.append([cenPer, arPer])
                        # Bicycle
                        if cls_pred == 1:
                            existsBicycle = True
                            # areaBic = int(box_w*box_h)
                            areaBic = int(box_h)
                            cenBic = [int(x1+box_w//2), int(y1+box_h//2)]
                            centerBicycles.append(cenBic)
                            areaBicycles.append(areaBic)
                            infoBicycle.append([cenBic, areaBic])
                            xValuesBic.append([int(x1), int(x2)])


                        if cls_pred == 1 or cls_pred == 0:
                            # print("Se detectó {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                            frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 5)
                            cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 2)
                            cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)# Nombre de la clase detectada
                            cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2) # Certeza de prediccion de la clase
                            
                            area = int(box_w*box_h)
                            c = [int(x1+box_w//2), int(y1+box_h//2)]
                            info = [c, area]
                            # print("c: {}, {}     area: {}".format(c[0], c[1], area))
                            cv2.circle(frame, (c[0], c[1]), 5, (0, 255, 0), cv2.FILLED) # Shows the center of the rectangle/face

                            # Some references:
                            cv2.circle(frame, (w//2, h//2), 5, (200, 0, 255), cv2.FILLED) # Shows the center of the frame
                            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (200, 0, 255), 2)
                        else:
                            print("No se detecto persona u bici")
                            # print("Se detectó {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))

                    # Checks if the two objects exists in the same frame
                    isPair = existsPerson & existsBicycle
                    if isPair:
                        # Look for the biggest bicycle
                        i = areaBicycles.index(max(areaBicycles))
                        j = -1

                        for index in range(len(centerPersons)):
                            if centerPersons[index][0] > xValuesBic[i][0] and centerPersons[index][0] > xValuesBic[i][0]:
                                j = index
                                print("Persona y bici ALV ALV ALV")
                                print(centerPersons[j])
                                cv2.circle(frame, (centerBicycles[i][0], centerBicycles[i][1]), 20, (120, 0, 200), cv2.FILLED) # Shows the center of the bicycle
                                cv2.circle(frame, (centerPersons[j][0], centerPersons[j][1]), 20, (0, 0, 255), cv2.FILLED) # Shows the center of the tracked person
                                infoForPID = [(centerBicycles[i][0], centerBicycles[i][1]), areaBicycles[i]]
                                pidTello.track(
                                    donatello, pidForYawBic, pidForZaxisBic, pidForXaxisBic, infoForPID, fbRange)
                                print("Alto Bici: ", areaBicycles[i])
                                break
        # # BodyTracking
        # if(optionMenu == 2):
        #     # Look for the biggest body and track it
        #     img, info = bodyTracking3.findBody(img)
        #     pidTello.track(donatello, pidForYaw, pidForZaxis, pidForXaxis, info, fbRange)

        # Convertimos de vuelta a BGR para que cv2 pueda desplegarlo en los colores correctos
        # print("OPTION MODE: ", optionMenu)

        cv2.imshow('frame', Convertir_BGR(RGBimg))
        # if opt.webcam==1:
        #     cv2.imshow('frame', Convertir_BGR(RGBimg))
        #     out.write(RGBimg)
        # else:
        #     out.write(Convertir_BGR(RGBimg))
        #     cv2.imshow('frame', RGBimg)
        # cv2.waitKey(0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    out.release()
    # cap.release()
    cv2.destroyAllWindows()
